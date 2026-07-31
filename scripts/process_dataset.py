#!/usr/bin/env python3
"""Clean, split, and normalize the NeuroCompiler transition dataset.

Stage 4 preserves the raw CSV, writes invalid/failed rows to an audit CSV,
splits by benchmark URI to prevent program leakage, and fits normalization
statistics using training benchmarks only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .extract_features import AUTOPHASE_FEATURE_NAMES
except ImportError:
    from extract_features import AUTOPHASE_FEATURE_NAMES  # type: ignore

LOGGER = logging.getLogger("neurocompiler.process_dataset")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "datasets" / "raw" / "pass_runtime_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: str) -> Optional[float]:
    if value is None or not str(value).strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _split_map(benchmarks: Sequence[str]) -> Dict[str, str]:
    """Assign whole benchmarks to stable 70/15/15-ish partitions."""

    ordered = sorted(
        set(benchmarks),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    count = len(ordered)
    if count == 0:
        return {}
    if count < 3:
        return {value: "train" for value in ordered}
    train_count = max(1, round(count * 0.70))
    val_count = max(1, round(count * 0.15))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    return {
        value: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + val_count
            else "test"
        )
        for index, value in enumerate(ordered)
    }


def _clean_runtime(row: Dict[str, str], prefix: str) -> int:
    """Remove sample-level runtime outliers using median absolute deviation."""

    raw = row.get(f"{prefix}runtime_samples_json", "")
    if not raw:
        return 0
    try:
        samples = [float(value) for value in json.loads(raw)]
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    samples = [value for value in samples if math.isfinite(value) and value >= 0]
    if len(samples) < 4:
        cleaned = samples
    else:
        median = statistics.median(samples)
        deviations = [abs(value - median) for value in samples]
        mad = statistics.median(deviations)
        cleaned = (
            samples
            if mad == 0
            else [value for value in samples if 0.6745 * abs(value - median) / mad <= 3.5]
        )
        if not cleaned:
            cleaned = samples
    removed = len(samples) - len(cleaned)
    if cleaned:
        row[f"{prefix}runtime_measurement_count"] = str(len(cleaned))
        row[f"{prefix}runtime_median_sec"] = repr(statistics.median(cleaned))
        row[f"{prefix}runtime_mean_sec"] = repr(statistics.fmean(cleaned))
        row[f"{prefix}runtime_std_sec"] = (
            repr(statistics.stdev(cleaned)) if len(cleaned) >= 2 else ""
        )
    return removed


def _normalization_columns(fieldnames: Sequence[str]) -> List[str]:
    candidates = [
        "pre_ir_instruction_count",
        "pre_object_text_size_bytes",
        "pre_total_basic_blocks",
        "pre_total_functions",
        "pre_total_instructions",
        "pre_total_memory_instructions",
    ] + [f"pre_autophase_{name}" for name in AUTOPHASE_FEATURE_NAMES]
    available = set(fieldnames)
    return [name for name in candidates if name in available]


def _fit_normalization(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for column in columns:
        values = [
            value
            for row in rows
            if (value := _finite_float(row.get(column, ""))) is not None
        ]
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) >= 2 else 0.0
        stats[column] = {
            "mean": mean,
            "std": std,
            "scale": std if std > 0 else 1.0,
            "count": float(len(values)),
        }
    return stats


def _rejection_reason(
    row: Mapping[str, str], *, require_runtime: bool = False
) -> str:
    if not _truthy(row.get("pass_success", "")):
        return row.get("error_type", "pass_failed") or "pass_failed"
    if not row.get("post_state_id", "").strip():
        return "missing_post_state"
    if _finite_float(row.get("step_reward", "")) is None:
        return "missing_or_invalid_reward"
    if _finite_float(row.get("pre_ir_instruction_count", "")) is None:
        return "missing_pre_instruction_count"
    if _finite_float(row.get("post_ir_instruction_count", "")) is None:
        return "missing_post_instruction_count"
    if require_runtime:
        if _finite_float(row.get("pre_runtime_median_sec", "")) is None:
            return "missing_pre_runtime"
        if _finite_float(row.get("post_runtime_median_sec", "")) is None:
            return "missing_post_runtime"
        if _finite_float(row.get("runtime_speedup", "")) is None:
            return "missing_runtime_speedup"
    for name in AUTOPHASE_FEATURE_NAMES:
        if _finite_float(row.get(f"pre_autophase_{name}", "")) is None:
            return f"missing_pre_autophase_{name}"
    return ""


def process_dataset(
    raw_path: Path, output_path: Path, *, require_runtime: bool = False
) -> Path:
    """Process one raw transition CSV into a training-ready CSV.

    Args:
        raw_path: Incrementally generated transition CSV.
        output_path: Destination training CSV.
        require_runtime: Reject transitions without valid pre/post runtimes.
            Enable this for a runtime-prediction dataset.
    """

    raw_path = Path(raw_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fields = list(reader.fieldnames or [])
        if "transition_key" not in raw_fields or "benchmark_uri" not in raw_fields:
            raise RuntimeError("Raw CSV is not a NeuroCompiler transition dataset")
        source_rows = list(reader)

    accepted: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    seen = set()
    duplicate_count = 0
    runtime_outliers_removed = 0

    for source in source_rows:
        row = dict(source)
        key = row.get("transition_key", "")
        if not key or key in seen:
            duplicate_count += 1
            rejected.append({**row, "rejection_reason": "duplicate_or_missing_key"})
            continue
        seen.add(key)
        reason = _rejection_reason(row, require_runtime=require_runtime)
        if reason:
            rejected.append({**row, "rejection_reason": reason})
            continue
        runtime_outliers_removed += _clean_runtime(row, "pre_")
        runtime_outliers_removed += _clean_runtime(row, "post_")
        accepted.append(row)

    if not accepted:
        raise RuntimeError("No valid rows remain after cleaning")

    splits = _split_map([row["benchmark_uri"] for row in accepted])
    for row in accepted:
        row["dataset_split"] = splits[row["benchmark_uri"]]

    train_rows = [row for row in accepted if row["dataset_split"] == "train"]
    normalization_columns = _normalization_columns(raw_fields)
    normalization = _fit_normalization(train_rows, normalization_columns)
    normalized_fields = [f"norm_{column}" for column in normalization]

    for row in accepted:
        for column, params in normalization.items():
            value = _finite_float(row.get(column, ""))
            row[f"norm_{column}"] = (
                "" if value is None else repr((value - params["mean"]) / params["scale"])
            )

    output_fields = raw_fields + ["dataset_split"] + normalized_fields
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in output_fields} for row in accepted
        )

    rejected_path = output_path.with_name(f"{output_path.stem}_rejected.csv")
    with rejected_path.open("w", newline="", encoding="utf-8") as handle:
        fields = raw_fields + ["rejection_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rejected
        )

    normalization_path = output_path.with_name("normalization.json")
    normalization_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "fit_partition": "train",
                "method": "zscore",
                "columns": normalization,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    split_counts = {
        split: sum(row["dataset_split"] == split for row in accepted)
        for split in ("train", "validation", "test")
    }
    manifest_path = output_path.with_name("dataset_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_csv": str(raw_path),
                "output_csv": str(output_path),
                "source_rows": len(source_rows),
                "accepted_rows": len(accepted),
                "rejected_rows": len(rejected),
                "duplicate_rows": duplicate_count,
                "runtime_sample_outliers_removed": runtime_outliers_removed,
                "split_rows": split_counts,
                "split_unit": "benchmark_uri",
                "normalization_fit_partition": "train",
                "require_runtime": require_runtime,
                "primary_runtime_targets": [
                    "runtime_reduction_sec",
                    "runtime_speedup",
                    "runtime_improvement_pct",
                ],
                "normalization_file": str(normalization_path),
                "rejected_file": str(rejected_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    LOGGER.info(
        "Processed dataset: accepted=%d rejected=%d splits=%s output=%s",
        len(accepted),
        len(rejected),
        split_counts,
        output_path,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the NeuroCompiler raw CSV.")
    parser.add_argument("--input", default=str(DEFAULT_RAW))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Keep only rows with valid pre/post runtime measurements.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    process_dataset(
        Path(args.input), Path(args.output), require_runtime=args.require_runtime
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
