#!/usr/bin/env python3
"""Generate the NeuroCompiler raw pass-transition dataset.

Stage 3 and the main pipeline entry point. By default, each LLVM pass is applied
independently from the benchmark's initial state. Rows are appended and flushed
incrementally, and completed transition keys are skipped on resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    from .extract_features import MeasurementConfig, ProgramFeatures, extract_features
    from .run_passes import (
        ActionMetadata,
        Transition,
        resolve_actions,
        run_pass_sequence,
        transition_fieldnames,
    )
except ImportError:
    from extract_features import MeasurementConfig, ProgramFeatures, extract_features  # type: ignore
    from run_passes import (  # type: ignore
        ActionMetadata,
        Transition,
        resolve_actions,
        run_pass_sequence,
        transition_fieldnames,
    )

LOGGER = logging.getLogger("neurocompiler.generate_dataset")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "datasets" / "raw" / "pass_runtime_dataset.csv"
DEFAULT_PROCESSED_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"
)

PROVENANCE_FIELDS = [
    "transition_key",
    "run_id",
    "generated_at_utc",
    "benchmark_suite",
    "compiler_gym_version",
    "compiler_version",
    "host_name",
    "python_version",
]
CSV_FIELDS = PROVENANCE_FIELDS + transition_fieldnames()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_uri(value: str) -> str:
    value = value.strip().rstrip("/")
    return value if "://" in value else f"benchmark://{value}"


def enumerate_benchmarks(
    env: Any,
    dataset_uri: str,
    explicit: Sequence[str],
    limit: Optional[int],
) -> List[str]:
    """Return a stable, optionally limited benchmark URI list."""

    if explicit:
        uris = [
            item if "://" in item else f"{dataset_uri}/{item.lstrip('/')}"
            for item in explicit
        ]
    else:
        dataset = env.datasets[dataset_uri]
        uris = sorted(str(uri) for uri in dataset.benchmark_uris())
    if limit is not None:
        uris = uris[:limit]
    if not uris:
        raise RuntimeError(f"No benchmarks selected from {dataset_uri}")
    return uris


def _parse_passes(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _transition_key(
    benchmark_uri: str,
    pre_state_id: str,
    action_id: int,
    reward_space: str,
    measurement: MeasurementConfig,
) -> str:
    payload = {
        "schema": "1.0.0",
        "benchmark_uri": benchmark_uri,
        "pre_state_id": pre_state_id,
        "pass_id": action_id,
        "pass_position": 0,
        "previous_pass_sequence": [],
        "reward_space": reward_space,
        "measurement": asdict(measurement),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_completed_keys(path: Path) -> Set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                "Existing raw CSV schema does not match this pipeline version. "
                "Move or rename the old file before starting a new run."
            )
        return {row["transition_key"] for row in reader if row.get("transition_key")}


def _failure_transition(
    pre: ProgramFeatures,
    action: ActionMetadata,
    reward_space: str,
    error: Exception,
) -> Transition:
    return Transition(
        benchmark_uri=pre.benchmark_uri,
        action=action,
        reward_space=reward_space,
        pass_position=0,
        previous_pass_sequence=(),
        pre=pre,
        post=None,
        step_reward=None,
        cumulative_reward=None,
        pass_success=False,
        action_had_no_effect=None,
        done=True,
        step_walltime_sec=0.0,
        error_type=type(error).__name__,
        error_message=str(error).replace("\n", " ")[:2000],
    )


def _write_manifest(
    output_path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
    benchmark_count: int,
    action_count: int,
    compiler_gym_version: str,
    compiler_version: str,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "updated_at_utc": _utc_now(),
        "raw_csv": str(output_path),
        "dataset": _dataset_uri(args.dataset),
        "benchmark_count_selected": benchmark_count,
        "action_count_selected": action_count,
        "reward_space": args.reward_space,
        "measurement": {
            "measure_runtime": args.measure_runtime,
            "require_runtime_in_processed_dataset": args.require_runtime,
            "runtime_count": args.runtime_count,
            "runtime_warmup_count": args.runtime_warmup_count,
            "measure_buildtime": args.measure_buildtime,
            "collect_object_text_size": not args.skip_object_text_size,
        },
        "compiler_gym_version": compiler_gym_version,
        "compiler_version": compiler_version,
        "python_version": platform.python_version(),
        "host_name": socket.gethostname(),
        "command": sys.argv,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> Path:
    """Execute raw dataset generation and return the output path."""

    try:
        import compiler_gym
    except ImportError as error:
        raise SystemExit(
            "CompilerGym is unavailable. Activate the 'neurocompiler' Conda "
            "environment before running this program."
        ) from error

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path) if args.resume else set()
    if not args.resume and output_path.exists():
        output_path.unlink()

    measurement = MeasurementConfig(
        measure_runtime=args.measure_runtime,
        runtime_count=args.runtime_count,
        runtime_warmup_count=args.runtime_warmup_count,
        measure_buildtime=args.measure_buildtime,
        collect_object_text_size=not args.skip_object_text_size,
    )
    run_id = uuid.uuid4().hex
    dataset_uri = _dataset_uri(args.dataset)
    generated_at = _utc_now()
    compiler_gym_version = str(getattr(compiler_gym, "__version__", "0.2.5"))

    env = compiler_gym.make("llvm-v0")
    try:
        compiler_version = str(getattr(env, "compiler_version", "unknown"))
        benchmarks = enumerate_benchmarks(
            env, dataset_uri, args.benchmark, args.max_benchmarks
        )
        actions = resolve_actions(env, _parse_passes(args.passes))
        if args.max_passes is not None:
            actions = actions[: args.max_passes]
        if not actions:
            raise RuntimeError("No LLVM passes selected")

        _write_manifest(
            output_path,
            run_id=run_id,
            args=args,
            benchmark_count=len(benchmarks),
            action_count=len(actions),
            compiler_gym_version=compiler_gym_version,
            compiler_version=compiler_version,
        )

        file_exists = output_path.exists() and output_path.stat().st_size > 0
        processed = skipped = failed = 0
        total = len(benchmarks) * len(actions)
        started_all = time.perf_counter()

        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
            if not file_exists:
                writer.writeheader()
                handle.flush()

            for benchmark_index, benchmark_uri in enumerate(benchmarks, start=1):
                LOGGER.info(
                    "Benchmark %d/%d: %s", benchmark_index, len(benchmarks), benchmark_uri
                )
                try:
                    env.reset(
                        benchmark=benchmark_uri,
                        reward_space=args.reward_space,
                        timeout=args.timeout,
                    )
                    baseline = extract_features(env, measurement)
                except Exception as error:
                    failed += len(actions)
                    LOGGER.exception("Skipping benchmark %s: %s", benchmark_uri, error)
                    continue

                for action_index, action in enumerate(actions, start=1):
                    key = _transition_key(
                        benchmark_uri,
                        baseline.state_id,
                        action.action_id,
                        args.reward_space,
                        measurement,
                    )
                    if key in completed:
                        skipped += 1
                        continue

                    LOGGER.info(
                        "Running %d/%d benchmark=%s pass=%s (%d/%d)",
                        processed + skipped + failed + 1,
                        total,
                        benchmark_uri,
                        action.flag,
                        action_index,
                        len(actions),
                    )
                    try:
                        # Every initial census row is an independent O0 -> pass
                        # transition, preventing cross-pass state contamination.
                        env.reset(
                            benchmark=benchmark_uri,
                            reward_space=args.reward_space,
                            timeout=args.timeout,
                        )
                        transitions = run_pass_sequence(
                            env,
                            [action],
                            reward_space=args.reward_space,
                            measurement=measurement,
                            initial_features=baseline,
                            timeout_sec=args.timeout,
                        )
                        transition = transitions[0]
                    except Exception as error:
                        LOGGER.exception(
                            "Experiment failed benchmark=%s pass=%s",
                            benchmark_uri,
                            action.flag,
                        )
                        transition = _failure_transition(
                            baseline, action, args.reward_space, error
                        )

                    row = {
                        "transition_key": key,
                        "run_id": run_id,
                        "generated_at_utc": generated_at,
                        "benchmark_suite": dataset_uri,
                        "compiler_gym_version": compiler_gym_version,
                        "compiler_version": compiler_version,
                        "host_name": socket.gethostname(),
                        "python_version": platform.python_version(),
                    }
                    row.update(transition.to_row())
                    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
                    handle.flush()
                    if args.fsync:
                        os.fsync(handle.fileno())
                    completed.add(key)
                    processed += 1
                    if not transition.pass_success:
                        failed += 1

        elapsed = time.perf_counter() - started_all
        LOGGER.info(
            "Raw generation complete: new=%d resumed=%d failed=%d elapsed=%.1fs output=%s",
            processed,
            skipped,
            failed,
            elapsed,
            output_path,
        )
        return output_path
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the complete NeuroCompiler pass-transition dataset."
    )
    parser.add_argument("--dataset", default="cbench-v1")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Specific benchmark URI/name; repeat for multiple programs.",
    )
    parser.add_argument("--max-benchmarks", type=int)
    parser.add_argument(
        "--passes",
        help=(
            "Comma-separated pass IDs, names, or flags. For flags beginning '-' "
            "use --passes=-adce,-sroa. Default: every available LLVM action."
        ),
    )
    parser.add_argument("--max-passes", type=int)
    parser.add_argument("--reward-space", default="IrInstructionCountO3")
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help=(
            "When processing, keep only transitions with valid pre/post runtime "
            "measurements. Requires --measure-runtime."
        ),
    )
    parser.add_argument("--runtime-count", type=int, default=5)
    parser.add_argument("--runtime-warmup-count", type=int, default=1)
    parser.add_argument("--measure-buildtime", action="store_true")
    parser.add_argument("--skip-object-text-size", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument(
        "--process",
        action="store_true",
        help="After generation, produce datasets/processed/hybrid_dataset.csv.",
    )
    parser.add_argument("--processed-output", default=str(DEFAULT_PROCESSED_PATH))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.require_runtime and not args.measure_runtime:
        raise SystemExit("--require-runtime requires --measure-runtime")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    raw_path = generate(args)
    if args.process:
        try:
            from .process_dataset import process_dataset
        except ImportError:
            from process_dataset import process_dataset  # type: ignore
        process_dataset(
            raw_path,
            Path(args.processed_output).expanduser().resolve(),
            require_runtime=args.require_runtime,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
