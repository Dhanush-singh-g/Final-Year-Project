#!/usr/bin/env python3
"""
Add per-benchmark z-scored runtime improvement to a processed dataset.

Research motivation (see scripts/reward.py: per_benchmark_zscore): the raw
``runtime_improvement_pct`` target is incomparable across programs — each
benchmark's candidate-pass runtime distribution has a different mean and
scale (e.g. gsm mean -21% vs another benchmark +57%). A scorer trained on raw
values can learn benchmark identity instead of pass quality. This script
writes a copy of the dataset with a ``z_runtime_improvement_pct`` column
(blank where no runtime or too few candidates), ready to train the SL model
with ``--target z_runtime_improvement_pct``.

Usage:
  python scripts/zscore_dataset.py \\
    --input datasets/processed/hybrid_dataset_scaled.csv \\
    --output datasets/processed/hybrid_dataset_scaled_z.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reward import add_zscored_runtime_column  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Processed dataset CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--group-col", default="benchmark_uri",
        help="Column to z-score within (default benchmark_uri; use "
        "pre_state_id for per-(benchmark, state) groups).",
    )
    parser.add_argument(
        "--value-col", default="runtime_improvement_pct",
        help="Column holding the value to z-score (default "
        "runtime_improvement_pct; RL replay buffers use runtime_improvement).",
    )
    parser.add_argument(
        "--sync-to", default=None,
        help="Optional column to also overwrite with the z-score (e.g. "
        "hybrid_reward) so RL training consumes the z-scored target.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames or not rows:
        raise SystemExit(f"No rows or fields in {input_path}")

    if args.value_col not in fieldnames:
        raise SystemExit(
            f"{input_path} has no {args.value_col} column; "
            "cannot z-score runtime."
        )

    out_col = "z_runtime_improvement_pct"
    if out_col not in fieldnames:
        fieldnames.append(out_col)
    converted = add_zscored_runtime_column(
        rows, value_col=args.value_col, out_col=out_col, group_col=args.group_col
    )
    if args.sync_to:
        if args.sync_to not in fieldnames:
            fieldnames.append(args.sync_to)
        for row in converted:
            row[args.sync_to] = row.get(out_col, "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted)

    filled = sum(1 for row in converted if (row.get(out_col) or "").strip())
    print(
        f"Wrote {len(converted)} rows to {output_path} "
        f"({filled} with {out_col}; {len(converted) - filled} blank)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
