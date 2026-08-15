#!/usr/bin/env python3
"""
Experiment A — leak-free SL training dataset.

Builds an SL training CSV from `datasets/processed/multistate_combined_z.csv`
using a BENCHMARK-LEVEL split: entire benchmark URIs are assigned to
train/validation, and the 8 held-out evaluation cBench benchmarks are removed
from the file entirely (they must be absent from SL training).

Training benchmarks (27): all CHStone (12) + all csmith (9) + 6 non-eval
cBench (patricia, qsort, sha, susan, tiffdither, tiffmedian).

Held-out evaluation benchmarks (8, removed): cBench gsm, dijkstra, jpeg-c,
bzip2, tiff2rgba, tiff2bw, bitcount, stringsearch.

The per-(benchmark, state) z_runtime_improvement_pct column is kept as-is:
its z-statistics are computed within each (benchmark, state) group, so
removing whole benchmarks does not change the values of the retained groups.

Usage:
  python scripts/build_clean_sl_dataset.py \
    --output datasets/processed/multistate_clean_sl_z.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_URIS = [
    # 6 non-eval cBench
    "benchmark://cbench-v1/patricia",
    "benchmark://cbench-v1/qsort",
    "benchmark://cbench-v1/sha",
    "benchmark://cbench-v1/susan",
    "benchmark://cbench-v1/tiffdither",
    "benchmark://cbench-v1/tiffmedian",
    # 12 CHStone
    "benchmark://chstone-v0/adpcm",
    "benchmark://chstone-v0/aes",
    "benchmark://chstone-v0/blowfish",
    "benchmark://chstone-v0/dfadd",
    "benchmark://chstone-v0/dfdiv",
    "benchmark://chstone-v0/dfmul",
    "benchmark://chstone-v0/dfsin",
    "benchmark://chstone-v0/gsm",
    "benchmark://chstone-v0/jpeg",
    "benchmark://chstone-v0/mips",
    "benchmark://chstone-v0/motion",
    "benchmark://chstone-v0/sha",
    # 9 csmith
    "generator://csmith-v0/12",
    "generator://csmith-v0/17",
    "generator://csmith-v0/24",
    "generator://csmith-v0/26",
    "generator://csmith-v0/29",
    "generator://csmith-v0/4",
    "generator://csmith-v0/6",
    "generator://csmith-v0/8",
    "generator://csmith-v0/9",
]

# The 8 held-out evaluation benchmarks (must be COMPLETELY absent).
EVAL_URIS = [
    "benchmark://cbench-v1/gsm",
    "benchmark://cbench-v1/dijkstra",
    "benchmark://cbench-v1/jpeg-c",
    "benchmark://cbench-v1/bzip2",
    "benchmark://cbench-v1/tiff2rgba",
    "benchmark://cbench-v1/tiff2bw",
    "benchmark://cbench-v1/bitcount",
    "benchmark://cbench-v1/stringsearch",
]

VALIDATION_FRACTION = 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "datasets" / "processed" / "multistate_combined_z.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "datasets" / "processed" / "multistate_clean_sl_z.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    train_uris = set(TRAIN_URIS)
    eval_uris = set(EVAL_URIS)
    if len(train_uris) != len(TRAIN_URIS):
        raise SystemExit("Duplicate URIs in TRAIN_URIS")
    if len(eval_uris) != len(EVAL_URIS):
        raise SystemExit("Duplicate URIs in EVAL_URIS")
    overlap = train_uris & eval_uris
    if overlap:
        raise SystemExit(f"TRAIN and EVAL overlap: {sorted(overlap)}")

    present = {r["benchmark_uri"] for r in rows}
    missing_train = sorted(train_uris - present)
    missing_eval = sorted(eval_uris - present)
    if missing_train:
        print(f"WARNING: training URIs not in source CSV: {missing_train}", file=sys.stderr)
    if missing_eval:
        print(f"WARNING: eval URIs not in source CSV: {missing_eval}", file=sys.stderr)

    # Benchmark-level split: shuffle the TRAINING benchmark URIs (seed), assign
    # the first VALIDATION_FRACTION to validation, the rest to train. Whole
    # benchmarks move together — no benchmark appears in more than one split.
    rng = random.Random(args.seed)
    shuffled = list(train_uris)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VALIDATION_FRACTION))
    validation_uris = set(shuffled[:n_val])
    split_of = {
        uri: ("validation" if uri in validation_uris else "train")
        for uri in train_uris
    }

    kept: list = []
    dropped = 0
    for row in rows:
        uri = row["benchmark_uri"]
        if uri in eval_uris or uri not in train_uris:
            dropped += 1
            continue
        row["dataset_split"] = split_of[uri]
        kept.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    from collections import Counter
    split_counts = Counter(r["dataset_split"] for r in kept)
    per_bm = Counter(
        f"{r['benchmark_uri'].split('/')[-1]}:{r['dataset_split']}" for r in kept
    )
    print(
        f"Wrote {len(kept)} rows to {output_path} "
        f"(dropped {dropped} rows from eval/unknown benchmarks)"
    )
    print(f"split counts: {dict(split_counts)}")
    print(f"validation benchmarks: {sorted(validation_uris)}")
    print("per-benchmark splits:")
    for key in sorted(per_bm):
        print(f"  {key} x{per_bm[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
