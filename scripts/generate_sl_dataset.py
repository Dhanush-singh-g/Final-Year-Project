#!/usr/bin/env python3
"""
Generate Supervised Learning transition dataset (Phase 3).

This is a curated wrapper around generate_dataset.py that uses the 27-pass
recommended set from curated_passes.py instead of all available passes.

Pipeline per benchmark:
  Load Benchmark -> Extract Initial Features -> Apply ONE Pass -> Extract New Features
  -> Measure Reward -> Save Transition -> Reset -> Next Pass

Every row: State_before, Optimization_pass, Reward, State_after

Total size estimate:
  cBench 23 * 27 = 621
  LLVM Test Suite 500 * 27 = 13500
  PolyBench 30 * 27 = 810
  AnghaBench 5000 * 27 = 135000
  Total ~150k+ supervised samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from curated_passes import get_curated_flags  # noqa: E402
from generate_dataset import generate, _parse_args as _base_parse  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SL dataset with curated 27-pass set (NeuroCompiler Phase 3)."
    )
    parser.add_argument("--dataset", default="cbench-v1", help="CompilerGym dataset URI")
    parser.add_argument("--benchmark", action="append", default=[], help="Specific benchmark")
    parser.add_argument("--max-benchmarks", type=int, default=None)
    parser.add_argument("--passes", default=None, help="Override curated passes: comma separated flags. Default: curated 27")
    parser.add_argument("--max-passes", type=int, default=None)
    parser.add_argument("--reward-space", default="IrInstructionCountO3")
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--runtime-count", type=int, default=5)
    parser.add_argument("--runtime-warmup-count", type=int, default=1)
    parser.add_argument("--measure-buildtime", action="store_true")
    parser.add_argument("--skip-object-text-size", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "datasets" / "raw" / "sl_transitions.csv"))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument("--process", action="store_true", help="Auto-process into hybrid_dataset.csv")
    parser.add_argument("--processed-output", default=str(PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.require_runtime and not args.measure_runtime:
        raise SystemExit("--require-runtime requires --measure-runtime")

    # If no passes specified, inject curated set
    if args.passes is None:
        curated = ",".join(get_curated_flags())
        args.passes = curated
        print(f"[SL] Using curated {len(get_curated_flags())} passes: {curated}")

    # Reuse generate() from generate_dataset.py
    # It expects attributes: dataset, benchmark, max_benchmarks, passes, max_passes, reward_space,
    # measure_runtime, require_runtime, runtime_count, runtime_warmup_count, measure_buildtime,
    # skip_object_text_size, timeout, output, resume, fsync, process, processed_output, log_level
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw_path = generate(args)

    if args.process:
        from process_dataset import process_dataset
        process_dataset(
            raw_path,
            Path(args.processed_output).expanduser().resolve(),
            require_runtime=args.require_runtime,
        )

    print(f"[SL] Done. Raw: {raw_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
