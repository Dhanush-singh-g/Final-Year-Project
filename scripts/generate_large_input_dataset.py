#!/usr/bin/env python3
"""
Generate a pass-quality SL dataset with runtime measured on a LARGE input.

Research motivation: the z-scored experiment showed the pass-quality signal is
not learnable from sub-10 ms default inputs (process startup dominates), so
the runtime z-scores carry no signal. This script measures each curated pass
variant natively on an explicit large input using the O3 harness's own
input-resolution and timing machinery, producing rows with a meaningful
per-pass runtime delta vs the unoptimised (O0) state.

Usage:
  python scripts/generate_large_input_dataset.py \\
    --benchmark benchmark://cbench-v1/gsm --inputs 11 \\
    --output datasets/processed/gsm_large_input_passes.csv

Column set is intentionally small (the SL trainer only needs pass_flag,
benchmark_uri, pre/post runtime + IR); extend if a different consumer needs
more features.

Runtime cost guidance: one native build + (warmup + runs) timed executions per
curated pass, per benchmark. Pick an input whose runtime is 0.2-2 s so the
whole 31-pass sweep fits in a bounded window (e.g. gsm 2.au ~0.6 s).
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.o3_runtime_harness import (  # type: ignore
    _command_args,
    _command_outfile,
    _pre_run_shells,
    build_native,
    resolve_input,
    timed_run,
)
from scripts.curated_passes import get_curated_flags  # type: ignore
from scripts.extract_features import MeasurementConfig, extract_features  # type: ignore

LOGGER = logging.getLogger("generate_large_input_dataset")

OUT_COLUMNS = [
    "benchmark_uri",
    "pass_flag",
    "runtime_improvement_pct",
    "o0_runtime_median_sec",
    "runtime_median_sec",
    "warmup",
    "runs",
]


def generate(benchmark_uri: str, input_index: int, args: argparse.Namespace) -> Path:
    import compiler_gym

    env = compiler_gym.make("llvm-v0")
    benchmark = env.datasets.benchmark(benchmark_uri)
    o0_bc = bytes(benchmark.proto.program.contents)
    dc = benchmark.proto.dynamic_config
    build_args = _command_args(dc.build_cmd)
    run_args = _command_args(dc.run_cmd)
    pre_cmds = _pre_run_shells(dc.pre_run_cmd)
    outfile = _command_outfile(dc.build_cmd)

    if not run_args:
        raise SystemExit(
            f"{benchmark_uri} has no dynamic run configuration; this generator "
            "needs a runnable benchmark (cBench with real inputs)."
        )

    workdir = Path(args.workdir) / benchmark_uri.split("/")[-1]
    workdir.mkdir(parents=True, exist_ok=True)

    run_args_i, pre_cmds_i, input_info = resolve_input(
        run_args, pre_cmds, input_index
    )
    run_cmd_i = " ".join(run_args_i)
    LOGGER.info(
        "Input %s: %s (runs ~%s)",
        input_index,
        input_info["input_file"],
        input_info.get("input_size", "?"),
    )

    # 1. O0 baseline: build once, time on the large input.
    o0_dir = workdir / "o0"
    build_native(o0_bc, o0_dir, build_args, outfile, args.timeout)
    o0_samples: List[float] = []
    for _ in range(args.warmup):
        timed_run(run_cmd_i, pre_cmds_i, o0_dir, args.cpu, args.timeout)
    for _ in range(args.runs):
        elapsed, rc, stdout_text, _ = timed_run(
            run_cmd_i, pre_cmds_i, o0_dir, args.cpu, args.timeout
        )
        if rc != 0:
            raise SystemExit(
                f"O0 run failed rc={rc}: {stdout_text[:200]!r}"
            )
        o0_samples.append(elapsed)
    o0_med = statistics.median(o0_samples)
    LOGGER.info("O0 baseline median %.4f s", o0_med)

    # 2. Extract the full pre-state feature set once at O0 (the SL scorer
    #    needs pre_* features per row; each pass variant shares this state).
    measurement = MeasurementConfig(
        measure_runtime=False,
        measure_buildtime=False,
        collect_object_text_size=True,
    )
    env.reset(benchmark=benchmark_uri)
    initial_features = extract_features(env, measurement)
    state_row = initial_features.flattened("pre_")
    # Keep only the deterministic feature columns the SL trainer consumes.
    state_row = {
        k: v
        for k, v in state_row.items()
        if k.startswith("pre_autophase_") or k in (
            "pre_ir_instruction_count",
            "pre_object_text_size_bytes",
            "pre_total_basic_blocks",
            "pre_total_functions",
            "pre_total_instructions",
            "pre_total_memory_instructions",
        )
    }

    # 3. Each curated pass applied once to O0, built, timed on the same input.
    rows: List[Dict[str, str]] = []
    flags = get_curated_flags()
    for pass_index, flag in enumerate(flags, start=1):
        started = time.perf_counter()
        try:
            env.reset(benchmark=benchmark_uri)
            _, _, done, _ = env.step(env.action_space.from_string(flag))
            bc_raw = env.observation["Bitcode"]
            if hasattr(bc_raw, "tobytes"):
                bc_raw = bc_raw.tobytes()
            bc_bytes = bc_raw if isinstance(bc_raw, bytes) else bytes(bc_raw)
        except Exception as error:
            LOGGER.warning("Pass %s failed (%s); skipping", flag, error)
            continue

        pass_dir = workdir / f"pass_{pass_index:02d}"
        try:
            build_native(bc_bytes, pass_dir, build_args, outfile, args.timeout)
        except Exception as error:
            LOGGER.warning("Build failed for %s (%s); skipping", flag, error)
            continue

        samples: List[float] = []
        for _ in range(args.warmup):
            timed_run(run_cmd_i, pre_cmds_i, pass_dir, args.cpu, args.timeout)
        failed = False
        for _ in range(args.runs):
            elapsed, rc, stdout_text, _ = timed_run(
                run_cmd_i, pre_cmds_i, pass_dir, args.cpu, args.timeout
            )
            if rc != 0:
                LOGGER.warning("Run failed for %s rc=%d; skipping", flag, rc)
                failed = True
                break
            samples.append(elapsed)
        if failed or not samples:
            continue

        med = statistics.median(samples)
        improvement = (
            100.0 * (o0_med - med) / o0_med if o0_med > 0 else 0.0
        )
        row = {
            "benchmark_uri": benchmark_uri,
            "pass_flag": flag,
            "runtime_improvement_pct": f"{improvement:.4f}",
            "o0_runtime_median_sec": f"{o0_med:.6f}",
            "runtime_median_sec": f"{med:.6f}",
            "warmup": str(args.warmup),
            "runs": str(args.runs),
        }
        row.update(state_row)
        rows.append(row)
        LOGGER.info(
            "[%d/%d] %-22s med %.4f s  imp %+.2f%%  (%.1f s)",
            pass_index, len(flags), flag, med, improvement,
            time.perf_counter() - started,
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fieldnames are dynamic: the pre-state feature columns come from the
    # extracted state row, not a static list.
    fieldnames = list(OUT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} pass rows to {output_path} (of {len(flags)} passes)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Runnable benchmark URI")
    parser.add_argument("--inputs", type=int, default=-1, help="Input index (or -1 = largest)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--workdir", default=str(PROJECT_ROOT / "results" / "large_input_work"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    generate(args.benchmark, args.inputs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
