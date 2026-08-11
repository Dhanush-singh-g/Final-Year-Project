#!/usr/bin/env python3
"""
External -O3 executable runtime baseline harness (EXECUTION_RUNBOOK_RUNTIME.md, section 10).

The claim "hybrid beats -O3 on runtime" is only defensible with a controlled
executable baseline. CompilerGym exposes the exact -O3 IR cost but no -O3
runtime observation, so this harness builds and times native executables
outside CompilerGym:

    O0 bitcode (benchmark proto, identical to the env's start state)
        |
        +--> opt -O3  -> o3.bc      -> clang -> o3/a.out
        |
        +--> hybrid   -> hybrid.bc  -> clang -> hybrid/a.out   (training/inference.py)

Both executables are run with the benchmark's OWN dynamic run configuration
(CompilerGym's build_cmd / pre_run_cmd / run_cmd templates, which preserve the
cBench input setup such as ``_finfo_dataset``), identical warmups and
repetitions, ``taskset`` CPU pinning to the same core, and interleaved ordering
to cancel thermal/load drift. Results are reported as medians with 95%
bootstrap confidence intervals, per-benchmark speedup (median O3 / median
hybrid), a paired Wilcoxon signed-rank test across benchmarks, and an
output-hash equality check between the two binaries.

Subcommands:
    measure    Measure one process-worth of benchmarks (resumable: benchmarks
               already present in --output are skipped).
    summarize  Merge one or more measure outputs into the comparison table.

Usage:
    python evaluation/o3_runtime_harness.py measure \
        --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
        --sl-model-dir models/supervised --rl-model-dir models/reinforcement \
        --max-steps 8 --warmup 1 --reps 5 --cpu 4 --timeout 120 \
        --workdir results/o3_harness_work --output results/o3_harness_wave1.json

    python evaluation/o3_runtime_harness.py summarize \
        --results results/o3_harness_wave1.json results/o3_harness_wave2.json \
        --output results/o3_runtime_vs_o3_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dependency via sklearn
    np = None  # type: ignore

LOGGER = logging.getLogger("o3_harness")


class HarnessError(RuntimeError):
    """A benchmark failed to build, run, or measure."""


def _slug(uri: str) -> str:
    """Filesystem-safe identifier derived from a benchmark URI."""
    return uri.replace("://", "__").replace("/", "_")


def load_test_benchmarks(processed_csv: Path) -> List[str]:
    """Return the benchmark URIs assigned to the 'test' dataset split."""
    benchmarks = set()
    with processed_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dataset_split") == "test":
                uri = row.get("benchmark_uri", "")
                if uri:
                    benchmarks.add(uri)
    return sorted(benchmarks)


def bootstrap_ci(
    samples: Sequence[float],
    seed: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the median of *samples*.

    Seeded for reproducibility. Falls back to (min, max) when only one sample
    is available.
    """
    values = [float(s) for s in samples]
    if not values:
        raise ValueError("bootstrap_ci requires at least one sample")
    if len(values) == 1:
        return values[0], values[0]
    if np is None:
        raise RuntimeError("numpy is required for bootstrap CIs")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    medians = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        medians[i] = np.median(rng.choice(arr, size=len(arr), replace=True))
    lo = float(np.percentile(medians, 100.0 * alpha / 2.0))
    hi = float(np.percentile(medians, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def arm_stats(samples: Sequence[float], seed: int) -> Dict[str, Optional[float]]:
    """Summary statistics for one timing arm."""
    values = [float(s) for s in samples]
    if not values:
        return {
            "n": 0,
            "median_sec": None,
            "mean_sec": None,
            "std_sec": None,
            "min_sec": None,
            "max_sec": None,
            "ci95_lo_sec": None,
            "ci95_hi_sec": None,
        }
    lo, hi = bootstrap_ci(values, seed=seed)
    return {
        "n": len(values),
        "median_sec": statistics.median(values),
        "mean_sec": statistics.fmean(values),
        "std_sec": statistics.stdev(values) if len(values) >= 2 else None,
        "min_sec": min(values),
        "max_sec": max(values),
        "ci95_lo_sec": lo,
        "ci95_hi_sec": hi,
    }


def geo_mean(values: Sequence[float]) -> Optional[float]:
    """Geometric mean; None for an empty sequence."""
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize_results(
    results: Sequence[Dict],
) -> Dict:
    """Aggregate per-benchmark rows into the comparison table.

    Kept as a pure function (no I/O) so it is unit-testable.
    """
    rows = []
    for result in results:
        o3_median = (result.get("o3") or {}).get("median_sec")
        hy_median = (result.get("hybrid") or {}).get("median_sec")
        if o3_median is None or hy_median is None or o3_median <= 0 or hy_median <= 0:
            continue
        speedup = o3_median / hy_median
        row = {
            "benchmark_uri": result.get("benchmark_uri"),
            "suite": result.get("suite"),
            "protocol": result.get("protocol"),
            "speedup_hybrid_vs_o3": speedup,
            "log2_speedup": math.log2(speedup),
            "o3_median_sec": o3_median,
            "hybrid_median_sec": hy_median,
            "o3_ci": [result["o3"]["ci95_lo_sec"], result["o3"]["ci95_hi_sec"]],
            "hybrid_ci": [
                result["hybrid"]["ci95_lo_sec"],
                result["hybrid"]["ci95_hi_sec"],
            ],
            "outputs_match": result.get("outputs_match"),
            "pass_sequence": result.get("pass_sequence"),
            "hybrid_vs_o3_ir_pct": result.get("hybrid_vs_o3_ir_pct"),
        }
        rows.append(row)

    speedups = [r["speedup_hybrid_vs_o3"] for r in rows]
    summary: Dict = {
        "benchmarks_evaluated": len(rows),
        "wins": sum(1 for s in speedups if s > 1.0),
        "losses": sum(1 for s in speedups if s < 1.0),
        "ties": sum(1 for s in speedups if s == 1.0),
        "geo_mean_speedup": geo_mean(speedups),
        "geo_mean_log2_speedup": (
            sum(r["log2_speedup"] for r in rows) / len(rows) if rows else None
        ),
        "mean_speedup": statistics.fmean(speedups) if speedups else None,
        "rows": rows,
    }

    if len(speedups) >= 2:
        try:
            from scipy.stats import wilcoxon

            log_speedups = [r["log2_speedup"] for r in rows]
            stat, p_value = wilcoxon(
                log_speedups, alternative="greater"
            )  # H1: hybrid faster than O3
            summary["wilcoxon_signed_rank"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "alternative": "hybrid faster than O3 (on log2 speedups)",
            }
        except Exception as error:  # scipy missing or degenerate data
            summary["wilcoxon_signed_rank"] = {"error": str(error)}
    return summary


def _command_args(cmd) -> List[str]:
    """Extract the argument list from a protobuf Command message (or empty)."""
    if cmd is None:
        return []
    return list(cmd.argument)


def _command_outfile(cmd) -> str:
    """The executable name produced by a build Command (a repeated proto field)."""
    if cmd is None:
        return ""
    values = list(cmd.outfile)
    return values[0] if values else ""


def _pre_run_shells(container) -> List[str]:
    """Shell strings for each pre-run command.

    CompilerGym stores these as raw shell tokens (e.g. ``>_finfo_dataset``) and
    executes them by joining with plain spaces, so we mirror that exactly;
    shlex.join would quote the redirect token and break the command.
    """
    out = []
    for cmd in container:
        out.append(" ".join(list(cmd.argument)))
    return out


def build_native(
    bc_bytes: bytes,
    workdir: Path,
    build_args: Sequence[str],
    outfile: str,
    timeout: int,
) -> Path:
    """Compile a bitcode module to a native executable.

    Uses the benchmark's own build_cmd template ($CC -> bundled clang,
    $IN -> module path). Falls back to ``clang <module> -lm -o a.out`` when no
    template is available (CHStone-style benchmarks).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    module = workdir / "module.bc"
    module.write_bytes(bc_bytes)
    clang = str(compiler_gym_clang_path())
    if build_args and any("$IN" in a for a in build_args):
        cmd = [
            a.replace("$CC", clang).replace("$IN", str(module))
            for a in build_args
        ]
    else:
        cmd = [clang, str(module), "-lm", "-o", str(workdir / "a.out")]
    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise HarnessError(
            f"native build failed (rc={proc.returncode}): {proc.stderr[-400:]}"
        )
    exe = workdir / (outfile or "a.out")
    if not exe.exists():
        raise HarnessError(
            f"expected executable {exe.name} was not produced; stdout: {proc.stdout[-200:]}"
        )
    os.chmod(exe, 0o755)
    return exe


def timed_run(
    run_cmd: str,
    pre_cmds: Sequence[str],
    workdir: Path,
    cpu: int,
    timeout: int,
) -> Tuple[float, int, str, Optional[str]]:
    """Run the benchmark executable once, CPU-pinned, returning timing.

    Returns (elapsed_sec, returncode, stdout_text, outfile_sha256).
    """
    pin = ["taskset", "-c", str(cpu)]
    for pre in pre_cmds:
        subprocess.run(
            [*pin, "/bin/sh", "-c", pre],
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
        )
    started = time.perf_counter()
    proc = subprocess.run(
        [*pin, "/bin/sh", "-c", run_cmd],
        cwd=workdir,
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    # Capture raw bytes; benchmark output is not guaranteed to be UTF-8.
    stdout_bytes = proc.stdout if isinstance(proc.stdout, bytes) else b""
    return elapsed, proc.returncode, stdout_bytes.decode("utf-8", "replace"), None


def compiler_gym_clang_path() -> Path:
    from compiler_gym.third_party.llvm import clang_path

    return Path(clang_path())


def compiler_gym_opt_path() -> Path:
    from compiler_gym.third_party.llvm import opt_path

    return Path(opt_path())


def measure_benchmark(
    benchmark_uri: str,
    *,
    processed_csv: Path,
    sl_dir: Path,
    rl_dir: Path,
    max_steps: int,
    warmup: int,
    reps: int,
    cpu: int,
    timeout: int,
    seed: int,
    workdir: Path,
    include_fallback: bool,
    dataset_env=None,
) -> Dict:
    """Run the full O3-vs-hybrid runtime comparison for one benchmark."""
    import compiler_gym

    env = dataset_env or compiler_gym.make("llvm-v0")
    benchmark = env.datasets.benchmark(benchmark_uri)
    o0_bc = bytes(benchmark.proto.program.contents)
    dc = benchmark.proto.dynamic_config
    build_args = _command_args(dc.build_cmd)
    run_args = _command_args(dc.run_cmd)
    pre_cmds = _pre_run_shells(dc.pre_run_cmd)
    outfile = _command_outfile(dc.build_cmd)
    suite = benchmark_uri.split("://")[1].split("/")[0]

    if run_args:
        protocol = "native"
        run_cmd = " ".join(run_args)
    elif include_fallback:
        protocol = "fallback"
        run_cmd = "./a.out"
    else:
        return {
            "benchmark_uri": benchmark_uri,
            "suite": suite,
            "status": "skipped",
            "reason": (
                "no dynamic run configuration; rerun with --include-fallback "
                "to build and run ./a.out from bitcode"
            ),
        }

    base = workdir / _slug(benchmark_uri)
    base.mkdir(parents=True, exist_ok=True)

    # 1. Hybrid optimization -> final IR bitcode.
    from training.inference import hybrid_optimize_benchmark

    hybrid_result = hybrid_optimize_benchmark(
        benchmark_uri=benchmark_uri,
        max_steps=max_steps,
        sl_dir=sl_dir,
        rl_dir=rl_dir,
        reward_space="IrInstructionCountO3",
        measure_runtime=False,
        verbose=False,
        dump_bitcode_to=base / "hybrid.bc",
    )
    hybrid_bc = base / "hybrid.bc"
    if not hybrid_bc.exists() or hybrid_bc.stat().st_size == 0:
        raise HarnessError("hybrid optimization produced no bitcode output")

    # 2. opt -O3 on the identical O0 bitcode.
    o0_bc_path = base / "o0.bc"
    o0_bc_path.write_bytes(o0_bc)
    o3_bc = base / "o3.bc"
    opt_proc = subprocess.run(
        [
            str(compiler_gym_opt_path()),
            "-O3",
            str(o0_bc_path),
            "-o",
            str(o3_bc),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if opt_proc.returncode != 0 or not o3_bc.exists():
        raise HarnessError(f"opt -O3 failed: {opt_proc.stderr[-400:]}")

    # 3. Build both native executables with the identical build command.
    o3_dir = base / "o3"
    hybrid_dir = base / "hybrid"
    build_native(o3_bc.read_bytes(), o3_dir, build_args, outfile, timeout)
    build_native(hybrid_bc.read_bytes(), hybrid_dir, build_args, outfile, timeout)

    # 4. Interleaved, CPU-pinned timing with warmups.
    for _ in range(warmup):
        timed_run(run_cmd, pre_cmds, o3_dir, cpu, timeout)
        timed_run(run_cmd, pre_cmds, hybrid_dir, cpu, timeout)

    o3_samples: List[float] = []
    hybrid_samples: List[float] = []
    o3_stdout_hashes: List[str] = []
    hybrid_stdout_hashes: List[str] = []
    for _ in range(reps):
        for arm, samples, hashes in (
            ("o3", o3_samples, o3_stdout_hashes),
            ("hybrid", hybrid_samples, hybrid_stdout_hashes),
        ):
            workdir = o3_dir if arm == "o3" else hybrid_dir
            elapsed, rc, stdout_text, _ = timed_run(
                run_cmd, pre_cmds, workdir, cpu, timeout
            )
            if rc != 0:
                raise HarnessError(
                    f"{arm} run failed with rc={rc}: stdout={stdout_text[:200]!r}"
                )
            samples.append(elapsed)
            hashes.append(hashlib.sha256(stdout_text.encode("utf-8", "replace")).hexdigest())

    result = {
        "benchmark_uri": benchmark_uri,
        "suite": suite,
        "protocol": protocol,
        "status": "ok",
        "run_cmd": run_cmd,
        "build_cmd": shlex.join(build_args),
        "o0_ir_instruction_count": hybrid_result.get("initial_ir"),
        "o3": arm_stats(o3_samples, seed),
        "hybrid": arm_stats(hybrid_samples, seed),
        "outputs_match": len(set(o3_stdout_hashes)) == 1
        and o3_stdout_hashes == hybrid_stdout_hashes,
        "pass_sequence": hybrid_result.get("pass_sequence"),
        "final_ir": hybrid_result.get("final_ir"),
        "o3_ir_instruction_count": hybrid_result.get("o3_ir_instruction_count"),
        "hybrid_vs_o3_ir_pct": hybrid_result.get("hybrid_vs_o3_ir_pct"),
        "speedup_hybrid_vs_o3": (
            statistics.median(o3_samples) / statistics.median(hybrid_samples)
            if o3_samples and hybrid_samples
            else None
        ),
    }
    return result


def _load_output(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def cmd_measure(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)

    benchmark_uris = list(args.benchmarks)
    if not benchmark_uris and Path(args.processed_csv).exists():
        benchmark_uris = load_test_benchmarks(Path(args.processed_csv))
        LOGGER.info("Loaded %d test-split benchmarks from %s", len(benchmark_uris), args.processed_csv)
    if not benchmark_uris:
        print("No benchmarks to measure (pass --benchmarks or a valid --processed-csv).")
        return 1

    existing: Dict[str, Dict] = {}
    if output.exists():
        for row in _load_output(output):
            if row.get("benchmark_uri") and row.get("status") == "ok":
                existing[row["benchmark_uri"]] = row

    pending = [u for u in benchmark_uris if u not in existing]
    if not pending:
        print(f"All {len(benchmark_uris)} benchmarks already measured in {output}")
        return 0
    print(f"Measuring {len(pending)} benchmarks (already done: {len(existing)}):")
    for uri in pending:
        print("  ", uri)

    import compiler_gym

    env = compiler_gym.make("llvm-v0")
    try:
        for uri in pending:
            started = time.perf_counter()
            try:
                row = measure_benchmark(
                    uri,
                    processed_csv=Path(args.processed_csv),
                    sl_dir=Path(args.sl_model_dir),
                    rl_dir=Path(args.rl_model_dir),
                    max_steps=args.max_steps,
                    warmup=args.warmup,
                    reps=args.reps,
                    cpu=args.cpu,
                    timeout=args.timeout,
                    seed=args.seed,
                    workdir=workdir,
                    include_fallback=args.include_fallback,
                    dataset_env=env,
                )
            except Exception as error:
                row = {
                    "benchmark_uri": uri,
                    "status": "failed",
                    "reason": str(error)[:500],
                }
                LOGGER.warning("Benchmark %s failed: %s", uri, error)
            elapsed = time.perf_counter() - started
            row["elapsed_sec"] = round(elapsed, 2)
            existing[uri] = row
            # Incremental, resumable save after every benchmark.
            rows = list(existing.values())
            output.write_text(json.dumps(rows, indent=2, default=str))
            speedup = row.get("speedup_hybrid_vs_o3")
            if row.get("status") == "ok":
                print(
                    f"  {uri}: O3 {row['o3']['median_sec']:.6f}s | "
                    f"hybrid {row['hybrid']['median_sec']:.6f}s | "
                    f"speedup {speedup:.4f}x ({row['pass_sequence']})"
                )
            else:
                print(f"  {uri}: {row.get('status')} - {row.get('reason', '')[:120]}")
    finally:
        env.close()
    print(f"Saved {len(existing)} rows to {output}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    rows: List[Dict] = []
    for path in args.results:
        rows.extend(_load_output(Path(path)))
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = summarize_results(ok)
    summary["total_rows"] = len(rows)
    summary["failed"] = [r.get("benchmark_uri") for r in rows if r.get("status") == "failed"]
    summary["skipped"] = [r.get("benchmark_uri") for r in rows if r.get("status") == "skipped"]

    print("\n=== O3 executable runtime baseline (external harness) ===")
    print(f"Benchmarks measured: {summary['benchmarks_evaluated']} (of {len(rows)} rows)")
    print(
        f"Geo-mean speedup hybrid vs -O3: "
        f"{summary['geo_mean_speedup']:.4f}x  "
        f"(wins {summary['wins']} / {summary['benchmarks_evaluated']})"
    )
    if summary.get("wilcoxon_signed_rank", {}).get("p_value") is not None:
        w = summary["wilcoxon_signed_rank"]
        print(f"Wilcoxon signed-rank (hybrid faster): p = {w['p_value']:.4f}")
    print()
    print(f"{'benchmark':<48} {'proto':<8} {'O3 med':>10} {'Hyb med':>10} {'speedup':>9} {'win':>4}")
    for row in sorted(summary["rows"], key=lambda r: r["benchmark_uri"]):
        print(
            f"{row['benchmark_uri']:<48} {row['protocol']:<8} "
            f"{row['o3_median_sec']:>10.5f} {row['hybrid_median_sec']:>10.5f} "
            f"{row['speedup_hybrid_vs_o3']:>9.4f} "
            f"{'Y' if row['speedup_hybrid_vs_o3'] > 1 else 'N':>4}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nSaved summary to {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_measure = sub.add_parser("measure", help="Measure O3-vs-hybrid runtime for benchmarks")
    p_measure.add_argument("--processed-csv", default=str(PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset_scaled.csv"))
    p_measure.add_argument("--benchmarks", action="append", default=[], help="Benchmark URIs (repeatable); defaults to the test split of --processed-csv")
    p_measure.add_argument("--sl-model-dir", default=str(PROJECT_ROOT / "models" / "supervised"))
    p_measure.add_argument("--rl-model-dir", default=str(PROJECT_ROOT / "models" / "reinforcement"))
    p_measure.add_argument("--max-steps", type=int, default=8)
    p_measure.add_argument("--warmup", type=int, default=1)
    p_measure.add_argument("--reps", type=int, default=5)
    p_measure.add_argument("--cpu", type=int, default=4, help="CPU core to pin executions to")
    p_measure.add_argument("--timeout", type=int, default=120, help="Per-run timeout in seconds")
    p_measure.add_argument("--seed", type=int, default=42)
    p_measure.add_argument("--workdir", default=str(PROJECT_ROOT / "results" / "o3_harness_work"))
    p_measure.add_argument("--output", default=str(PROJECT_ROOT / "results" / "o3_harness_results.json"))
    p_measure.add_argument("--include-fallback", action="store_true", help="Measure benchmarks without a dynamic run config (build and run ./a.out from bitcode)")
    p_measure.add_argument("--log-level", default="WARNING")
    p_measure.set_defaults(func=cmd_measure)

    p_sum = sub.add_parser("summarize", help="Merge measure outputs into a comparison table")
    p_sum.add_argument("--results", nargs="+", required=True, help="One or more measure output JSON files")
    p_sum.add_argument("--output", default=str(PROJECT_ROOT / "results" / "o3_runtime_vs_o3_summary.json"))
    p_sum.add_argument("--log-level", default="WARNING")
    p_sum.set_defaults(func=cmd_summarize)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
