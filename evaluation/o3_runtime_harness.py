#!/usr/bin/env python3
"""
External -O3 executable runtime baseline harness (EXECUTION_RUNBOOK_RUNTIME.md, section 10).

The claim "hybrid beats -O3 on runtime" is only defensible with a controlled
executable baseline. CompilerGym exposes the exact -O3 IR cost but no -O3
runtime observation, so this harness builds and times native executables
outside CompilerGym:

    O0 bitcode (benchmark proto, identical to the env's start state)
        |
        +--> clang -O3      -> clang_o3/a.out   (clang's own full O3 pipeline)
        |
        +--> opt -O3 -> o3.bc -> clang -O3 -> o3/a.out
        |
        +--> hybrid -> hybrid.bc -> clang -O3 -> hybrid/a.out  (training/inference.py)

All three arms are compiled with clang's ``-O3`` codegen so the comparison
isolates the middle-end pass sequence:

* ``clang_o3`` - clang's own O3 pipeline on the same O0 bitcode (the reference
  "what -O3 gives you" baseline).
* ``o3`` - the earlier ``opt -O3``-only baseline, now with -O3 codegen (kept
  as a sanity check that opt's pipeline matches clang's).
* ``hybrid`` - NeuroCompiler's learned pass sequence applied as a pre-pass
  before the same -O3 codegen (the deployment story: the tool augments a
  normal -O3 build).

Note: plain ``clang module.bc`` without an -O flag emits effectively O0-level
codegen, so binaries built that way are NOT a valid -O3 runtime baseline; the
harness always passes -O3 now. (This invalidates the pre-2026 waves that used
default codegen.)

The executables are run with the benchmark's OWN dynamic run configuration
(CompilerGym's build_cmd / pre_run_cmd / run_cmd templates, which preserve the
cBench input setup such as ``_finfo_dataset``), identical warmups and
repetitions, ``taskset`` CPU pinning to the same core, and interleaved ordering
to cancel thermal/load drift. Results are reported as medians with 95%
bootstrap confidence intervals, per-benchmark speedup against BOTH baselines
(median baseline / median hybrid), paired Wilcoxon signed-rank tests, and an
output-hash equality check across the three binaries.

Since the hybrid pass sequence is input-independent, the binaries are built
ONCE per benchmark and measured on MULTIPLE inputs cheaply (--inputs), which
matters because cBench's default inputs are sub-10ms and dominated by process
startup; larger inputs make the runtime comparison meaningful.

Subcommands:
    measure    Measure one process-worth of benchmarks (resumable: benchmarks
               already present in --output are skipped).
    summarize  Merge one or more measure outputs into the comparison table
               (per benchmark, the input with the largest O3 median is the
               representative for the headline comparison).

Usage:
    python evaluation/o3_runtime_harness.py measure \
        --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
        --sl-model-dir models/supervised --rl-model-dir models/reinforcement \
        --max-steps 15 --warmup 1 --reps 5 --cpu 4 --timeout 120 \
        --inputs 0,largest \
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
import re
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


INPUT_INDEX_LARGEST = -1


def _numeric_inputs(input_path: Path) -> List[Path]:
    """Files in the input's directory with a numeric stem, same suffix first.

    cBench data directories contain the numbered datasets (1.dat, 2.dat, ...)
    plus helper artifacts (convert_tiff2bw, unpacked/), so we restrict to
    ``<digits>.<suffix>`` files and prefer the same suffix as the original
    input (stringsearch wants .txt, tiff wants .tif, bzip2 wants .bz2).
    """
    parent = input_path.parent
    if not parent.is_dir():
        return []
    numeric = [
        f
        for f in parent.iterdir()
        if f.is_file() and re.match(r"^\d+\.", f.name)
    ]
    same_suffix = [f for f in numeric if f.suffix == input_path.suffix]
    return sorted(same_suffix or numeric, key=lambda f: f.name)


def resolve_input(
    run_args: Sequence[str],
    pre_cmds: Sequence[str],
    input_index: int,
) -> Tuple[List[str], List[str], Dict]:
    """Swap the benchmark input file(s) and sync the pre-run dataset id.

    CompilerGym cBench run commands reference numbered input files (e.g.
    ``.../network_dijkstra_data/1.dat`` or stringsearch's ``1.txt`` +
    ``1.s.txt``) and the pre-run command echoes the matching dataset index
    into ``_finfo_dataset``. Choosing dataset ``N`` swaps *every* argument in
    the same data directory that belongs to the same numbered family
    (``1.txt`` -> ``20.txt``, ``1.s.txt`` -> ``20.s.txt``) and rewrites the
    echoed index. ``INPUT_INDEX_LARGEST`` selects the largest input by file
    size. Benchmarks without an input file (e.g. bitcount) are returned
    unchanged.
    """
    info = {"input_file": None, "input_index": None, "input_candidates": 0}
    chosen = None
    original: Optional[Path] = None
    orig_stem: Optional[str] = None
    chosen_stem: Optional[str] = None
    inputs: List[Path] = []
    for arg in run_args:
        candidate = Path(arg)
        if not (candidate.is_file() and candidate.parent.is_dir()):
            continue
        inputs = _numeric_inputs(candidate)
        if len(inputs) < 2:
            continue
        original = candidate
        orig_match = re.match(r"^(\d+)\.", candidate.name)
        orig_stem = orig_match.group(1) if orig_match else None
        if input_index == INPUT_INDEX_LARGEST:
            chosen = max(inputs, key=lambda f: f.stat().st_size)
        else:
            idx = min(max(input_index, 0), len(inputs) - 1)
            chosen = inputs[idx]
        info["input_candidates"] = len(inputs)
        stem = re.match(r"^(\d+)\.", chosen.name)
        chosen_stem = stem.group(1) if stem else None
        break

    if chosen is None:
        return list(run_args), list(pre_cmds), info

    new_run = []
    for arg in run_args:
        p = Path(arg)
        if original is not None and p.resolve() == original.resolve():
            # The argument that points at the dataset: use the chosen file as-is
            # (1.tif -> 17.nocomp.tif; the rest of the family may differ).
            new_run.append(str(chosen))
        elif (
            orig_stem is not None
            and chosen_stem is not None
            and p.parent == chosen.parent
            and p.is_file()
            and re.match(rf"^{orig_stem}\.", p.name)
        ):
            # Other numbered family members in the same directory: swap the
            # index while preserving their own rest-of-name (1.s.txt -> 20.s.txt).
            new_run.append(str(p.parent / (chosen_stem + p.name[len(orig_stem):])))
        else:
            new_run.append(arg)

    new_pre = list(pre_cmds)
    if chosen_stem is not None:
        new_pre = [
            re.sub(
                r"echo\s+(\d+)\s*>_finfo_dataset",
                f"echo {chosen_stem} >_finfo_dataset",
                pre,
            )
            for pre in new_pre
        ]
    info.update(
        {
            "input_file": str(chosen),
            "input_index": (
                inputs.index(chosen) if chosen in inputs else None
            ),
        }
    )
    return new_run, new_pre, info


def parse_inputs(spec: str) -> List[int]:
    """Parse the --inputs CLI spec: comma-separated indices and/or 'largest'."""
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token == "largest":
            out.append(INPUT_INDEX_LARGEST)
        else:
            out.append(int(token))
    return out


def build_native(
    bc_bytes: bytes,
    workdir: Path,
    build_args: Sequence[str],
    outfile: str,
    timeout: int,
    opt_level: str = "-O3",
) -> Path:
    """Compile a bitcode module to a native executable with -O3 codegen.

    Uses the benchmark's own build_cmd template ($CC -> bundled clang,
    $IN -> module path). Falls back to ``clang -O3 <module> -lm -o a.out``
    when no template is available (CHStone-style benchmarks).

    ``opt_level`` is deliberately applied to EVERY arm: plain ``clang``
    without an -O flag emits effectively O0-level codegen, which would
    handicap the -O3 baseline as much as the hybrid arm.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    module = workdir / "module.bc"
    module.write_bytes(bc_bytes)
    clang = str(compiler_gym_clang_path())
    if build_args and any("$IN" in a for a in build_args):
        # cBench templates tokenize as e.g. ["$CC", "$IN", "-o", "a.out", "-lm"].
        # Emit the compiler as separate argv entries so opt_level never lands
        # inside argv[0] ("clang -O3" as one token would fail to exec).
        cmd: List[str] = []
        for a in build_args:
            if a == "$CC":
                cmd.extend([clang, opt_level])
            elif "$CC" in a:
                cmd.append(a.replace("$CC", f"{clang} {opt_level}"))
            elif "$IN" in a:
                cmd.append(a.replace("$IN", str(module)))
            else:
                cmd.append(a)
    else:
        cmd = [clang, opt_level, str(module), "-lm", "-o", str(workdir / "a.out")]
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
    inputs: Sequence[int],
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
        # Default no_op_limit=1: episodes terminate at the first no-op. An
        # Aug 2026 experiment relaxed this (no_op_limit=max_steps) to let the
        # learned STOP/longer-horizon policy act; the longer 15-pass sequences
        # did NOT help runtime (dijkstra 0.957x vs clang-O3 vs 0.984x for the
        # short sequence — the backend re-optimizes the extra IR anyway), so
        # first-no-op termination remains the measured configuration.
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

    # 3. Build all three native executables once (input-independent).
    #    All arms get -O3 codegen; only the middle-end pass sequence differs.
    o3_dir = base / "o3"
    clang_o3_dir = base / "clang_o3"
    hybrid_dir = base / "hybrid"
    build_native(o3_bc.read_bytes(), o3_dir, build_args, outfile, timeout)
    build_native(o0_bc_path.read_bytes(), clang_o3_dir, build_args, outfile, timeout)
    build_native(hybrid_bc.read_bytes(), hybrid_dir, build_args, outfile, timeout)

    arms = (("o3", o3_dir), ("clang_o3", clang_o3_dir), ("hybrid", hybrid_dir))

    # 4. Interleaved, CPU-pinned timing with warmups, per requested input.
    measured_inputs: List[Dict] = []
    for input_index in inputs:
        run_args_i, pre_cmds_i, input_info = resolve_input(
            run_args, pre_cmds, input_index
        )
        run_cmd_i = " ".join(run_args_i)
        for _ in range(warmup):
            for _, arm_dir in arms:
                timed_run(run_cmd_i, pre_cmds_i, arm_dir, cpu, timeout)

        samples: Dict[str, List[float]] = {name: [] for name, _ in arms}
        hashes: Dict[str, List[str]] = {name: [] for name, _ in arms}
        for _ in range(reps):
            for name, arm_dir in arms:
                elapsed, rc, stdout_text, _ = timed_run(
                    run_cmd_i, pre_cmds_i, arm_dir, cpu, timeout
                )
                if rc != 0:
                    raise HarnessError(
                        f"{name} run failed with rc={rc} "
                        f"(input {input_info['input_file']}): "
                        f"stdout={stdout_text[:200]!r}"
                    )
                samples[name].append(elapsed)
                hashes[name].append(
                    hashlib.sha256(
                        stdout_text.encode("utf-8", "replace")
                    ).hexdigest()
                )

        o3_med = statistics.median(samples["o3"])
        clang_o3_med = statistics.median(samples["clang_o3"])
        hybrid_med = statistics.median(samples["hybrid"])
        measured_inputs.append(
            {
                "input_file": input_info["input_file"],
                "input_index": input_info["input_index"],
                "input_candidates": input_info["input_candidates"],
                "o3": arm_stats(samples["o3"], seed),
                "clang_o3": arm_stats(samples["clang_o3"], seed),
                "hybrid": arm_stats(samples["hybrid"], seed),
                "outputs_match": (
                    len(set(hashes["o3"])) == 1
                    and hashes["o3"] == hashes["clang_o3"]
                    and hashes["o3"] == hashes["hybrid"]
                ),
                "speedup_hybrid_vs_o3": o3_med / hybrid_med,
                "speedup_hybrid_vs_clang_o3": clang_o3_med / hybrid_med,
            }
        )

    first = measured_inputs[0] if measured_inputs else {}
    result = {
        "benchmark_uri": benchmark_uri,
        "suite": suite,
        "protocol": protocol,
        "status": "ok",
        "run_cmd": run_cmd,
        "build_cmd": shlex.join(build_args),
        "codegen": "-O3",
        "o0_ir_instruction_count": hybrid_result.get("initial_ir"),
        "pass_sequence": hybrid_result.get("pass_sequence"),
        "final_ir": hybrid_result.get("final_ir"),
        "o3_ir_instruction_count": hybrid_result.get("o3_ir_instruction_count"),
        "hybrid_vs_o3_ir_pct": hybrid_result.get("hybrid_vs_o3_ir_pct"),
        "inputs": measured_inputs,
        # Compatibility mirror of the first input (kept for older consumers).
        "o3": first.get("o3"),
        "clang_o3": first.get("clang_o3"),
        "hybrid": first.get("hybrid"),
        "outputs_match": first.get("outputs_match"),
        "speedup_hybrid_vs_o3": first.get("speedup_hybrid_vs_o3"),
        "speedup_hybrid_vs_clang_o3": first.get("speedup_hybrid_vs_clang_o3"),
    }
    return result


def _baseline_median_sec(entry: Dict) -> Optional[float]:
    """The largest baseline median in an input row (opt-o3 or clang-o3)."""
    o3 = (entry.get("o3") or {}).get("median_sec")
    c3 = (entry.get("clang_o3") or {}).get("median_sec")
    medians = [m for m in (o3, c3) if m]
    return max(medians) if medians else None


def _representative_input(result: Dict) -> Optional[Dict]:
    """The input row that best represents the benchmark (largest baseline median).

    Larger inputs give the most reliable timings (process startup does not
    dominate), so the representative is the measured input with the largest
    baseline median (max of the opt-o3 and clang-o3 medians, which are near
    identical). Falls back to the flat top-level stats for older result rows
    without clang-o3 data.
    """
    entries = result.get("inputs")
    if entries:
        valid = [
            e
            for e in entries
            if _baseline_median_sec(e) and (e.get("hybrid") or {}).get("median_sec")
        ]
        if not valid:
            return None
        return max(valid, key=_baseline_median_sec)
    hy = (result.get("hybrid") or {}).get("median_sec")
    if not hy or not _baseline_median_sec(result):
        return None
    return result


def summarize_results(
    results: Sequence[Dict],
) -> Dict:
    """Aggregate per-benchmark rows into the comparison table.

    Each benchmark is represented by the measurement (across all input
    results, e.g. a default-input wave and a large-input wave) with the
    largest O3 median, which is the most trustworthy timing (see
    :func:`_representative_input`). Kept as a pure function (no I/O) so it is
    unit-testable.
    """
    # Deduplicate by benchmark URI, keeping the largest-O3-median
    # representative across waves (default-input + large-input rows for the
    # same benchmark must not be double-counted).
    best: Dict[str, Dict] = {}
    for result in results:
        uri = result.get("benchmark_uri")
        if not uri:
            continue
        rep = _representative_input(result)
        if rep is None:
            continue
        o3_median = (rep.get("o3") or {}).get("median_sec")
        c3_median = (rep.get("clang_o3") or {}).get("median_sec")
        hy_median = (rep.get("hybrid") or {}).get("median_sec")
        baseline_median = _baseline_median_sec(rep)
        if (
            baseline_median is None
            or not hy_median
            or baseline_median <= 0
            or hy_median <= 0
        ):
            continue
        if uri not in best or baseline_median > best[uri]["baseline_median_sec"]:
            best[uri] = {
                "result": result,
                "rep": rep,
                "baseline_median_sec": baseline_median,
                "o3_median_sec": o3_median,
                "c3_median_sec": c3_median,
                "hy_median_sec": hy_median,
            }

    rows = []
    for uri, entry in sorted(best.items()):
        result = entry["result"]
        rep = entry["rep"]
        o3_median = entry["o3_median_sec"]
        c3_median = entry["c3_median_sec"]
        hy_median = entry["hy_median_sec"]
        speedup_o3 = o3_median / hy_median if o3_median else None
        speedup_c3 = c3_median / hy_median if c3_median else None
        row = {
            "benchmark_uri": uri,
            "suite": result.get("suite"),
            "protocol": result.get("protocol"),
            "codegen": result.get("codegen", "-O3"),
            "speedup_hybrid_vs_o3": speedup_o3,
            "log2_speedup": math.log2(speedup_o3) if speedup_o3 else None,
            "speedup_hybrid_vs_clang_o3": speedup_c3,
            "log2_speedup_vs_clang_o3": (
                math.log2(speedup_c3) if speedup_c3 else None
            ),
            "o3_median_sec": o3_median,
            "clang_o3_median_sec": c3_median,
            "hybrid_median_sec": hy_median,
            "o3_ci": [rep["o3"]["ci95_lo_sec"], rep["o3"]["ci95_hi_sec"]],
            "clang_o3_ci": (
                [
                    rep["clang_o3"]["ci95_lo_sec"],
                    rep["clang_o3"]["ci95_hi_sec"],
                ]
                if rep.get("clang_o3")
                else None
            ),
            "hybrid_ci": [
                rep["hybrid"]["ci95_lo_sec"],
                rep["hybrid"]["ci95_hi_sec"],
            ],
            "outputs_match": rep.get("outputs_match"),
            "input_file": rep.get("input_file"),
            "inputs_count": len(result.get("inputs") or []) or 1,
            "pass_sequence": result.get("pass_sequence"),
            "hybrid_vs_o3_ir_pct": result.get("hybrid_vs_o3_ir_pct"),
        }
        rows.append(row)

    speedups_o3 = [r["speedup_hybrid_vs_o3"] for r in rows if r["speedup_hybrid_vs_o3"]]
    speedups_c3 = [
        r["speedup_hybrid_vs_clang_o3"] for r in rows if r["speedup_hybrid_vs_clang_o3"]
    ]
    summary: Dict = {
        "benchmarks_evaluated": len(rows),
        "wins": sum(1 for s in speedups_o3 if s > 1.0),
        "losses": sum(1 for s in speedups_o3 if s < 1.0),
        "ties": sum(1 for s in speedups_o3 if s == 1.0),
        "wins_vs_clang_o3": sum(1 for s in speedups_c3 if s > 1.0),
        "losses_vs_clang_o3": sum(1 for s in speedups_c3 if s < 1.0),
        "ties_vs_clang_o3": sum(1 for s in speedups_c3 if s == 1.0),
        "geo_mean_speedup": geo_mean(speedups_o3),
        "geo_mean_speedup_vs_clang_o3": geo_mean(speedups_c3),
        "geo_mean_log2_speedup": (
            sum(r["log2_speedup"] for r in rows if r["log2_speedup"]) / len(rows)
            if rows
            else None
        ),
        "mean_speedup": statistics.fmean(speedups_o3) if speedups_o3 else None,
        "rows": rows,
    }

    if len(speedups_o3) >= 2:
        try:
            from scipy.stats import wilcoxon

            log_o3 = [r["log2_speedup"] for r in rows if r["log2_speedup"]]
            stat, p_value = wilcoxon(log_o3, alternative="greater")
            summary["wilcoxon_signed_rank"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "alternative": "hybrid faster than opt -O3 (on log2 speedups)",
            }
        except Exception as error:  # scipy missing or degenerate data
            summary["wilcoxon_signed_rank"] = {"error": str(error)}
    if len(speedups_c3) >= 2:
        try:
            from scipy.stats import wilcoxon

            log_c3 = [
                r["log2_speedup_vs_clang_o3"]
                for r in rows
                if r["log2_speedup_vs_clang_o3"]
            ]
            stat, p_value = wilcoxon(log_c3, alternative="greater")
            summary["wilcoxon_signed_rank_vs_clang_o3"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "alternative": "hybrid faster than clang -O3 (on log2 speedups)",
            }
        except Exception as error:  # scipy missing or degenerate data
            summary["wilcoxon_signed_rank_vs_clang_o3"] = {"error": str(error)}
    return summary


def _load_output(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def cmd_measure(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    inputs = parse_inputs(args.inputs)

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
    print(f"Measuring {len(pending)} benchmarks (already done: {len(existing)}), inputs={inputs}:")
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
                    inputs=inputs,
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
            if row.get("status") == "ok":
                rep = _representative_input(row)
                if rep is not None and rep.get("speedup_hybrid_vs_clang_o3") is not None:
                    print(
                        f"  {uri}: best-measured input "
                        f"{Path(rep['input_file']).name if rep.get('input_file') else '(default)'} | "
                        f"clang-O3 {rep['clang_o3']['median_sec']:.6f}s | "
                        f"opt-O3 {rep['o3']['median_sec']:.6f}s | "
                        f"hybrid {rep['hybrid']['median_sec']:.6f}s | "
                        f"spd/vO3 {rep['speedup_hybrid_vs_o3']:.4f}x | "
                        f"spd/vcO3 {rep['speedup_hybrid_vs_clang_o3']:.4f}x "
                        f"({len(row.get('inputs') or [])} inputs, {row['pass_sequence']})"
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

    print("\n=== O3 executable runtime baseline (external harness, -O3 codegen) ===")
    print(f"Benchmarks measured: {summary['benchmarks_evaluated']} (of {len(rows)} rows)")
    print(
        f"Geo-mean speedup hybrid vs opt -O3: "
        f"{summary['geo_mean_speedup']:.4f}x  "
        f"(wins {summary['wins']} / {summary['benchmarks_evaluated']})"
    )
    print(
        f"Geo-mean speedup hybrid vs clang -O3: "
        f"{summary['geo_mean_speedup_vs_clang_o3']:.4f}x  "
        f"(wins {summary['wins_vs_clang_o3']} / {summary['benchmarks_evaluated']})"
    )
    if summary.get("wilcoxon_signed_rank", {}).get("p_value") is not None:
        w = summary["wilcoxon_signed_rank"]
        print(f"Wilcoxon signed-rank (hybrid faster than opt -O3): p = {w['p_value']:.4f}")
    if summary.get("wilcoxon_signed_rank_vs_clang_o3", {}).get("p_value") is not None:
        w = summary["wilcoxon_signed_rank_vs_clang_o3"]
        print(
            f"Wilcoxon signed-rank (hybrid faster than clang -O3): "
            f"p = {w['p_value']:.4f}"
        )
    print()
    print(
        f"{'benchmark':<44} {'input':<18} {'vO3 med':>9} {'cO3 med':>9} "
        f"{'Hyb med':>9} {'spd/O3':>7} {'spd/cO3':>7} {'win':>4}"
    )
    for row in sorted(summary["rows"], key=lambda r: r["benchmark_uri"]):
        input_name = Path(row["input_file"]).name if row["input_file"] else "-"
        print(
            f"{row['benchmark_uri']:<44} {input_name:<18} "
            f"{row['o3_median_sec']:>9.5f} {row['clang_o3_median_sec'] or float('nan'):>9.5f} "
            f"{row['hybrid_median_sec']:>9.5f} "
            f"{row['speedup_hybrid_vs_o3'] or float('nan'):>7.4f} "
            f"{row['speedup_hybrid_vs_clang_o3'] or float('nan'):>7.4f} "
            f"{'Y' if (row['speedup_hybrid_vs_clang_o3'] or 0) > 1 else 'N':>4}"
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
    p_measure.add_argument(
        "--max-steps", type=int, default=15,
        help="Hybrid pass-sequence budget (longer horizons let the learned "
        "STOP action decide when to stop; default 15)",
    )
    p_measure.add_argument("--warmup", type=int, default=1)
    p_measure.add_argument("--reps", type=int, default=5)
    p_measure.add_argument("--cpu", type=int, default=4, help="CPU core to pin executions to")
    p_measure.add_argument("--timeout", type=int, default=120, help="Per-run timeout in seconds")
    p_measure.add_argument("--seed", type=int, default=42)
    p_measure.add_argument(
        "--inputs", default="0",
        help="Comma-separated input indices to measure per benchmark, or "
        "'largest' for the biggest input by file size (e.g. '0,largest'). "
        "The hybrid pass sequence is input-independent, so the binaries are "
        "built once and each input is timed with the same protocol.",
    )
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
