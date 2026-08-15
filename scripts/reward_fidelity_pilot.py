#!/usr/bin/env python3
"""Reward-fidelity pilot (Aug 2026).

Diagnostic pilot for docs/DIAGNOSIS_MULTISTEP_RL.md §3/§5: is the runtime
reward stable enough to order pass effects, or is the measurement protocol the
dominant noise source?

Protocol A (current training-time protocol, mirrors the RL buffer / multistate
dataset measurement):

  * input = default benchmark input (index 0) for cBench, ./a.out fallback for
    CHStone/csmith
  * warmup 0, runs 3 (median of 3), sequential measurement, taskset-pinned
    (identical to ``generate_multistate_dataset._median_time`` with the
    collector defaults)

Protocol B (existing high-fidelity protocol, mirrors the large-input sweep and
the O3 harness):

  * input = explicit large input for cBench (0.3-1.3 s workloads; CHStone/csmith
    have NO input files, so the workload cannot be scaled -- only the statistics
    improve: warmup 1, runs 7)
  * warmup 1, runs 5, INTERLEAVED round-robin timing across all 9 binaries
    (same interleaving structure as the O3 harness arm measurement), taskset-
    pinned, output hashes recorded for correctness

Each benchmark is measured under BOTH protocols, TWICE (batch 1, batch 2), so
repeatability can be quantified: sign agreement and Spearman rank correlation of
the 8-pass improvement ranking between batches.

No model training, no RL, no feature extraction: the same O0 bitcode and the
same 8 LOOP_PASSES the RL agent acts on, built once per benchmark with -O3
codegen (identical to the harness's build_native).

Usage:
  python scripts/reward_fidelity_pilot.py --benchmark benchmark://cbench-v1/gsm \\
      --input-idx 11 --output results/rf_pilot_samples.csv [--resume]
  python scripts/reward_fidelity_pilot.py --summarize --samples results/rf_pilot_samples.csv \\
      --output results/rf_pilot_summary.json

The driver writes every individual timing sample to --output incrementally and
is idempotent: (benchmark, protocol, batch) combinations already fully present
are skipped, so a timed-out run can simply be re-invoked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.o3_runtime_harness import (  # type: ignore
    _command_args,
    _command_outfile,
    _pre_run_shells,
    arm_stats,
    bootstrap_ci,
    build_native,
    resolve_input,
    timed_run,
)
from scripts.generate_multistate_dataset import LOOP_PASSES  # type: ignore

LOGGER = logging.getLogger("reward_fidelity_pilot")

# Protocol settings (fixed by the pilot design; do not tune).
PROTOCOLS = {
    "A": {"warmup": 0, "runs": 3, "interleaved": False, "input_idx": 0},
    "B": {"warmup": 1, "runs": 5, "interleaved": True, "input_idx": None},  # input set per benchmark
}
FALLBACK_PROTOCOLS = {  # CHStone/csmith: no inputs, so B only adds statistics
    "A": {"warmup": 0, "runs": 3, "interleaved": False},
    "B": {"warmup": 1, "runs": 7, "interleaved": True},
}


def _slug(uri: str) -> str:
    return uri.replace("://", "__").replace("/", "_")


def _apply_pass_bitcode(env, benchmark_uri: str, flag: str) -> Optional[bytes]:
    """Apply one loop pass to the O0 state and return the resulting bitcode."""
    try:
        env.reset(benchmark=benchmark_uri)
        _, _, done, _ = env.step(env.action_space.from_string(flag))
        raw = env.observation["Bitcode"]
        if hasattr(raw, "tobytes"):
            raw = raw.tobytes()
        return raw if isinstance(raw, bytes) else bytes(raw)
    except Exception:
        return None


def measure_benchmark(
    benchmark_uri: str,
    *,
    input_idx: Optional[int],
    workdir: Path,
    output_csv: Path,
    cpu: int,
    timeout: int,
    resume: bool,
) -> Dict:
    import compiler_gym

    env = compiler_gym.make("llvm-v0")
    done_combos: set = set()
    if resume and output_csv.exists() and output_csv.stat().st_size:
        with output_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["benchmark_uri"] == benchmark_uri:
                    done_combos.add((row["protocol"], row["batch"]))
    out = open(output_csv, "a", newline="")
    writer = None
    try:
        # Write the header exactly once: at creation, or on resume only if the
        # file has no header yet.
        fieldnames = [
            "benchmark_uri", "suite", "protocol", "batch",
            "pass_flag", "sample_idx", "elapsed_sec",
            "returncode", "out_sha256", "input_file",
        ]
        if output_csv.stat().st_size == 0:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
        benchmark = env.datasets.benchmark(benchmark_uri)
        o0_bc = bytes(benchmark.proto.program.contents)
        dc = benchmark.proto.dynamic_config
        build_args = _command_args(dc.build_cmd)
        run_args = _command_args(dc.run_cmd)
        pre_cmds = _pre_run_shells(dc.pre_run_cmd)
        outfile = _command_outfile(dc.build_cmd)
        suite = benchmark_uri.split("://")[1].split("/")[0]
        is_native = bool(run_args)
        base = workdir / _slug(benchmark_uri)
        base.mkdir(parents=True, exist_ok=True)

        # ---- Build all 9 binaries once (O0 + 8 loop passes), -O3 codegen. ----
        binaries: List[Tuple[str, Path]] = [("O0", base / "o0")]
        build_native(o0_bc, binaries[0][1], build_args, outfile, timeout)
        for i, flag in enumerate(LOOP_PASSES):
            bc = _apply_pass_bitcode(env, benchmark_uri, flag)
            if bc is None:
                LOGGER.warning("pass %s failed to apply on %s", flag, benchmark_uri)
                continue
            d = base / f"pass_{i:02d}"
            try:
                build_native(bc, d, build_args, outfile, timeout)
            except Exception as error:
                LOGGER.warning("build failed for %s: %s", flag, error)
                continue
            binaries.append((flag, d))
        LOGGER.info("%s: built %d binaries", benchmark_uri, len(binaries))

        # Resolve the run command for each protocol's input.
        proto_cmds: Dict[str, Tuple[str, List[str], Dict]] = {}
        if is_native:
            for proto, cfg in PROTOCOLS.items():
                idx = cfg["input_idx"] if proto == "A" else (input_idx or 0)
                ra, pr, info = resolve_input(run_args, pre_cmds, idx)
                proto_cmds[proto] = (" ".join(ra), pr, info)
        else:
            for proto, cfg in FALLBACK_PROTOCOLS.items():
                proto_cmds[proto] = ("./a.out", list(pre_cmds), {"input_file": None})
        LOGGER.info(
            "  inputs: A=%s B=%s",
            proto_cmds["A"][2].get("input_file"),
            proto_cmds["B"][2].get("input_file"),
        )

        results: Dict[Tuple[str, str], Dict] = {}
        for proto in ("A", "B"):
            cfg = PROTOCOLS[proto] if is_native else FALLBACK_PROTOCOLS[proto]
            run_cmd, pre_i, info = proto_cmds[proto]
            for batch in ("1", "2"):
                if resume and (proto, batch) in done_combos:
                    LOGGER.info("  skipping %s batch %s (already done)", proto, batch)
                    continue
                samples: Dict[str, List[Tuple[float, int, str]]] = {
                    name: [] for name, _ in binaries
                }
                if writer is None:
                    # Header already written at creation; only create the writer
                    # object if the file pre-existed with a header (resume).
                    writer = csv.DictWriter(out, fieldnames=fieldnames)
                if cfg["interleaved"]:
                    # Round-robin across all binaries, rotation per round.
                    for _ in range(cfg["warmup"]):
                        for name, d in binaries:
                            timed_run(run_cmd, pre_i, d, cpu, timeout)
                    for round_i in range(cfg["runs"]):
                        order = list(range(len(binaries)))
                        order = order[round_i % len(order):] + order[:round_i % len(order)]
                        for j in order:
                            name, d = binaries[j]
                            elapsed, rc, stdout_text, _ = timed_run(
                                run_cmd, pre_i, d, cpu, timeout
                            )
                            if rc != 0:
                                LOGGER.warning(
                                    "run failed %s %s batch %s: rc=%d",
                                    name, proto, batch, rc,
                                )
                            samples[name].append(
                                (elapsed, rc, hashlib.sha256(
                                    stdout_text.encode("utf-8", "replace")
                                ).hexdigest())
                            )
                else:
                    # Sequential per binary (current protocol A semantics).
                    for name, d in binaries:
                        for _ in range(cfg["warmup"]):
                            timed_run(run_cmd, pre_i, d, cpu, timeout)
                        for _ in range(cfg["runs"]):
                            elapsed, rc, stdout_text, _ = timed_run(
                                run_cmd, pre_i, d, cpu, timeout
                            )
                            samples[name].append(
                                (elapsed, rc, hashlib.sha256(
                                    stdout_text.encode("utf-8", "replace")
                                ).hexdigest())
                            )
                for name, d in binaries:
                    for k, (el, rc, h) in enumerate(samples[name]):
                        writer.writerow({
                            "benchmark_uri": benchmark_uri,
                            "suite": suite,
                            "protocol": proto,
                            "batch": batch,
                            "pass_flag": name,
                            "sample_idx": k,
                            "elapsed_sec": f"{el:.9f}",
                            "returncode": rc,
                            "out_sha256": h,
                            "input_file": info.get("input_file") or "",
                        })
                out.flush()
                # per-(benchmark, protocol, batch) pass summary
                o0_med = statistics.median(
                    [s[0] for s in samples["O0"] if s[1] == 0]
                ) if any(s[1] == 0 for s in samples["O0"]) else None
                for name, _ in binaries:
                    vals = [s[0] for s in samples[name] if s[1] == 0]
                    if not vals:
                        continue
                    med = statistics.median(vals)
                    summary = arm_stats(vals, seed=42)
                    summary["n"] = len(vals)
                    results[(proto, batch, name)] = {
                        "benchmark_uri": benchmark_uri,
                        "suite": suite,
                        "protocol": proto,
                        "batch": batch,
                        "pass_flag": name,
                        "n": len(vals),
                        "median_sec": med,
                        "mean_sec": summary["mean_sec"],
                        "std_sec": summary["std_sec"],
                        "cv_pct": (
                            100.0 * summary["std_sec"] / summary["mean_sec"]
                            if summary["std_sec"] and summary["mean_sec"]
                            else None
                        ),
                        "iqr_sec": (
                            float(
                                statistics.median(sorted(vals)[len(vals)//4:3*len(vals)//4] or vals)
                            ) * 0.0
                        ),  # placeholder replaced below
                        "ci95_lo_sec": summary["ci95_lo_sec"],
                        "ci95_hi_sec": summary["ci95_hi_sec"],
                        "improvement_pct": (
                            100.0 * (o0_med - med) / o0_med
                            if o0_med and o0_med > 0
                            else None
                        ),
                    }
        # Real IQR (percentiles) for each pass, computed from raw samples.
        q = {25: [], 50: [], 75: []}
        return {"benchmark_uri": benchmark_uri, "suite": suite, "summary": results}
    finally:
        out.close()
        env.close()


def _iqr(vals: Sequence[float]) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    lo = s[len(s) // 4]
    hi = s[(3 * len(s)) // 4]
    return hi - lo


def summarize(samples_path: Path) -> Dict:
    rows = list(csv.DictReader(open(samples_path)))
    if not rows:
        raise SystemExit(f"no samples in {samples_path}")
    # Per (benchmark, protocol, batch, pass) aggregated stats.
    agg: Dict[Tuple[str, str, str, str], List[float]] = {}
    meta: Dict[Tuple[str, str, str, str], Dict] = {}
    for r in rows:
        if r["benchmark_uri"] == "benchmark_uri":
            continue  # duplicate header artifact from interrupted runs
        key = (r["benchmark_uri"], r["protocol"], r["batch"], r["pass_flag"])
        agg.setdefault(key, []).append(float(r["elapsed_sec"]))
        meta[key] = {
            "suite": r["suite"],
            "input_file": r["input_file"],
            "returncode": r["returncode"],
        }

    benchmarks = sorted({r["benchmark_uri"] for r in rows})
    out: Dict = {"benchmarks": {}, "family_summary": []}
    for b in benchmarks:
        bd: Dict = {"suite": next(r["suite"] for r in rows if r["benchmark_uri"] == b)}
        per_proto: Dict[str, Dict] = {}
        for proto in ("A", "B"):
            batches = sorted({r["batch"] for r in rows if r["benchmark_uri"] == b and r["protocol"] == proto})
            pass_rows: Dict[str, Dict] = {}
            for pname in sorted({r["pass_flag"] for r in rows if r["benchmark_uri"] == b and r["protocol"] == proto}):
                o0s = [agg[(b, proto, batch, "O0")] for batch in batches if (b, proto, batch, "O0") in agg]
                o0_med = statistics.median([x for batch in o0s for x in batch]) if o0s else None
                by_batch: Dict[str, Dict] = {}
                for batch in batches:
                    vals = agg.get((b, proto, batch, pname), [])
                    if not vals:
                        continue
                    med = statistics.median(vals)
                    by_batch[batch] = {
                        "median_sec": med,
                        "mean_sec": statistics.fmean(vals),
                        "std_sec": statistics.stdev(vals) if len(vals) >= 2 else None,
                        "cv_pct": (100.0 * statistics.stdev(vals) / statistics.fmean(vals))
                        if len(vals) >= 2 and statistics.fmean(vals)
                        else None,
                        "iqr_sec": _iqr(vals),
                        "ci95_lo_sec": bootstrap_ci(vals, seed=42)[0],
                        "ci95_hi_sec": bootstrap_ci(vals, seed=42)[1],
                        "improvement_pct": (
                            100.0 * (o0_med - med) / o0_med
                            if o0_med and o0_med > 0
                            else None
                        ),
                        "n": len(vals),
                    }
                pass_rows[pname] = {
                    "batches": by_batch,
                    "improvement_pct_mean": statistics.mean(
                        [by_batch[bb]["improvement_pct"] for bb in by_batch if by_batch[bb].get("improvement_pct") is not None]
                    ) if by_batch and any(by_batch[bb].get("improvement_pct") is not None for bb in by_batch) else None,
                    "cv_pct_mean": statistics.mean(
                        [by_batch[bb]["cv_pct"] for bb in by_batch if by_batch[bb].get("cv_pct") is not None]
                    ) if by_batch and any(by_batch[bb].get("cv_pct") is not None for bb in by_batch) else None,
                }
            per_proto[proto] = {
                "passes": pass_rows,
                "o0_median_sec": o0_med,
                "batches": batches,
            }
        bd["protocols"] = per_proto
        out["benchmarks"][b] = bd

        # Repeatability metrics per protocol.
        for proto in ("A", "B"):
            passes = [p for p in per_proto[proto]["passes"] if p != "O0"]
            if len(passes) < 2:
                continue
            b1 = {p: per_proto[proto]["passes"][p]["batches"].get("1", {}).get("improvement_pct") for p in passes}
            b2 = {p: per_proto[proto]["passes"][p]["batches"].get("2", {}).get("improvement_pct") for p in passes}
            valid = [(b1[p], b2[p]) for p in passes if b1[p] is not None and b2[p] is not None]
            if len(valid) < 2:
                continue
            sign_agree = sum(1 for x, y in valid if (x > 0) == (y > 0)) / len(valid)
            from scipy.stats import spearmanr
            rho, _ = spearmanr([x for x, _ in valid], [y for _, y in valid])
            per_proto[proto]["repeatability"] = {
                "n_passes": len(valid),
                "sign_agreement": sign_agree,
                "spearman_batch1_vs_batch2": float(rho),
                "mean_abs_improvement_pct": statistics.mean(
                    [abs(x) for x, _ in valid]
                ),
            }
    return out


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    measure_benchmark(
        args.benchmark,
        input_idx=args.input_idx,
        workdir=Path(args.workdir).resolve(),
        output_csv=output,
        cpu=args.cpu,
        timeout=args.timeout,
        resume=args.resume,
    )
    print(f"done in {time.perf_counter() - started:.0f}s -> {output}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = summarize(Path(args.samples))
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    # Table: family | benchmark | runtime scale | CV A/B | reward stability A/B
    print(f"\n{'family':9s} {'benchmark':12s} {'proto':5s} {'O0 med':>9s} {'medCV%':>7s} {'signAgr':>7s} {'rankCorr':>8s} {'meanAbsImp%':>10s}")
    for b, bd in sorted(summary["benchmarks"].items()):
        fam = "cBench" if bd["suite"] == "cbench-v1" else bd["suite"].split("-")[0].upper()
        for proto in ("A", "B"):
            if proto not in bd["protocols"]:
                continue
            p = bd["protocols"][proto]
            o0 = p.get("o0_median_sec")
            rep = p.get("repeatability")
            cvs = [x.get("cv_pct_mean") for x in p["passes"].values() if x.get("cv_pct_mean")]
            print(f"{fam:9s} {b.split('/')[-1]:12s} {proto:5s} {o0*1000 if o0 else 0:8.1f}ms "
                  f"{statistics.median(cvs) if cvs else 0:6.1f}% "
                  f"{rep['sign_agreement']*100 if rep else float('nan'):6.0f}% "
                  f"{rep['spearman_batch1_vs_batch2'] if rep else float('nan'):8.2f} "
                  f"{rep['mean_abs_improvement_pct'] if rep else float('nan'):9.2f}%")
    print(f"\nSaved summary to {args.output}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("run", help="Measure one benchmark under both protocols, two batches")
    pr.add_argument("--benchmark", required=True)
    pr.add_argument("--input-idx", type=int, default=None,
                    help="Protocol B large input index (cBench only); default 0")
    pr.add_argument("--workdir", default=str(PROJECT_ROOT / "results" / "rf_pilot_work"))
    pr.add_argument("--output", default=str(PROJECT_ROOT / "results" / "rf_pilot_samples.csv"))
    pr.add_argument("--cpu", type=int, default=4)
    pr.add_argument("--timeout", type=int, default=120)
    pr.add_argument("--resume", action="store_true")
    pr.set_defaults(func=cmd_run)
    ps = sub.add_parser("summarize", help="Aggregate samples into repeatability tables")
    ps.add_argument("--samples", default=str(PROJECT_ROOT / "results" / "rf_pilot_samples.csv"))
    ps.add_argument("--output", default=str(PROJECT_ROOT / "results" / "rf_pilot_summary.json"))
    ps.set_defaults(func=cmd_summarize)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(parse_args().func(parse_args()))
