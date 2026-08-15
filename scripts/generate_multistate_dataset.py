#!/usr/bin/env python3
"""
Generate a MULTI-STATE pass-quality SL dataset with runtime measured natively.

Research motivation: the single-pass-from-O0 dataset design is structurally
incapable of teaching pass selection — every pass row of a benchmark shares the
same pre-state feature vector, so a model cannot discriminate between candidate
passes (verified: LOBO top-3 = 0% on all 8 large-input benchmarks). The fix is
a *transition* structure: evaluate candidate passes at SEVERAL distinct states
per benchmark (O0 plus random pass prefixes), so pre-state features vary across
rows and the label is the pass's effect AT that state.

Each row: (benchmark_uri, state, pass_flag, pre-state features of THAT state,
runtime_improvement_pct vs the state's own baseline). z-score this per
(benchmark, state) with scripts/zscore_dataset.py (after extending it) or
inline when training.

Usage:
  python scripts/generate_multistate_dataset.py \\
    --benchmark benchmark://cbench-v1/gsm --inputs 11 \\
    --output datasets/processed/gsm_multistate.csv

Cost: per (state, pass) one native build + (warmup + runs) timed executions.
Default eval set is the loop-pass subset (8 passes); pick a 0.2-2 s input so a
3-state x 8-pass sweep fits in a bounded window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
from scripts.extract_features import (  # type: ignore
    AUTOPHASE_FEATURE_NAMES,
    MeasurementConfig,
    extract_features,
)

LOGGER = logging.getLogger("generate_multistate_dataset")

LOOP_PASSES = [
    "-licm",
    "-loop-rotate",
    "-loop-unroll",
    "-loop-vectorize",
    "-loop-deletion",
    "-loop-unswitch",
    "-loop-distribute",
    "-indvars",
]

# Passes with real IR effects, used to build the non-O0 states. Loop passes
# are deliberately NOT here: -loop-unroll/-loop-vectorize/-loop-deletion are
# no-ops at O0 on several cBench benchmarks (verified), so they cannot
# construct meaningfully different states. These scalar passes changed IR on
# every benchmark in the large-input sweep.
STATE_BUILDER_PASSES = [
    "-sroa",
    "-instcombine",
    "-gvn",
    "-newgvn",
    "-simplifycfg",
    "-dce",
    "-adce",
    "-memcpyopt",
    "-reassociate",
    "-licm",
]

# Minimum feature distance a candidate state must have from EVERY accepted
# state to be accepted. Calibrated on real data: near-duplicate states (e.g.
# -memcpyopt at O0 changes the bitcode but leaves the model-visible features
# identical) measure ~0.000, while genuinely different states (IR-reducing
# prefixes like -newgvn) measure ~0.25-0.31. The bitcode-signature check alone
# is NOT enough — it accepts near-duplicates (verified failure mode), so a
# candidate is accepted only if its autophase/IR feature vector is at least
# this far from every previously accepted state.
STATE_DIVERSITY_THRESHOLD = 0.05

BASE_COLUMNS = [
    "benchmark_uri",
    "state_index",
    "state_id",
    "prefix_sequence",
    "state_ir_instruction_count",
    "pass_flag",
    "runtime_improvement_pct",
    "state_runtime_median_sec",
    "runtime_median_sec",
    "o0_runtime_median_sec",
    "warmup",
    "runs",
]

# RL replay-buffer schema (compatible with training/train_rl.py's reader):
# one row per measured (state, pass) transition with pre/post features and the
# measured runtime improvement as the reward. ``hybrid_reward`` is the raw
# per-state improvement pct at collection time; run scripts/zscore_dataset.py
# with ``--group-col pre_state_id --sync-to hybrid_reward`` to convert it to a
# per-(benchmark, state) z-score (the same target the SL loop scorer uses).
# Episodes are one-step (done=True): the measured transition is the whole
# episode. training/train_rl.py synthesizes STOP transitions for these.
REPLAY_FIELDS = [
    "episode_id",
    "benchmark_uri",
    "state_index",
    "step_index",
    "pass_flag",
    "done",
    "hybrid_reward",
    "runtime_improvement",
    "pre_state_id",
    "post_state_id",
    "pre_ir_instruction_count",
    "post_ir_instruction_count",
    "pre_runtime_median_sec",
    "post_runtime_median_sec",
    "pre_object_text_size_bytes",
    "post_object_text_size_bytes",
    "pre_total_basic_blocks",
    "post_total_basic_blocks",
    "pre_total_functions",
    "post_total_functions",
    "pre_total_instructions",
    "post_total_instructions",
    "pre_total_memory_instructions",
    "post_total_memory_instructions",
] + [f"pre_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES] \
  + [f"post_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES]


def _replay_episode_id(benchmark_uri: str, state_id: str) -> str:
    """Deterministic episode id per (benchmark, state): all 8 pass transitions
    measured at that state belong to the same one-step episode."""
    return hashlib.sha256(f"{benchmark_uri}|{state_id}".encode()).hexdigest()[:24]


def build_replay_row(
    benchmark_uri: str,
    state_index: int,
    step_index: int,
    pass_flag: str,
    state_id: str,
    post_state_id: str,
    state_features: Dict[str, str],
    post_features: Dict[str, str],
    state_ir: Optional[int],
    post_ir: Optional[int],
    state_med: float,
    post_med: float,
    improvement: float,
) -> Dict[str, str]:
    """Build one RL-schema transition row from a measured (state, pass) pair.

    ``state_features``/``post_features`` are pre_*-prefixed feature dicts from
    ``_pre_state_row``; the post dict is relabeled to post_* here. The reward
    (``hybrid_reward``) is the raw runtime improvement pct vs the state's own
    baseline; z-score it per (benchmark, state) before training so it matches
    the SL scorer's target exactly.
    """
    row: Dict[str, str] = {
        "episode_id": _replay_episode_id(benchmark_uri, state_id),
        "benchmark_uri": benchmark_uri,
        "state_index": str(state_index),
        "step_index": str(step_index),
        "pass_flag": pass_flag,
        "done": "True",
        "hybrid_reward": f"{improvement:.6f}",
        "runtime_improvement": f"{improvement:.6f}",
        "pre_state_id": state_id,
        "post_state_id": post_state_id,
        "pre_ir_instruction_count": str(state_ir or ""),
        "post_ir_instruction_count": str(post_ir or ""),
        "pre_runtime_median_sec": f"{state_med:.6f}",
        "post_runtime_median_sec": f"{post_med:.6f}",
    }
    for key, value in state_features.items():
        row[key] = value
        if key.startswith("pre_"):
            row["post_" + key[len("pre_"):]] = post_features.get(key, "")
    return row


def _state_signature(bc_bytes: bytes) -> str:
    return hashlib.sha256(bc_bytes).hexdigest()[:16]


def _bitcode_bytes(env) -> bytes:
    raw = env.observation["Bitcode"]
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    return raw if isinstance(raw, bytes) else bytes(raw)


def _ir_count(env) -> Optional[int]:
    try:
        raw = env.observation["IrInstructionCount"]
        if isinstance(raw, int):
            return raw
        if hasattr(raw, "reshape"):
            return int(raw.reshape(-1)[0])
        return int(raw[0])
    except Exception as error:
        LOGGER.warning("Could not read IrInstructionCount: %s", error)
        return None


def _pre_state_row(env) -> Dict[str, str]:
    """The deterministic pre_* feature columns the SL trainer consumes."""
    measurement = MeasurementConfig(
        measure_runtime=False,
        measure_buildtime=False,
        collect_object_text_size=True,
    )
    features = extract_features(env, measurement)
    flat = features.flattened("pre_")
    return {
        k: v
        for k, v in flat.items()
        if k.startswith("pre_autophase_")
        or k
        in (
            "pre_ir_instruction_count",
            "pre_object_text_size_bytes",
            "pre_total_basic_blocks",
            "pre_total_functions",
            "pre_total_instructions",
            "pre_total_memory_instructions",
        )
    }


def _autophase_proportions(row: Dict[str, str]) -> List[float]:
    """Normalized (L1=1) autophase histogram of a pre-state row."""
    values = [float(row.get(c, "") or 0) for c in sorted(row) if c.startswith("pre_autophase_")]
    total = sum(values) or 1.0
    return [v / total for v in values]


def _feature_distance(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Distance between two pre-state feature rows: autophase-proportion L2
    plus relative IR-count difference. Near-duplicate states measure ~0.0;
    genuinely different states measure ~0.25+ (calibrated on cBench)."""
    pa, pb = _autophase_proportions(a), _autophase_proportions(b)
    l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))
    ira = float(a.get("pre_ir_instruction_count", "") or 0)
    irb = float(b.get("pre_ir_instruction_count", "") or 0)
    rel_ir = abs(ira - irb) / max(ira, irb, 1.0)
    return l2 + rel_ir


def _median_time(
    run_cmd: str,
    pre_cmds: Sequence[str],
    workdir: Path,
    cpu: int,
    timeout: int,
    warmup: int,
    runs: int,
) -> Optional[float]:
    for _ in range(warmup):
        timed_run(run_cmd, pre_cmds, workdir, cpu, timeout)
    samples: List[float] = []
    for _ in range(runs):
        elapsed, rc, stdout_text, _ = timed_run(run_cmd, pre_cmds, workdir, cpu, timeout)
        if rc != 0:
            LOGGER.warning("Run failed rc=%d: %r", rc, stdout_text[:200])
            return None
        samples.append(elapsed)
    return statistics.median(samples) if samples else None


def generate(args: argparse.Namespace) -> Path:
    import compiler_gym

    env = compiler_gym.make("llvm-v0")
    try:
        benchmark = env.datasets.benchmark(args.benchmark)
        o0_bc = bytes(benchmark.proto.program.contents)
        dc = benchmark.proto.dynamic_config
        build_args = _command_args(dc.build_cmd)
        run_args = _command_args(dc.run_cmd)
        pre_cmds = _pre_run_shells(dc.pre_run_cmd)
        outfile = _command_outfile(dc.build_cmd)
        if not run_args and not args.fallback:
            raise SystemExit(
                f"{args.benchmark} has no dynamic run configuration; rerun "
                "with --fallback to build ./a.out and run it with no inputs "
                "(CHStone/csmith-style)."
            )
        if args.fallback and not run_args:
            LOGGER.info("Fallback protocol: build ./a.out and run with no inputs")
            run_cmd_i = "./a.out"
            pre_cmds_i = []
            input_info = {"input_file": None, "input_index": None, "input_candidates": 0}
        else:
            run_args_i, pre_cmds_i, input_info = resolve_input(
                run_args, pre_cmds, args.inputs
            )
            run_cmd_i = " ".join(run_args_i)

        workdir = Path(args.workdir) / args.benchmark.split("/")[-1]
        workdir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Input %s: %s (runs ~%s)",
            args.inputs,
            input_info.get("input_file") or "(fallback: ./a.out)",
            input_info.get("input_size", "?"),
        )

        eval_passes = (
            [p.strip() for p in args.eval_passes.split(",") if p.strip()]
            if args.eval_passes
            else LOOP_PASSES
        )

        # States: O0 plus states built by applying IR-reducing scalar passes
        # one at a time, accepting a new state only when the bitcode ACTUALLY
        # changed (no-op prefixes would collapse states and defeat the
        # multi-state design — verified failure mode of random prefixes).
        builder = list(STATE_BUILDER_PASSES)
        rng = random.Random(args.seed)
        rng.shuffle(builder)

        states: List[Dict] = []  # each: bc, features, prefix, ir
        accepted_sigs = {_state_signature(o0_bc)}
        # State 0 = O0.
        env.reset(benchmark=args.benchmark)
        states.append(
            {
                "bc": o0_bc,
                "features": _pre_state_row(env),
                "prefix": [],
                "ir": _ir_count(env),
            }
        )
        # Extra states: walk the builder passes, accepting a state only when
        # BOTH the bitcode signature is new AND its feature vector is at least
        # STATE_DIVERSITY_THRESHOLD away from every accepted state (a new
        # signature is not enough — -memcpyopt changes the bitcode while
        # leaving the model-visible features identical).
        env.reset(benchmark=args.benchmark)
        prefix: List[str] = []
        current_sig = _state_signature(o0_bc)
        guard = 0
        while len(states) < args.states and guard < 60:
            flag = builder[guard % len(builder)]
            guard += 1
            env.step(env.action_space.from_string(flag))
            prefix.append(flag)
            bc_bytes = _bitcode_bytes(env)
            sig = _state_signature(bc_bytes)
            if sig == current_sig:
                continue  # no-op pass; keep walking
            current_sig = sig
            if sig in accepted_sigs:
                continue  # back to a state we already have
            cand_features = _pre_state_row(env)
            min_dist = min(
                _feature_distance(cand_features, acc["features"])
                for acc in states
            )
            if min_dist < args.diversity_threshold:
                LOGGER.info(
                    "Rejected near-duplicate state after %s (min dist %.4f < %.2f)",
                    "->".join(prefix), min_dist, args.diversity_threshold,
                )
                continue
            accepted_sigs.add(sig)
            states.append(
                {
                    "bc": bc_bytes,
                    "features": cand_features,
                    "prefix": list(prefix),
                    "ir": _ir_count(env),
                }
            )
            LOGGER.info(
                "Accepted state %d: prefix %s (IR %s, min dist %.4f)",
                len(states) - 1, "->".join(prefix), states[-1]["ir"], min_dist,
            )

        rows: List[Dict[str, str]] = []

        # RL replay output (append mode: each benchmark invocation adds its
        # rows to the shared buffer file; header written once on first use).
        replay_handle = None
        if args.emit_replay:
            replay_path = Path(args.emit_replay).expanduser().resolve()
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_new = not replay_path.exists() or replay_path.stat().st_size == 0
            replay_handle = replay_path.open("a", newline="", encoding="utf-8")
            replay_writer = csv.DictWriter(replay_handle, fieldnames=REPLAY_FIELDS, extrasaction="ignore")
            if replay_new:
                replay_writer.writeheader()
                replay_handle.flush()

        o0_dir = workdir / "state_00_o0"
        build_native(o0_bc, o0_dir, build_args, outfile, args.timeout)
        o0_med = _median_time(
            run_cmd_i, pre_cmds_i, o0_dir, args.cpu, args.timeout,
            args.warmup, args.runs,
        )
        LOGGER.info("O0 baseline median %.4f s", o0_med or float("nan"))

        for state_idx, state in enumerate(states):
            state_dir = workdir / f"state_{state_idx:02d}"
            build_native(state["bc"], state_dir, build_args, outfile, args.timeout)
            state_med = _median_time(
                run_cmd_i, pre_cmds_i, state_dir, args.cpu, args.timeout,
                args.warmup, args.runs,
            )
            if state_med is None:
                LOGGER.warning("State %d run failed; skipping", state_idx)
                continue
            LOGGER.info(
                "State %d (prefix %s) median %.4f s",
                state_idx,
                "->".join(state["prefix"]) or "O0",
                state_med,
            )
            for pass_index, flag in enumerate(eval_passes, start=1):
                started = time.perf_counter()
                try:
                    env.reset(benchmark=args.benchmark)
                    for pf in state["prefix"]:
                        env.step(env.action_space.from_string(pf))
                    env.step(env.action_space.from_string(flag))
                    pass_bc = _bitcode_bytes(env)
                    post_features = _pre_state_row(env)
                    post_ir = _ir_count(env)
                except Exception as error:
                    LOGGER.warning("Pass %s at state %d failed (%s); skipping",
                                   flag, state_idx, error)
                    continue
                pass_dir = workdir / f"state_{state_idx:02d}_pass_{pass_index:02d}"
                try:
                    build_native(pass_bc, pass_dir, build_args, outfile, args.timeout)
                except Exception as error:
                    LOGGER.warning("Build failed for %s at state %d (%s); skipping",
                                   flag, state_idx, error)
                    continue
                med = _median_time(
                    run_cmd_i, pre_cmds_i, pass_dir, args.cpu, args.timeout,
                    args.warmup, args.runs,
                )
                if med is None:
                    continue
                improvement = 100.0 * (state_med - med) / state_med if state_med > 0 else 0.0
                row: Dict[str, str] = {
                    "benchmark_uri": args.benchmark,
                    "state_index": str(state_idx),
                    "state_id": _state_signature(state["bc"]),
                    "prefix_sequence": "->".join(state["prefix"]) or "O0",
                    "state_ir_instruction_count": str(state["ir"] or ""),
                    "pass_flag": flag,
                    "runtime_improvement_pct": f"{improvement:.4f}",
                    "state_runtime_median_sec": f"{state_med:.6f}",
                    "runtime_median_sec": f"{med:.6f}",
                    "o0_runtime_median_sec": f"{o0_med:.6f}" if o0_med else "",
                    "warmup": str(args.warmup),
                    "runs": str(args.runs),
                }
                row.update(state["features"])
                rows.append(row)
                if args.emit_replay:
                    replay_row = build_replay_row(
                        benchmark_uri=args.benchmark,
                        state_index=state_idx,
                        step_index=pass_index - 1,
                        pass_flag=flag,
                        state_id=_state_signature(state["bc"]),
                        post_state_id=_state_signature(pass_bc),
                        state_features=state["features"],
                        post_features=post_features,
                        state_ir=state["ir"],
                        post_ir=post_ir,
                        state_med=state_med,
                        post_med=med,
                        improvement=improvement,
                    )
                    replay_writer.writerow(replay_row)
                    replay_handle.flush()
                LOGGER.info(
                    "[state %d | pass %d/%d] %-20s med %.4f s  imp %+.2f%%  (%.1f s)",
                    state_idx, pass_index, len(eval_passes), flag, med, improvement,
                    time.perf_counter() - started,
                )

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(BASE_COLUMNS)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if replay_handle is not None:
            replay_handle.close()

        print(
            f"\nWrote {len(rows)} rows "
            f"({args.states} states x {len(eval_passes)} passes) to {output_path}"
        )
        if args.emit_replay:
            print(f"RL replay rows appended to {Path(args.emit_replay).resolve()}")
        return output_path
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Runnable benchmark URI")
    parser.add_argument("--inputs", type=int, default=-1, help="Input index (or -1 = largest)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--emit-replay", default=None,
        help="Optional RL replay-buffer CSV path: also append one transition "
        "row per measured (state, pass) with pre/post features and the raw "
        "runtime improvement as reward (then z-score per pre_state_id and "
        "train with training/train_rl.py).",
    )
    parser.add_argument("--states", type=int, default=3, help="Total states per benchmark incl. O0")
    parser.add_argument(
        "--eval-passes", default=",".join(LOOP_PASSES),
        help="Comma-separated passes to evaluate at each state (default: loop subset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fallback", action="store_true",
        help="Build ./a.out from bitcode and run it with no inputs when the "
        "benchmark has no dynamic run config (CHStone/csmith-style). Runtime "
        "signal is startup-dominated at ms scale; use --runs >= 5.",
    )
    parser.add_argument(
        "--diversity-threshold", type=float, default=STATE_DIVERSITY_THRESHOLD,
        help="Minimum feature distance from every accepted state for a candidate "
        "state to be accepted (near-duplicates rejected; default %.2f)" % STATE_DIVERSITY_THRESHOLD,
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT / "results" / "multistate_work"))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
