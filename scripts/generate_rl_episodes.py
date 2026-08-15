#!/usr/bin/env python3
"""Generate GENUINE multi-step RL training episodes for NeuroCompiler.

Research fix (Aug 2026): the previous RL replay buffers were one-step — every
row was ``done=True`` and post-states were never chained, so the fitted-Q
Bellman branch in ``train_sklearn_dqn`` never bootstrapped and the "RL" model
was a myopic reward regressor. This collector fixes the data side:

    S0 --A0--> S1 --A1--> S2 --A2--> ... --STOP--> terminal

* One CompilerGym env per episode: ``env.step(action)`` mutates the SAME
  LLVM state that the next step reads. The state is NEVER regenerated from
  S0 between steps.
* ``state_id`` is the ``IrSha1`` observation (identical identity to
  ``training/inference.py`` masking).
* Non-terminal transitions carry ``done=False``; only genuine episode
  termination carries ``done=True``. STOP is an explicit terminal action and
  only ever appears as the final transition of an episode (real STOP rows,
  not synthesized).
* Per-(state_id, action) masking: an action that produces no state change is
  masked for the rest of the episode; the transition is kept with ``r=0`` and
  ``post_state_id == pre_state_id`` (the FQI target clamps self-loops).
* Start distribution: 70% episodes from O0, 30% from deeper valid states
  (built with the same diversity-guarded scalar-prefix walk as
  ``generate_multistate_dataset.py``), per the approved design.
* Reward: raw per-step ``100*(t_before - t_after)/t_before`` (incremental
  effect of the selected pass, NOT vs O3), z-scored per benchmark later with
  ``scripts/zscore_dataset.py --group-col benchmark_uri``.
* Static first-round collection only: behavior policy = SL-top-1 prior with
  epsilon-greedy + a fixed share of fully random episodes + small STOP
  probability. No policy-recollection iteration (Phase 8).

Output schema is the RL trainer's replay-buffer schema (see REPLAY_FIELDS)
plus ``available_actions`` (JSON; the unmasked candidate set at the POST
state, recomputed exactly at episode end), ``no_op``, ``terminal_reason`` and
``start_state``.

Usage:
  python scripts/generate_rl_episodes.py \\
      --episodes-per-benchmark 12 --workers 4 \\
      --emit-buffer datasets/replay_buffer/rl_experiences_multistep_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.o3_runtime_harness import (  # type: ignore
    _command_args,
    _command_outfile,
    _pre_run_shells,
    build_native,
    resolve_input,
)
from scripts.generate_multistate_dataset import (  # type: ignore
    LOOP_PASSES,
    STATE_BUILDER_PASSES,
    STATE_DIVERSITY_THRESHOLD,
    _bitcode_bytes,
    _feature_distance,
    _median_time,
)
from scripts.extract_features import (  # type: ignore
    AUTOPHASE_FEATURE_NAMES,
    MeasurementConfig,
    extract_features,
)
from training.common import derive_ratio_features  # type: ignore

LOGGER = logging.getLogger("generate_rl_episodes")

STOP_FLAG = "-stop"
MAX_STEPS_DEFAULT = 15

# Feature columns written for both pre_ and post_ states (SL/RL schema).
CORE_COLUMNS = [
    "ir_instruction_count",
    "object_text_size_bytes",
    "total_basic_blocks",
    "total_functions",
    "total_instructions",
    "total_memory_instructions",
]

BUFFER_FIELDS = [
    "episode_id",
    "benchmark_uri",
    "step_index",
    "pass_flag",
    "done",
    "hybrid_reward",
    "runtime_improvement",
    "pre_state_id",
    "post_state_id",
    "no_op",
    "terminal_reason",
    "start_state",
    "available_actions",
    "pre_runtime_median_sec",
    "post_runtime_median_sec",
] + [f"pre_{c}" for c in CORE_COLUMNS] + [f"post_{c}" for c in CORE_COLUMNS] \
  + [f"pre_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES] \
  + [f"post_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES] \
  + [f"pre_{c}" for c in ("ir_per_func", "mem_frac", "size_per_inst", "blocks_per_func", "insts_per_block")] \
  + [f"post_{c}" for c in ("ir_per_func", "mem_frac", "size_per_inst", "blocks_per_func", "insts_per_block")]

# The 27 non-evaluation training benchmarks, taken from the clean SL dataset
# (benchmark-level split used for Experiment A). The 8 large-input cBench
# evaluation programs are deliberately NOT here.
NON_EVAL_BENCHMARKS_PATH = PROJECT_ROOT / "datasets" / "processed" / "multistate_clean_sl_z.csv"

FAST_MEASUREMENT = MeasurementConfig(
    measure_runtime=False,
    measure_buildtime=False,
    collect_object_text_size=True,
)


def _feature_value(features, col: str) -> str:
    if col == "ir_instruction_count":
        return str(features.ir_instruction_count)
    if col == "object_text_size_bytes":
        return str(features.object_text_size_bytes) if features.object_text_size_bytes is not None else ""
    if col == "total_basic_blocks":
        return str(features.total_basic_blocks)
    if col == "total_functions":
        return str(features.total_functions)
    if col == "total_instructions":
        return str(features.total_instructions)
    if col == "total_memory_instructions":
        return str(features.total_memory_instructions)
    return ""


def _state_row(features, prefix: str) -> Dict[str, str]:
    """Flatten a ProgramFeatures snapshot to pre_/post_ feature columns."""
    row: Dict[str, str] = {}
    for col in CORE_COLUMNS:
        row[f"{prefix}{col}"] = _feature_value(features, col)
    for name in AUTOPHASE_FEATURE_NAMES:
        row[f"{prefix}autophase_{name}"] = str(features.autophase[name])
    return row


def _pre_state_dict(features) -> Dict[str, str]:
    """pre_*-prefixed feature dict (core + autophase) for a state."""
    out = _state_row(features, "pre_")
    # Ratio features derived online (scale-free representation).
    out.update(derive_ratio_features({k: v for k, v in out.items()}, "pre_"))
    return out


def _measure_state_runtime(env, workdir, build_args, outfile, run_cmd, pre_cmds, cpu, timeout, warmup, runs) -> Tuple[Optional[float], Optional[bytes]]:
    """Build the native executable of the CURRENT env state and time it.

    State workdirs include the process id so parallel workers never write the
    same module.bc / a.out concurrently (content-hash-only names raced: one
    worker's build overwrote another worker's executable mid-run, rc=126).
    """
    bc = _bitcode_bytes(env)
    if bc is None:
        return None, None
    try:
        token = f"{os.getpid()}"
        state_dir = workdir / ("s_" + hashlib.sha256(bc).hexdigest()[:16] + "_" + token)
        build_native(bc, state_dir, build_args, outfile, timeout)
        med = _median_time(run_cmd, pre_cmds, state_dir, cpu, timeout, warmup, runs)
    except Exception as error:
        LOGGER.warning("State runtime build/measure failed: %s", error)
        return None, bc
    return med, bc


def _build_deeper_states(env, benchmark_uri: str, seed: int, max_states: int) -> List[List[str]]:
    """Diversity-guarded scalar-prefix walk (mirrors generate_multistate.py).

    Returns a list of pass prefixes (each a list of flags) that lead to
    genuinely distinct states. Loop passes are NOT used here (no-ops at O0 on
    several benchmarks); scalar passes build the states the RL policy then
    acts on.
    """
    from scripts.generate_multistate_dataset import _state_signature
    benchmark = env.datasets.benchmark(benchmark_uri)
    o0_bc = bytes(benchmark.proto.program.contents)
    rng = random.Random(f"{seed}:{benchmark_uri}")
    builder = list(STATE_BUILDER_PASSES)
    rng.shuffle(builder)
    env.reset(benchmark=benchmark_uri)
    accepted_rows = [_pre_state_dict(extract_features(env, FAST_MEASUREMENT))]
    accepted_sigs = {_state_signature(o0_bc)}
    prefixes: List[List[str]] = []
    prefix: List[str] = []
    current_sig = _state_signature(o0_bc)
    guard = 0
    while len(prefixes) < max_states and guard < 60:
        flag = builder[guard % len(builder)]
        guard += 1
        env.step(env.action_space.from_string(flag))
        prefix.append(flag)
        bc = _bitcode_bytes(env)
        sig = _state_signature(bc)
        if sig == current_sig:
            continue
        current_sig = sig
        if sig in accepted_sigs:
            continue
        cand_row = _pre_state_dict(extract_features(env, FAST_MEASUREMENT))
        min_dist = min(_feature_distance(cand_row, acc) for acc in accepted_rows)
        if min_dist < STATE_DIVERSITY_THRESHOLD:
            continue
        accepted_sigs.add(sig)
        accepted_rows.append(cand_row)
        prefixes.append(list(prefix))
    return prefixes


def _load_sl_scorer(sl_dir: Path):
    """Return (model, feature_cols, vocab, feature_meta) or (None,...) if absent."""
    try:
        from training.inference import load_sl_model
        model, feature_cols, vocab, _pass_list, feature_meta = load_sl_model(Path(sl_dir))
        return model, feature_cols, vocab, feature_meta
    except Exception as error:
        LOGGER.warning("SL scorer unavailable (%s); using random behavior policy", error)
        return None, None, None, None


def _sl_top_flag(sl_scorer, features, available: List[str]):
    """Best available pass under the SL scorer (one-step ranking prior)."""
    model, feature_cols, vocab, feature_meta = sl_scorer
    if model is None or feature_cols is None or vocab is None:
        return None
    try:
        from training.inference import predict_sl_distribution
        ranked = predict_sl_distribution(
            model, feature_cols, vocab, features, available,
            temperature=1.0, feature_meta=feature_meta,
        )
        return ranked[0][0] if ranked else None
    except Exception:
        return None


def _episode_rng(seed: int, benchmark_uri: str, episode_index: int) -> random.Random:
    return random.Random(f"{seed}:{benchmark_uri}:{episode_index}")


def collect_episode(
    env,
    benchmark_uri: str,
    episode_index: int,
    start_prefix: List[str],
    args,
    sl_scorer,
    workdir,
    build_args,
    outfile,
    run_cmd,
    pre_cmds,
    cpu,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Run one episode starting from O0 (start_prefix=[]) or a deeper state.

    Returns (rows, terminal_reason) or (None, error) when the episode cannot
    start (measurement failure etc.). The episode_id encodes the start.
    """
    episode_id = hashlib.sha256(f"{benchmark_uri}|{episode_index:05d}".encode()).hexdigest()[:24]
    rng = _episode_rng(args.seed, benchmark_uri, episode_index)
    random_episode = rng.random() < args.random_episode_frac

    env.reset(benchmark=benchmark_uri)
    for flag in start_prefix:
        try:
            env.step(env.action_space.from_string(flag))
        except Exception as error:
            return None, f"prefix_failed:{flag}:{error}"

    pre = extract_features(env, FAST_MEASUREMENT)
    pre_med, _ = _measure_state_runtime(
        env, workdir, build_args, outfile, run_cmd, pre_cmds, cpu,
        args.timeout, args.warmup, args.runs,
    )
    if pre_med is None:
        return None, "initial_measure_failed"

    start_state = "o0" if not start_prefix else "deeper"
    rows: List[Dict[str, str]] = []
    tried_in_state: Dict[str, set] = {}
    terminal_reason: Optional[str] = None

    for step in range(args.max_steps):
        tried = tried_in_state.setdefault(pre.state_id, set())
        available = [f for f in LOOP_PASSES if f not in tried]
        if not available:
            terminal_reason = "all_tried"
            break

        # Behavior policy: STOP with probability p_stop; else epsilon-greedy
        # over the SL top-1 prior (or uniform for random episodes).
        if rng.random() < args.p_stop:
            stop_row = _make_stop_row(
                episode_id, benchmark_uri, step, pre, pre_med,
                start_state, available,
            )
            rows.append(stop_row)
            terminal_reason = "stop"
            break

        if random_episode or rng.random() < args.explore_eps:
            action = rng.choice(available)
        else:
            action = _sl_top_flag(sl_scorer, pre, available)
            if action is None or action not in available:
                action = rng.choice(available)

        try:
            _, _, done, info = env.step(env.action_space.from_string(action))
        except Exception as error:
            terminal_reason = f"action_failed:{error}"
            break

        post = extract_features(env, FAST_MEASUREMENT)
        post_med, _ = _measure_state_runtime(
            env, workdir, build_args, outfile, run_cmd, pre_cmds, cpu,
            args.timeout, args.warmup, args.runs,
        )
        if post_med is None:
            # Cannot compute r_t without the post runtime; keep rows so far.
            terminal_reason = "measure_failed"
            break

        no_op = post.state_id == pre.state_id
        r_raw = 100.0 * (pre_med - post_med) / pre_med if pre_med > 0 else 0.0
        is_last = (step == args.max_steps - 1)
        row = {
            "episode_id": episode_id,
            "benchmark_uri": benchmark_uri,
            "step_index": str(step),
            "pass_flag": action,
            "done": "True" if is_last else "False",
            "hybrid_reward": f"{r_raw:.6f}",
            "runtime_improvement": f"{r_raw:.6f}",
            "pre_state_id": pre.state_id,
            "post_state_id": post.state_id,
            "no_op": "True" if no_op else "False",
            "terminal_reason": "",
            "start_state": start_state,
            "available_actions": json.dumps(sorted(available)),
            "pre_runtime_median_sec": f"{pre_med:.6f}",
            "post_runtime_median_sec": f"{post_med:.6f}",
        }
        row.update(_pre_state_dict(pre))
        row.update({f"post_{k[len('pre_'):]}": v for k, v in _pre_state_dict(post).items()})
        rows.append(row)

        if no_op:
            tried.add(action)
        if is_last:
            terminal_reason = "max_steps"
            break
        pre = post
        pre_med = post_med

    if rows and terminal_reason:
        rows[-1]["terminal_reason"] = terminal_reason

    # Exact available-action sets at each POST state (final mask of the
    # episode): the bootstrap max in FQI must only range over actions still
    # selectable at s', plus the always-available STOP.
    for row in rows:
        if row.get("pass_flag") == STOP_FLAG:
            row["available_actions"] = ""
            continue
        post_id = row["post_state_id"]
        tried_at_end = tried_in_state.get(post_id, set())
        avail = [f for f in LOOP_PASSES if f not in tried_at_end]
        row["available_actions"] = json.dumps(sorted(avail))

    return rows, terminal_reason


def _make_stop_row(
    episode_id, benchmark_uri, step, pre, pre_med, start_state, available,
) -> Dict[str, str]:
    row = {
        "episode_id": episode_id,
        "benchmark_uri": benchmark_uri,
        "step_index": str(step),
        "pass_flag": STOP_FLAG,
        "done": "True",
        "hybrid_reward": "0.0",
        "runtime_improvement": "0.0",
        "pre_state_id": pre.state_id,
        "post_state_id": pre.state_id,
        "no_op": "False",
        "terminal_reason": "stop",
        "start_state": start_state,
        "available_actions": "",
        "pre_runtime_median_sec": f"{pre_med:.6f}",
        "post_runtime_median_sec": f"{pre_med:.6f}",
    }
    row.update(_pre_state_dict(pre))
    row.update({f"post_{k[len('pre_'):]}": v for k, v in _pre_state_dict(pre).items()})
    return row


def _worker(work: Tuple) -> List[Dict[str, str]]:
    """multiprocessing worker: one env, collect a batch of episodes."""
    (
        benchmark_uri,
        episodes,           # list of (episode_index, start_prefix)
        args_dict,
    ) = work
    import compiler_gym
    args = argparse.Namespace(**args_dict)
    sl_scorer = _load_sl_scorer(args.sl_model_dir)
    workdir = Path(args.workdir) / benchmark_uri.split("/")[-1]
    workdir.mkdir(parents=True, exist_ok=True)

    env = compiler_gym.make("llvm-v0")
    rows_out: List[Dict[str, str]] = []
    try:
        try:
            benchmark = env.datasets.benchmark(benchmark_uri)
        except Exception as error:
            LOGGER.warning("benchmark %s unresolvable: %s", benchmark_uri, error)
            return []
        dc = benchmark.proto.dynamic_config
        build_args = _command_args(dc.build_cmd)
        run_args = _command_args(dc.run_cmd)
        pre_cmds = _pre_run_shells(dc.pre_run_cmd)
        outfile = _command_outfile(dc.build_cmd)
        if run_args:
            run_args_i, pre_cmds_i, _ = resolve_input(run_args, pre_cmds, args.inputs)
            run_cmd = " ".join(run_args_i)
        elif args.fallback:
            run_cmd = "./a.out"
            pre_cmds_i = []
        else:
            LOGGER.warning("%s has no run config and --fallback not set; skipped", benchmark_uri)
            return []

        for episode_index, start_prefix in episodes:
            rows, reason = collect_episode(
                env, benchmark_uri, episode_index, start_prefix, args,
                sl_scorer, workdir, build_args, outfile, run_cmd,
                pre_cmds_i, args.cpu,
            )
            if rows is None:
                LOGGER.warning("episode %s[%d] failed: %s", benchmark_uri, episode_index, reason)
                continue
            rows_out.extend(rows)
    finally:
        env.close()
    return rows_out


def _load_benchmark_uris(benchmarks: List[str]) -> List[str]:
    if benchmarks:
        return benchmarks
    if NON_EVAL_BENCHMARKS_PATH.exists():
        import csv as _csv
        uris = []
        with NON_EVAL_BENCHMARKS_PATH.open(newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                uri = row.get("benchmark_uri", "")
                if uri and uri not in uris:
                    uris.append(uri)
        if uris:
            return uris
    # Fallback hardcoded list (verified against the clean split).
    cbench = ["patricia", "qsort", "sha", "susan", "tiffdither", "tiffmedian"]
    chstone = ["adpcm", "aes", "blowfish", "dfadd", "dfdiv", "dfmul", "dfsin",
               "gsm", "jpeg", "mips", "motion", "sha"]
    csmith = ["4", "6", "8", "9", "12", "17", "24", "26", "29"]
    return (
        [f"benchmark://cbench-v1/{b}" for b in cbench]
        + [f"benchmark://chstone-v0/{b}" for b in chstone]
        + [f"generator://csmith-v0/{b}" for b in csmith]
    )


def _existing_episodes(buffer_path: Path) -> set:
    if not buffer_path.exists() or buffer_path.stat().st_size == 0:
        return set()
    seen = set()
    with buffer_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row.get("episode_id", "")
            if eid:
                seen.add(eid)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", action="append", default=[],
                        help="Benchmark URIs (repeatable); default = the 27 clean training benchmarks")
    parser.add_argument("--episodes-per-benchmark", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--start-deeper-frac", type=float, default=0.3)
    parser.add_argument("--deeper-states", type=int, default=2,
                        help="How many deeper start states to build per benchmark (beyond O0)")
    parser.add_argument("--random-episode-frac", type=float, default=0.25)
    parser.add_argument("--explore-eps", type=float, default=0.3)
    parser.add_argument("--p-stop", type=float, default=0.08)
    parser.add_argument("--sl-model-dir", default=str(PROJECT_ROOT / "models" / "supervised_loop_multistate_clean"))
    parser.add_argument("--inputs", type=int, default=0, help="cBench input index (0 = smallest)")
    parser.add_argument("--fallback", action="store_true",
                        help="Build ./a.out and run with no inputs when the benchmark has no run config (CHStone/csmith)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--workdir", default=str(PROJECT_ROOT / "results" / "msrl_episode_work"))
    parser.add_argument("--emit-buffer", default=str(PROJECT_ROOT / "datasets" / "replay_buffer" / "rl_experiences_multistep_raw.csv"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip (benchmark, episode) pairs already present in --emit-buffer")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # build_native uses the workdir as both cwd and module path; a relative
    # path would misresolve (module.bc looked up relative to its own dir).
    args.workdir = str(Path(args.workdir).resolve())

    uris = _load_benchmark_uris(args.benchmarks)
    if not uris:
        raise SystemExit("No benchmark URIs to collect")

    # Build deeper start states for each benchmark (one env, in main).
    import compiler_gym
    deeper_prefixes: Dict[str, List[List[str]]] = {}
    env = compiler_gym.make("llvm-v0")
    try:
        for uri in uris:
            try:
                deeper_prefixes[uri] = _build_deeper_states(
                    env, uri, args.seed, max_states=args.deeper_states,
                )
            except Exception as error:
                LOGGER.warning("deeper-state build failed for %s: %s", uri, error)
                deeper_prefixes[uri] = []
    finally:
        env.close()

    buffer_path = Path(args.emit_buffer)
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_episodes(buffer_path) if args.resume else set()

    work_items: List[Tuple] = []
    for uri in uris:
        episodes = []
        prefixes = deeper_prefixes.get(uri, []) or []
        for i in range(args.episodes_per_benchmark):
            if not prefixes or (i % 10) / 10.0 < (1.0 - args.start_deeper_frac):
                start_prefix: List[str] = []
            else:
                start_prefix = prefixes[(i // 2) % len(prefixes)]
            eid = hashlib.sha256(f"{uri}|{i:05d}".encode()).hexdigest()[:24]
            if args.resume and eid in existing:
                continue
            episodes.append((i, start_prefix))
        if not episodes:
            continue
        # Split the benchmark's episodes across workers so parallelism helps
        # WITHIN a benchmark; each split pins its timed runs to its own core.
        n_splits = max(1, min(args.workers, len(episodes)))
        for s in range(n_splits):
            sub = episodes[s::n_splits]
            if not sub:
                continue
            item_args = dict(vars(args))
            item_args["cpu"] = args.cpu + (s % max(args.workers, 1))
            work_items.append((uri, sub, item_args))

    total_episodes = sum(len(w[1]) for w in work_items)
    LOGGER.info("Collecting %d episodes over %d benchmarks (deeper prefixes built)",
                total_episodes, len(work_items))

    all_rows: List[Dict[str, str]] = []
    if args.workers > 1 and len(work_items) > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for rows in pool.imap_unordered(_worker, work_items):
                all_rows.extend(rows)
    else:
        for item in work_items:
            all_rows.extend(_worker(item))

    # Deterministic order and append.
    all_rows.sort(key=lambda r: (r["benchmark_uri"], r["episode_id"], int(r["step_index"])))
    new_file = not buffer_path.exists() or buffer_path.stat().st_size == 0
    with buffer_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BUFFER_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    episodes_written = len({r["episode_id"] for r in all_rows})
    print(f"Wrote {len(all_rows)} transitions from {episodes_written} new episodes to {buffer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
