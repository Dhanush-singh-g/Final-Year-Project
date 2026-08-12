#!/usr/bin/env python3
"""
Phase 5 — RL Dataset Generation (Experience Collection)

Difference from SL:
- No fixed CSV of independent samples
- RL agent creates experience via episodes

One episode:
  Program -> State S0 -> Choose Pass -> State S1 -> Choose Pass -> State S2 -> ... -> Terminal

State: Current LLVM IR -> extract_features.py -> Feature Vector (same extractor as SL)
Action: One LLVM Pass
Environment: CompilerGym applies it, returns S_{t+1}
Reward: 0.6*Runtime Improvement + 0.3*IR Reduction + 0.1*Code Size Reduction

Store Transition: (State, Action, Reward, Next State, Done) in replay buffer

Termination if:
- Reward becomes zero
- No IR change
- Repeated state
- Maximum passes (10/15/20)

How many RL episodes?
  For every benchmark 200 random episodes (as per design)
  500 benchmarks * 200 = 100k episodes
  Each ~10 transitions => ~1M transitions

This file implements that pipeline with resumability and incremental flushing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from .extract_features import MeasurementConfig, ProgramFeatures, extract_features, AUTOPHASE_FEATURE_NAMES
    from .run_passes import ActionMetadata, resolve_actions, run_pass_sequence, Transition
    from .curated_passes import get_curated_flags
    from .reward import compute_hybrid_reward, RewardWeights
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from extract_features import MeasurementConfig, ProgramFeatures, extract_features, AUTOPHASE_FEATURE_NAMES  # type: ignore
    from run_passes import ActionMetadata, resolve_actions, run_pass_sequence, Transition  # type: ignore
    from curated_passes import get_curated_flags  # type: ignore
    from reward import compute_hybrid_reward, RewardWeights  # type: ignore

LOGGER = logging.getLogger("neurocompiler.collect_rl")

DEFAULT_RL_OUTPUT = PROJECT_ROOT / "datasets" / "replay_buffer" / "rl_experiences.csv"

PROVENANCE_FIELDS = [
    "episode_id",
    "transition_key",
    "run_id",
    "generated_at_utc",
    "benchmark_suite",
    "compiler_gym_version",
    "compiler_version",
    "host_name",
    "python_version",
]

# RL replay buffer schema
RL_FIELDS = PROVENANCE_FIELDS + [
    "benchmark_uri",
    "episode_index",
    "step_index",
    "pass_id",
    "pass_name",
    "pass_flag",
    "pass_position",
    "previous_pass_sequence",
    "reward_space",
    "raw_step_reward",          # raw from CompilerGym reward space
    "hybrid_reward",            # 0.6*RT + 0.3*IR + 0.1*Size scaled x100
    "runtime_improvement",
    "ir_improvement",
    "size_improvement",
    "cumulative_reward",
    "pass_success",
    "action_had_no_effect",
    "done",
    "termination_reason",
    "step_walltime_sec",
    "error_type",
    "error_message",
] + [f"pre_{name}" for name in ["state_id", "ir_instruction_count", "object_text_size_bytes", "total_basic_blocks", "total_functions", "total_instructions", "total_memory_instructions"]] \
  + [f"post_{name}" for name in ["state_id", "ir_instruction_count", "object_text_size_bytes", "total_basic_blocks", "total_functions", "total_instructions", "total_memory_instructions"]] \
  + [f"pre_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES] \
  + [f"post_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES] \
  + ["delta_ir_instruction_count", "delta_object_text_size_bytes", "pre_runtime_median_sec", "post_runtime_median_sec"]

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _dataset_uri(value: str) -> str:
    value = value.strip().rstrip("/")
    return value if "://" in value else f"benchmark://{value}"

def enumerate_benchmarks(env: Any, dataset_uri: str, explicit: Sequence[str], limit: Optional[int]) -> List[str]:
    if explicit:
        return [item if "://" in item else f"{dataset_uri}/{item.lstrip('/')}" for item in explicit]
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
    return [item.strip() for item in raw.split(",") if item.strip()]

def _transition_key(benchmark_uri: str, pre_state_id: str, action_id: int, episode_id: str, step_index: int) -> str:
    payload = {
        "benchmark_uri": benchmark_uri,
        "pre_state_id": pre_state_id,
        "pass_id": action_id,
        "episode_id": episode_id,
        "step_index": step_index,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

def _episode_payload(
    benchmark_uri: str, ep_idx: int, seed: int, action_flags: Sequence[str], max_steps: int
) -> str:
    """Canonical episode identity string (also used for the resume key)."""
    return (
        f"{benchmark_uri}|{ep_idx}|{seed}|{max_steps}|{','.join(action_flags)}"
    )


def episode_id_for(
    benchmark_uri: str, ep_idx: int, seed: int, action_flags: Sequence[str], max_steps: int
) -> str:
    """Deterministic episode resume key."""
    payload = _episode_payload(benchmark_uri, ep_idx, seed, action_flags, max_steps)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def episode_rng_for(
    benchmark_uri: str, ep_idx: int, seed: int, action_flags: Sequence[str], max_steps: int
) -> random.Random:
    """Per-episode RNG (review fix): reproducible across resume.

    The stream is derived ONLY from the episode identity, so a resumed run
    regenerates the exact same episode content instead of shifting a single
    global RNG position (which made the old resume silently produce different
    data for all later episodes).
    """
    payload = _episode_payload(benchmark_uri, ep_idx, seed, action_flags, max_steps)
    episode_id = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return random.Random(f"{payload}|{episode_id}")


def _load_completed_episodes(path: Path) -> Set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # tolerate old schema
        if not reader.fieldnames or "episode_id" not in reader.fieldnames:
            return set()
        return {row["episode_id"] for row in reader if row.get("episode_id")}

def collect_rl(args: argparse.Namespace) -> Path:
    try:
        import compiler_gym
    except ImportError as error:
        raise SystemExit(
            "CompilerGym is unavailable. Activate the 'neurocompiler' Conda "
            "environment before running this program."
        ) from error

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_episodes = _load_completed_episodes(output_path) if args.resume else set()
    if not args.resume and output_path.exists():
        output_path.unlink()

    measurement = MeasurementConfig(
        measure_runtime=args.measure_runtime,
        runtime_count=args.runtime_count,
        runtime_warmup_count=args.runtime_warmup_count,
        measure_buildtime=False,
        collect_object_text_size=not args.skip_object_text_size,
    )

    weights = RewardWeights(
        runtime=args.reward_weight_runtime,
        ir=args.reward_weight_ir,
        code_size=args.reward_weight_size,
    )

    random.seed(args.seed)
    run_id = uuid.uuid4().hex
    dataset_uri = _dataset_uri(args.dataset)
    generated_at = _utc_now()
    compiler_gym_version = str(getattr(compiler_gym, "__version__", "0.2.5"))

    env = compiler_gym.make("llvm-v0")
    try:
        compiler_version = str(getattr(env, "compiler_version", "unknown"))
        benchmarks = enumerate_benchmarks(env, dataset_uri, args.benchmark, args.max_benchmarks)
        actions = resolve_actions(env, _parse_passes(args.passes) or get_curated_flags())
        if args.max_passes is not None:
            actions = actions[: args.max_passes]
        if not actions:
            raise RuntimeError("No LLVM passes selected")
        LOGGER.info(f"RL collection: {len(benchmarks)} benchmarks x {args.episodes_per_benchmark} episodes x max {args.max_steps_per_episode} steps, {len(actions)} actions")

        file_exists = output_path.exists() and output_path.stat().st_size > 0
        total_transitions = 0
        skipped_episodes = 0

        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RL_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                handle.flush()

            for bench_idx, benchmark_uri in enumerate(benchmarks, start=1):
                LOGGER.info(f"Benchmark {bench_idx}/{len(benchmarks)}: {benchmark_uri}")

                for ep_idx in range(args.episodes_per_benchmark):
                    # Deterministic ID makes --resume effective across reruns.
                    action_flags = [a.flag for a in actions]
                    episode_id = episode_id_for(
                        benchmark_uri, ep_idx, args.seed, action_flags, args.max_steps_per_episode
                    )
                    if episode_id in resume_episodes:
                        skipped_episodes += 1
                        continue

                    # Per-episode RNG (review fix): every episode draws from its
                    # own stream derived from (benchmark, episode, seed, action
                    # set), so a resumed run reproduces the SAME episode content
                    # instead of shifting a single global RNG stream (which made
                    # the old resume silently produce different data).
                    episode_rng = episode_rng_for(
                        benchmark_uri, ep_idx, args.seed, action_flags, args.max_steps_per_episode
                    )

                    # For random episodes vs guided: random choices
                    # Could later use SL model as prior; for now pure random as per design

                    try:
                        env.reset(benchmark=benchmark_uri, reward_space=args.reward_space, timeout=args.timeout)
                        state = extract_features(env, measurement)
                    except Exception as e:
                        LOGGER.warning(f"Skip benchmark {benchmark_uri} episode {ep_idx}: {e}")
                        continue

                    visited_states = {state.state_id}
                    previous_passes: List[int] = []
                    cumulative = 0.0

                    for step_idx in range(args.max_steps_per_episode):
                        # Choose action from the episode's own RNG (deterministic
                        # under the same seed / resume state). The 10% early-stop
                        # branch is a crude stand-in for a real STOP action until
                        # one is added to the action space (see runbook).
                        if previous_passes and episode_rng.random() < 0.1:
                            break

                        action = episode_rng.choice(actions)
                        # Avoid repeating same pass if previous had no effect? we will check after step

                        started = time.perf_counter()
                        try:
                            transitions = run_pass_sequence(
                                env,
                                [action],
                                reward_space=args.reward_space,
                                measurement=measurement,
                                initial_features=state,
                                timeout_sec=args.timeout,
                            )
                            trans = transitions[0]
                            post_state = trans.post
                            raw_reward = trans.step_reward
                        except Exception as e:
                            LOGGER.warning(f"Pass failed {benchmark_uri} ep {ep_idx} step {step_idx} action {action.flag}: {e}")
                            break

                        if post_state is None:
                            break

                        # Compute hybrid reward breakdown
                        reward_info = compute_hybrid_reward(
                            pre_ir=state.ir_instruction_count,
                            post_ir=post_state.ir_instruction_count,
                            pre_size=state.object_text_size_bytes,
                            post_size=post_state.object_text_size_bytes,
                            pre_runtime=state.runtime_median_sec,
                            post_runtime=post_state.runtime_median_sec,
                            weights=weights,
                        )

                        hybrid = reward_info["hybrid_reward_scaled"]
                        cumulative += hybrid if hybrid is not None else 0.0
                        elapsed = time.perf_counter() - started

                        # Termination logic per design
                        termination_reason = ""
                        done = False

                        # No IR change
                        delta_ir = post_state.ir_instruction_count - state.ir_instruction_count
                        if delta_ir == 0 and not args.allow_no_effect:
                            termination_reason = "no_ir_change"
                            done = True
                        # Zero reward (review fix: the flag was parsed but never
                        # set ``done``, so this was dead intent).
                        if hybrid == 0.0 and args.terminate_on_zero_reward:
                            termination_reason = termination_reason or "zero_reward"
                            done = True
                        # Repeated state
                        if post_state.state_id in visited_states:
                            termination_reason = "repeated_state"
                            done = True
                        # Action had no effect flag
                        if trans.action_had_no_effect:
                            if not args.allow_no_effect:
                                termination_reason = "action_no_effect"
                                done = True

                        # Build row
                        row = {
                            "episode_id": episode_id,
                            "transition_key": _transition_key(benchmark_uri, state.state_id, action.action_id, episode_id, step_idx),
                            "run_id": run_id,
                            "generated_at_utc": generated_at,
                            "benchmark_suite": dataset_uri,
                            "compiler_gym_version": compiler_gym_version,
                            "compiler_version": compiler_version,
                            "host_name": socket.gethostname(),
                            "python_version": platform.python_version(),
                            "benchmark_uri": benchmark_uri,
                            "episode_index": ep_idx,
                            "step_index": step_idx,
                            "pass_id": action.action_id,
                            "pass_name": action.name,
                            "pass_flag": action.flag,
                            "pass_position": step_idx,
                            "previous_pass_sequence": json.dumps(previous_passes),
                            "reward_space": args.reward_space,
                            "raw_step_reward": raw_reward,
                            "hybrid_reward": hybrid,
                            "runtime_improvement": reward_info["runtime_improvement"],
                            "ir_improvement": reward_info["ir_improvement"],
                            "size_improvement": reward_info["size_improvement"],
                            "cumulative_reward": cumulative,
                            "pass_success": trans.pass_success,
                            "action_had_no_effect": trans.action_had_no_effect,
                            "done": done or (step_idx == args.max_steps_per_episode - 1),
                            "termination_reason": termination_reason,
                            "step_walltime_sec": elapsed,
                            "error_type": trans.error_type,
                            "error_message": trans.error_message,
                            "pre_state_id": state.state_id,
                            "pre_ir_instruction_count": state.ir_instruction_count,
                            "pre_object_text_size_bytes": state.object_text_size_bytes,
                            "pre_total_basic_blocks": state.total_basic_blocks,
                            "pre_total_functions": state.total_functions,
                            "pre_total_instructions": state.total_instructions,
                            "pre_total_memory_instructions": state.total_memory_instructions,
                            "post_state_id": post_state.state_id,
                            "post_ir_instruction_count": post_state.ir_instruction_count,
                            "post_object_text_size_bytes": post_state.object_text_size_bytes,
                            "post_total_basic_blocks": post_state.total_basic_blocks,
                            "post_total_functions": post_state.total_functions,
                            "post_total_instructions": post_state.total_instructions,
                            "post_total_memory_instructions": post_state.total_memory_instructions,
                            "delta_ir_instruction_count": delta_ir,
                            "delta_object_text_size_bytes": (post_state.object_text_size_bytes - state.object_text_size_bytes) if (state.object_text_size_bytes is not None and post_state.object_text_size_bytes is not None) else "",
                            "pre_runtime_median_sec": state.runtime_median_sec,
                            "post_runtime_median_sec": post_state.runtime_median_sec,
                        }
                        # Autophase
                        for k, v in state.autophase.items():
                            row[f"pre_autophase_{k}"] = v
                        for k, v in post_state.autophase.items():
                            row[f"post_autophase_{k}"] = v

                        writer.writerow(row)
                        handle.flush()
                        total_transitions += 1
                        visited_states.add(post_state.state_id)

                        if done:
                            break

                        # Advance state
                        previous_passes.append(action.action_id)
                        state = post_state

                    # End of episode

                # End benchmark loop

        LOGGER.info(f"RL collection complete: {total_transitions} transitions, skipped_episodes={skipped_episodes}, output={output_path}")
        return output_path

    finally:
        env.close()

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect RL replay buffer via random episodes (NeuroCompiler Phase 5)")
    p.add_argument("--dataset", default="cbench-v1")
    p.add_argument("--benchmark", action="append", default=[])
    p.add_argument("--max-benchmarks", type=int, default=None)
    p.add_argument("--passes", default=None, help="Comma-separated flags; default curated 27")
    p.add_argument("--max-passes", type=int, default=None)
    p.add_argument("--reward-space", default="IrInstructionCountO3")
    p.add_argument("--episodes-per-benchmark", type=int, default=20, help="200 recommended for full 1M transitions")
    p.add_argument("--max-steps-per-episode", type=int, default=10, help="Episode length 10-20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--measure-runtime", action="store_true")
    p.add_argument("--runtime-count", type=int, default=3)
    p.add_argument("--runtime-warmup-count", type=int, default=1)
    p.add_argument("--skip-object-text-size", action="store_true")
    p.add_argument("--reward-weight-runtime", type=float, default=0.6)
    p.add_argument("--reward-weight-ir", type=float, default=0.3)
    p.add_argument("--reward-weight-size", type=float, default=0.1)
    p.add_argument("--allow-no-effect", action="store_true", help="Allow actions with no effect to continue episode")
    p.add_argument("--terminate-on-zero-reward", action="store_true")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output", default=str(DEFAULT_RL_OUTPUT))
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    collect_rl(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
