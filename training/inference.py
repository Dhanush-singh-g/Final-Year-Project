#!/usr/bin/env python3
"""
Phase 7 — Hybrid Inference

Now a completely new program arrives.

Program A

The optimizer works like this:

Step 1: Program -> LLVM IR -> Extract Features S0
Step 2: Supervised model predicts P(GVN) 0.34, P(LICM) 0.29, etc
Step 3: Instead of allowing all 30 passes, RL considers mainly these high-probability candidates while still retaining exploration
Step 4: RL chooses GVN, LLVM applies it
Step 5: Extract features again S1
Step 6: SL predicts new pass probabilities for updated state (LICM 0.41 becomes most promising)
Step 7: RL chooses again
Repeat: S0 -> GVN -> S1 -> LICM -> S2 -> InstCombine -> S3 -> DCE -> Final Program

This file implements hybrid inference end-to-end.

Usage:
  python training/inference.py --benchmark benchmark://cbench-v1/qsort --max-steps 10
  python training/inference.py --benchmark benchmark://cbench-v1/qsort --compare-with-o_levels

Requires:
  - CompilerGym environment
  - models/supervised/sl_reward_model.joblib (from train_sl.py)
  - models/reinforcement/rl_agent.joblib (from train_rl.py)
If the RL agent is missing, inference falls back to the SL scorer's greedy
best pass (deterministic by default; ``--explore`` adds seeded random
exploration among the available passes).

Episode control (review fixes):
  - Actions that had no effect in a state are masked for the rest of the
    episode (per ``(state_id, action)`` pair); after ``--no-op-limit``
    consecutive no-op actions the episode terminates ("no_effect").
  - Repeated states terminate ("repeated_state"); a selected pass is never
    re-applied while the state is unchanged.
  - ``--enable-stop`` adds an explicit STOP candidate scored with
    ``--stop-prior`` (default 0.0): selecting it terminates cleanly ("stop").
  - Inference is deterministic by default (``--explore-epsilon 0``,
    ``--explore`` off); pass ``--seed`` to reproduce seeded runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.extract_features import MeasurementConfig, extract_features
    from scripts.run_passes import resolve_actions, run_pass_sequence
    from scripts.curated_passes import get_curated_flags
    from scripts.reward import compute_hybrid_reward, RewardWeights
    from training.common import SCALE_FREE_RATIO_COLS, derive_ratio_features
except ImportError:
    from extract_features import MeasurementConfig, extract_features  # type: ignore
    from run_passes import resolve_actions, run_pass_sequence  # type: ignore
    from curated_passes import get_curated_flags  # type: ignore
    from reward import compute_hybrid_reward, RewardWeights  # type: ignore
    from common import SCALE_FREE_RATIO_COLS, derive_ratio_features  # type: ignore

LOGGER = logging.getLogger("hybrid_inference")
DEFAULT_SL_DIR = PROJECT_ROOT / "models" / "supervised"
DEFAULT_RL_DIR = PROJECT_ROOT / "models" / "reinforcement"

# Sentinel candidate flag for the explicit STOP action (see select_action).
STOP_FLAG = "-stop"


def step_is_no_op(
    action_had_no_effect: Optional[bool],
    state_changed: bool,
    delta_ir: int,
    hybrid_reward_scaled: float,
) -> bool:
    """True when the last step produced no measurable progress.

    A step is a no-op when the environment reported the action had no effect,
    the IR state did not move, or both IR count and hybrid reward are
    unchanged. Such actions are masked for the rest of the episode.
    """
    return (
        action_had_no_effect is True
        or not state_changed
        or (delta_ir == 0 and hybrid_reward_scaled == 0.0)
    )


def select_action(
    available: Sequence[str],
    sl_ranked: Sequence[Tuple[str, float, float]],
    *,
    rl_q_values: Optional[Dict[str, float]] = None,
    rl_best: Optional[str] = None,
    stop_prior: Optional[float] = None,
    explore: bool = False,
    rng=None,
) -> Optional[str]:
    """Pick the next pass under per-(state, action) masking.

    Args:
        available: candidate flags not yet tried in the current state.
        sl_ranked: (flag, expected_reward, prob) from the SL scorer, sorted by
            expected reward descending.
        rl_q_values: fitted-Q values over the agent's action space; the
            agent's own pick (``rl_best``) wins only when it is still
            available, otherwise the masked Q argmax is used.
        stop_prior: SL-only fallback — when given and no RL agent is present,
            STOP wins if it scores above every available SL candidate.
            With an RL agent, the *learned* Q(STOP) (a real member of the
            agent's action vocabulary) is used instead: STOP is selected
            exactly when its Q-value beats every available pass.
        explore: seeded 10% random exploration among available passes.

    Returns the chosen flag, STOP_FLAG, or None when nothing is available.
    """
    if not available:
        return None
    rng = rng if rng is not None else random
    sl_scores = {flag: score for flag, score, _ in sl_ranked if flag in available}
    if explore and len(available) >= 2 and rng.random() < 0.1:
        return rng.choice(list(available))
    if rl_q_values is not None:
        masked = {flag: q for flag, q in rl_q_values.items() if flag in available}
        # Learned STOP: when the agent's Q(STOP) beats every available pass,
        # terminate cleanly. STOP is never in ``available`` (it is not a real
        # LLVM pass), so it must be scored separately.
        stop_q = rl_q_values.get(STOP_FLAG)
        if stop_q is not None and masked:
            best_avail_q = max(masked.values())
            if stop_q > best_avail_q:
                return STOP_FLAG
        if masked and rl_best in masked:
            return rl_best
        if masked:
            return max(masked, key=masked.get)
        # RL returned nothing usable (e.g. an empty Q dict): fall through to SL.
    if stop_prior is not None and sl_scores:
        best_sl = max(sl_scores, key=sl_scores.get)
        if stop_prior > sl_scores[best_sl]:
            return STOP_FLAG
    return max(sl_scores, key=sl_scores.get) if sl_scores else available[0]


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    # temperature: higher = more uniform
    max_score = max(scores)
    exps = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(exps)
    return [e/total for e in exps] if total>0 else [1.0/len(scores)]*len(scores)

def load_sl_model(model_dir: Path):
    model_dir = Path(model_dir)
    model_path_joblib = model_dir / "sl_reward_model.joblib"
    model_path_pkl = model_dir / "sl_reward_model.pkl"
    vocab_path = model_dir / "sl_action_vocab.json"
    feature_path = model_dir / "sl_feature_columns.json"
    pass_list_path = model_dir / "sl_pass_list.json"

    model = None
    feature_cols = None
    action_vocab = None
    pass_list = None

    try:
        if model_path_joblib.exists():
            import joblib
            model = joblib.load(model_path_joblib)
            LOGGER.info(f"Loaded SL model {model_path_joblib}")
        elif model_path_pkl.exists():
            import pickle
            with model_path_pkl.open("rb") as f:
                model = pickle.load(f)
            LOGGER.info(f"Loaded SL model {model_path_pkl}")
    except Exception as e:
        LOGGER.warning(f"Failed to load SL model: {e}")

    feature_meta = {}
    try:
        if vocab_path.exists():
            action_vocab = json.loads(vocab_path.read_text())
        if feature_path.exists():
            feature_meta = json.loads(feature_path.read_text())
            feature_cols = feature_meta.get("feature_cols")
        if pass_list_path.exists():
            pass_list = json.loads(pass_list_path.read_text())
    except Exception as e:
        LOGGER.warning(f"Failed to load SL meta: {e}")

    return model, feature_cols, action_vocab, pass_list, feature_meta

def load_rl_agent(model_dir: Path):
    model_dir = Path(model_dir)
    agent_path = model_dir / "rl_agent.joblib"
    config_path = model_dir / "rl_config.json"
    config = None
    agent = None
    try:
        if config_path.exists():
            config = json.loads(config_path.read_text())
        if agent_path.exists():
            # Lazy import to avoid dependency
            sys.path.insert(0, str(PROJECT_ROOT / "training"))
            from train_rl import SklearnDQNAgent  # type: ignore
            agent = SklearnDQNAgent.load(agent_path)
            LOGGER.info(f"Loaded RL agent {agent_path}")
    except Exception as e:
        LOGGER.warning(f"Failed to load RL agent: {e}")
    return agent, config

def featurize_for_sl(
    pre_state, feature_cols, action_vocab, action_flag, feature_meta=None
):
    """Encode inference inputs exactly as train_sl.py encoded them."""
    row = pre_state.flattened("pre_")
    feats = []
    for col in feature_cols:
        # Runtime SL training defaults to raw pre_* features. If a model was
        # trained with normalized features, refuse silent all-zero inference.
        if col.startswith("norm_"):
            raise RuntimeError(
                "This model expects normalized features, but online inference "
                "has no normalization parameters. Retrain without --use-normalized."
            )
        v = row.get(col, 0)
        try:
            feats.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            feats.append(0.0)

    encoding = (feature_meta or {}).get("action_encoding", "one_hot")
    if encoding == "one_hot":
        if action_flag not in action_vocab:
            raise ValueError(f"Pass {action_flag!r} was not present during training")
        one_hot = [0.0] * len(action_vocab)
        one_hot[action_vocab[action_flag]] = 1.0
        feats.extend(one_hot)
    else:
        # Legacy models only. New models always use one-hot encoding.
        feats.append(float(action_vocab.get(action_flag, 0)))
    return feats


def predict_sl_distribution(
    sl_model, feature_cols, action_vocab, pre_state, candidate_flags,
    temperature=1.0, feature_meta=None
):
    """
    Returns list of (flag, expected_reward, prob) sorted descending by reward.
    """
    if sl_model is None or feature_cols is None:
        # Fallback heuristic: uniform random
        scores = [random.random() for _ in candidate_flags]
        probs = softmax(scores, temperature)
        return sorted([(f, s, p) for f,s,p in zip(candidate_flags, scores, probs)], key=lambda x: x[1], reverse=True)

    scores = []
    for flag in candidate_flags:
        try:
            feats = featurize_for_sl(
                pre_state, feature_cols, action_vocab, flag, feature_meta
            )
            score = float(sl_model.predict([feats])[0])
        except Exception as e:
            LOGGER.warning("SL scoring failed for %s: %s", flag, e)
            score = float("-inf")
        scores.append(score)

    probs = softmax(scores, temperature)
    ranked = sorted([(f, sc, pr) for f, sc, pr in zip(candidate_flags, scores, probs)], key=lambda x: x[1], reverse=True)
    return ranked

def hybrid_optimize_benchmark(
    benchmark_uri: str,
    max_steps: int = 10,
    sl_dir: Path = DEFAULT_SL_DIR,
    rl_dir: Path = DEFAULT_RL_DIR,
    reward_space: str = "IrInstructionCountO3",
    measure_runtime: bool = False,
    verbose: bool = True,
    dump_bitcode_to: Optional[Path] = None,
    no_op_limit: int = 1,
    enable_stop: bool = False,
    stop_prior: float = 0.0,
    explore_epsilon: float = 0.0,
    explore: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Run hybrid optimization on one benchmark URI.
    Returns dict with pass sequence and improvements.
    """
    try:
        import compiler_gym
    except ImportError:
        raise SystemExit("CompilerGym not available. Activate neurocompiler env.")

    sl_model, sl_feature_cols, sl_vocab, sl_pass_list, sl_feature_meta = load_sl_model(sl_dir)
    rl_agent, rl_config = load_rl_agent(rl_dir)
    if sl_model is None:
        raise FileNotFoundError(
            f"No trained SL model found in {sl_dir}. Run training/train_sl.py first."
        )

    # Choose candidate action set: use curated 27 or from SL pass list if available
    candidate_flags = sl_pass_list or get_curated_flags()

    # Per-step extraction deliberately runs WITHOUT the Runtime observation:
    # measuring runtime rebuilds the executable and can transiently fail in a
    # way that leaves the environment unusable, invalidating an otherwise good
    # pass. The SL model only consumes static pre_* features and per-step
    # rewards do not affect action selection, so runtime is measured once at
    # the start and once at the end instead.
    measurement = MeasurementConfig(
        measure_runtime=False,
        runtime_count=3,
        runtime_warmup_count=1,
        measure_buildtime=False,
        collect_object_text_size=True,
    )
    measurement_full = MeasurementConfig(
        measure_runtime=measure_runtime,
        runtime_count=3,
        runtime_warmup_count=1,
        measure_buildtime=False,
        collect_object_text_size=True,
    )
    weights = RewardWeights()

    env = compiler_gym.make("llvm-v0")
    try:
        env.reset(benchmark=benchmark_uri, reward_space=reward_space)
        try:
            initial_state = extract_features(env, measurement_full)
        except Exception as error:
            # A failed runtime measurement can break the environment; recover
            # by resetting and extracting without runtime.
            LOGGER.warning(
                "Initial runtime measurement failed (%s); continuing without runtime",
                error,
            )
            env.reset(benchmark=benchmark_uri, reward_space=reward_space)
            initial_state = extract_features(env, measurement)
        current_state = initial_state
        # CompilerGym provides deterministic -O3 baseline cost observations.
        # Runtime -O3 is not exposed directly, so only IR comparison is exact here.
        try:
            raw_o3_ir = env.observation["IrInstructionCountO3"]
            if isinstance(raw_o3_ir, int):
                o3_ir_instruction_count = raw_o3_ir
            elif hasattr(raw_o3_ir, "reshape"):
                o3_ir_instruction_count = int(raw_o3_ir.reshape(-1)[0])
            else:
                o3_ir_instruction_count = int(raw_o3_ir[0])
        except Exception as error:
            LOGGER.warning("Could not read IrInstructionCountO3: %s", error)
            o3_ir_instruction_count = None

        pass_sequence = []
        step_details = []
        visited = {initial_state.state_id}
        cumulative_hybrid = 0.0
        # Per-(state_id, action) mask: a no-op action is never re-tried while
        # the state is unchanged (review fix — previously the same pass could
        # be selected repeatedly and wasted the fixed pass budget).
        tried_in_state: Dict[str, set] = {}
        consecutive_no_op = 0
        termination_reason: Optional[str] = None
        explore_rng = random.Random(seed) if (explore or explore_epsilon > 0) else None

        if verbose:
            print(f"\n[Hybrid] Optimizing {benchmark_uri}")
            print(f"  Initial IR instrs: {initial_state.ir_instruction_count}, blocks: {initial_state.total_basic_blocks}, funcs: {initial_state.total_functions}")
            print(f"  SL model loaded: {sl_model is not None}, RL agent loaded: {rl_agent is not None}")
            print(f"  Candidates: {len(candidate_flags)} passes")

        for step in range(max_steps):
            tried = tried_in_state.setdefault(current_state.state_id, set())
            available = [flag for flag in candidate_flags if flag not in tried]
            if not available:
                termination_reason = "all_actions_tried"
                if verbose:
                    print("   -> All candidate actions tried in this state, terminating")
                break

            # Step 2: SL predicts distribution over the masked candidate set.
            sl_ranked = predict_sl_distribution(
                sl_model,
                sl_feature_cols or [],
                sl_vocab or {},
                current_state,
                available,
                temperature=5.0,
                feature_meta=sl_feature_meta,
            )

            if verbose:
                top5 = sl_ranked[:5]
                print(f"\n Step {step} | State {current_state.state_id[:8]} | IR {current_state.ir_instruction_count}")
                print(f"   SL top: {[(f'{fl}:{sc:.2f}({pr:.2f})') for fl,sc,pr in top5]}")

            # Step 3 & 4: RL considers high-prob candidates + exploration
            # If RL agent present: it gets sl_probs dict and current state row
            sl_probs_dict = {flag: prob for flag, _, prob in sl_ranked}
            rl_best: Optional[str] = None
            q_values: Dict[str, float] = {}
            if rl_agent is not None:
                # Need to construct state row dict similar to training: pre_*
                state_row = current_state.flattened("pre_")
                # Scale-free derived features (per-function/per-instruction
                # ratios): injected only when the agent was trained with them,
                # so raw-count and other agents are unaffected.
                if any(c in rl_agent.feature_cols for c in SCALE_FREE_RATIO_COLS):
                    state_row.update(derive_ratio_features(state_row, "pre_"))
                try:
                    rl_best, q_values = rl_agent.predict(
                        state_row, sl_probs=sl_probs_dict, epsilon=explore_epsilon
                    )
                except Exception as error:
                    LOGGER.warning(
                        "RL predict failed (%s); falling back to SL top-1", error
                    )
                    q_values = {}

            # Step 5: masked selection (deterministic by default).
            best_flag = select_action(
                available,
                sl_ranked,
                rl_q_values=q_values if rl_agent is not None else None,
                rl_best=rl_best,
                stop_prior=stop_prior if enable_stop else None,
                explore=explore,
                rng=explore_rng,
            )
            if best_flag is None:
                termination_reason = "all_actions_tried"
                if verbose:
                    print("   -> All candidate actions tried in this state, terminating")
                break
            if best_flag == STOP_FLAG:
                termination_reason = "stop"
                if verbose:
                    print("   -> STOP selected, terminating")
                break

            # Apply pass
            actions = resolve_actions(env, [best_flag])
            if not actions:
                LOGGER.warning(f"Action {best_flag} not found in env, skipping")
                continue

            try:
                transitions = run_pass_sequence(env, actions, reward_space=reward_space, measurement=measurement, initial_features=current_state)
                trans = transitions[0]
                next_state = trans.post
            except Exception as e:
                LOGGER.warning(f"Failed to apply {best_flag}: {e}")
                termination_reason = "action_failed"
                break

            if next_state is None:
                if verbose:
                    print(f"   -> No post state, terminating")
                termination_reason = "post_state_none"
                break

            # Compute rewards
            reward_info = compute_hybrid_reward(
                pre_ir=current_state.ir_instruction_count,
                post_ir=next_state.ir_instruction_count,
                pre_size=current_state.object_text_size_bytes,
                post_size=next_state.object_text_size_bytes,
                pre_runtime=current_state.runtime_median_sec,
                post_runtime=next_state.runtime_median_sec,
                weights=weights,
            )

            cumulative_hybrid += reward_info["hybrid_reward_scaled"]

            if verbose:
                print(f"   -> Chose {best_flag}, delta IR {next_state.ir_instruction_count - current_state.ir_instruction_count}, hybrid {reward_info['hybrid_reward_scaled']:.3f}, cum {cumulative_hybrid:.3f}")

            step_details.append({
                "step": step,
                "pre_state_id": current_state.state_id,
                "post_state_id": next_state.state_id,
                "chosen_pass": best_flag,
                "delta_ir": next_state.ir_instruction_count - current_state.ir_instruction_count,
                "ir_improvement": reward_info["ir_improvement"],
                "hybrid_reward": reward_info["hybrid_reward_scaled"],
                "sl_top3": sl_ranked[:3],
                "q_values_top3": sorted(q_values.items(), key=lambda kv: kv[1], reverse=True)[:3] if q_values else [],
            })

            pass_sequence.append(best_flag)

            # Termination conditions per design (review fix: the previous
            # "no IR change" branch printed 'terminating' but did not break,
            # so no-op/redundant actions could repeat for the whole budget).
            no_op = step_is_no_op(
                trans.action_had_no_effect,
                next_state.state_id != current_state.state_id,
                next_state.ir_instruction_count - current_state.ir_instruction_count,
                reward_info["hybrid_reward_scaled"],
            )
            if no_op:
                consecutive_no_op += 1
                # Mask this (state, action) pair for the rest of the episode.
                tried.add(best_flag)
                if consecutive_no_op >= no_op_limit:
                    termination_reason = "no_effect"
                    if verbose:
                        print(
                            f"   -> No effect for {no_op_limit} consecutive "
                            f"action(s), terminating"
                        )
                    break
            else:
                consecutive_no_op = 0

            if next_state.ir_instruction_count == 0:
                termination_reason = "zero_ir"
                break

            # A repeated state is no longer an early-termination signal:
            # per-(state, action) masking already prevents looping (every
            # action tried in that state is masked, so the episode ends via
            # ``all_actions_tried`` / learned STOP / max_steps instead).
            # Terminating early here is what capped sequences at 2-3 passes
            # and made the longer-horizon budget inert.
            visited.add(next_state.state_id)
            current_state = next_state

        if termination_reason is None:
            termination_reason = "max_steps"

        final_state = current_state
        if measure_runtime:
            try:
                final_state = extract_features(env, measurement_full)
            except Exception as error:
                LOGGER.warning(
                    "Final runtime measurement failed (%s); reporting null runtime",
                    error,
                )
        ir_reduction = initial_state.ir_instruction_count - final_state.ir_instruction_count
        ir_reduction_pct = (ir_reduction / initial_state.ir_instruction_count * 100) if initial_state.ir_instruction_count else 0
        initial_runtime = initial_state.runtime_median_sec
        final_runtime = final_state.runtime_median_sec
        runtime_speedup = (
            initial_runtime / final_runtime
            if initial_runtime is not None and final_runtime is not None and final_runtime > 0
            else None
        )
        runtime_improvement_pct = (
            100.0 * (initial_runtime - final_runtime) / initial_runtime
            if initial_runtime is not None and final_runtime is not None and initial_runtime > 0
            else None
        )
        hybrid_vs_o3_ir_pct = (
            100.0 * (o3_ir_instruction_count - final_state.ir_instruction_count)
            / o3_ir_instruction_count
            if o3_ir_instruction_count is not None and o3_ir_instruction_count > 0
            else None
        )

        if dump_bitcode_to is not None:
            try:
                dump_bitcode_to = Path(dump_bitcode_to)
                dump_bitcode_to.parent.mkdir(parents=True, exist_ok=True)
                raw = env.observation["Bitcode"]
                if hasattr(raw, "tobytes"):
                    raw = raw.tobytes()
                if isinstance(raw, bytes):
                    dump_bitcode_to.write_bytes(raw)
                else:
                    dump_bitcode_to.write_text(str(raw))
                LOGGER.info("Dumped final hybrid bitcode to %s", dump_bitcode_to)
            except Exception as error:  # dumping is best-effort
                LOGGER.warning("Failed to dump final bitcode to %s: %s", dump_bitcode_to, error)

        if verbose:
            print(f"\n[Hybrid] Final sequence ({len(pass_sequence)}): {' -> '.join(pass_sequence)}")
            print(f"  Initial IR: {initial_state.ir_instruction_count} -> Final IR: {final_state.ir_instruction_count} (reduction {ir_reduction} = {ir_reduction_pct:.2f}%)")
            if o3_ir_instruction_count is not None:
                print(
                    f"  -O3 IR baseline: {o3_ir_instruction_count}; hybrid vs -O3: "
                    f"{hybrid_vs_o3_ir_pct:+.2f}% (positive means fewer instructions)"
                )
            if runtime_speedup is not None:
                print(
                    f"  Runtime: {initial_runtime:.6f}s -> {final_runtime:.6f}s "
                    f"(speedup {runtime_speedup:.4f}x, improvement {runtime_improvement_pct:.2f}%)"
                )
            print(f"  Cum hybrid reward: {cumulative_hybrid:.3f}")

        return {
            "benchmark_uri": benchmark_uri,
            "initial_ir": initial_state.ir_instruction_count,
            "final_ir": final_state.ir_instruction_count,
            "ir_reduction": ir_reduction,
            "ir_reduction_pct": ir_reduction_pct,
            "initial_runtime_median_sec": initial_runtime,
            "final_runtime_median_sec": final_runtime,
            "runtime_speedup": runtime_speedup,
            "runtime_improvement_pct": runtime_improvement_pct,
            "o3_ir_instruction_count": o3_ir_instruction_count,
            "hybrid_vs_o3_ir_pct": hybrid_vs_o3_ir_pct,
            "sl_target": sl_feature_meta.get("target"),
            "pass_sequence": pass_sequence,
            "steps": step_details,
            "cumulative_hybrid": cumulative_hybrid,
            "initial_state_id": initial_state.state_id,
            "final_state_id": final_state.state_id,
            "termination_reason": termination_reason,
        }

    finally:
        env.close()

def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 - Hybrid Inference (SL-guided RL)")
    p.add_argument("--benchmark", default="benchmark://cbench-v1/qsort", help="Benchmark URI")
    p.add_argument("--max-steps", type=int, default=10, help="Episode length 10-20")
    p.add_argument("--sl-model-dir", default=str(DEFAULT_SL_DIR))
    p.add_argument("--rl-model-dir", default=str(DEFAULT_RL_DIR))
    p.add_argument("--reward-space", default="IrInstructionCountO3")
    p.add_argument("--measure-runtime", action="store_true")
    p.add_argument(
        "--no-op-limit", type=int, default=1,
        help="Terminate after N consecutive no-effect actions (default 1)",
    )
    p.add_argument(
        "--enable-stop", action="store_true",
        help="SL-only STOP fallback (no RL agent): add an explicit STOP "
        "candidate that terminates when it scores best. With an RL agent, the "
        "learned Q(STOP) action is used automatically.",
    )
    p.add_argument(
        "--stop-prior", type=float, default=0.0,
        help="SL-only STOP score vs SL expected rewards (default 0.0)",
    )
    p.add_argument(
        "--explore-epsilon", type=float, default=0.0,
        help="RL epsilon-greedy exploration (default 0 = deterministic)",
    )
    p.add_argument(
        "--explore", action="store_true",
        help="SL-only mode: seeded 10%% random exploration among available passes",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for seeded exploration")
    p.add_argument(
        "--output", default=None, help="JSON output path for result"
    )
    p.add_argument("--log-level", default="INFO")
    # Runtime-vs-O3 comparison is NOT exposed here: it is measured by the
    # controlled executable harness (evaluation/o3_runtime_harness.py, runbook
    # section 10). CompilerGym has no -O3 Runtime observation.
    return p.parse_args()

def main():
    args = parse_args()
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    result = hybrid_optimize_benchmark(
        benchmark_uri=args.benchmark,
        max_steps=args.max_steps,
        sl_dir=Path(args.sl_model_dir),
        rl_dir=Path(args.rl_model_dir),
        reward_space=args.reward_space,
        measure_runtime=args.measure_runtime,
        verbose=True,
        no_op_limit=args.no_op_limit,
        enable_stop=args.enable_stop,
        stop_prior=args.stop_prior,
        explore_epsilon=args.explore_epsilon,
        explore=args.explore,
        seed=args.seed,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nSaved result to {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
