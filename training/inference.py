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
  If models missing, falls back to heuristic: always pick pass with largest immediate reward (greedy) or random for demo.
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
except ImportError:
    from extract_features import MeasurementConfig, extract_features  # type: ignore
    from run_passes import resolve_actions, run_pass_sequence  # type: ignore
    from curated_passes import get_curated_flags  # type: ignore
    from reward import compute_hybrid_reward, RewardWeights  # type: ignore

LOGGER = logging.getLogger("hybrid_inference")
DEFAULT_SL_DIR = PROJECT_ROOT / "models" / "supervised"
DEFAULT_RL_DIR = PROJECT_ROOT / "models" / "reinforcement"

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

    measurement = MeasurementConfig(
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
        initial_state = extract_features(env, measurement)
        current_state = initial_state
        # CompilerGym provides deterministic -O3 baseline cost observations.
        # Runtime -O3 is not exposed directly, so only IR comparison is exact here.
        try:
            raw_o3_ir = env.observation["IrInstructionCountO3"]
            o3_ir_instruction_count = int(raw_o3_ir.reshape(-1)[0]) if hasattr(raw_o3_ir, "reshape") else int(raw_o3_ir[0])
        except Exception as error:
            LOGGER.warning("Could not read IrInstructionCountO3: %s", error)
            o3_ir_instruction_count = None

        pass_sequence = []
        step_details = []
        visited = {initial_state.state_id}
        cumulative_hybrid = 0.0

        if verbose:
            print(f"\n[Hybrid] Optimizing {benchmark_uri}")
            print(f"  Initial IR instrs: {initial_state.ir_instruction_count}, blocks: {initial_state.total_basic_blocks}, funcs: {initial_state.total_functions}")
            print(f"  SL model loaded: {sl_model is not None}, RL agent loaded: {rl_agent is not None}")
            print(f"  Candidates: {len(candidate_flags)} passes")

        for step in range(max_steps):
            # Step 2: SL predicts distribution
            sl_ranked = predict_sl_distribution(
                sl_model,
                sl_feature_cols or [],
                sl_vocab or {},
                current_state,
                candidate_flags,
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

            if rl_agent is not None:
                # Need to construct state row dict similar to training: pre_*
                state_row = current_state.flattened("pre_")
                # RL's feature cols might differ; but we pass state_row directly
                # Agent's predict will internally handle
                try:
                    best_flag, q_values = rl_agent.predict(state_row, sl_probs=sl_probs_dict, epsilon=0.05)
                except Exception as e:
                    LOGGER.warning(f"RL predict failed: {e}, fallback to SL top1")
                    best_flag = sl_ranked[0][0] if sl_ranked else candidate_flags[0]
                    q_values = {}
            else:
                # No RL: pick SL best, with occasional exploration of top-3
                if random.random() < 0.1 and len(sl_ranked) >= 3:
                    best_flag = random.choice([f for f,_,_ in sl_ranked[:3]])
                else:
                    best_flag = sl_ranked[0][0] if sl_ranked else candidate_flags[0]
                q_values = {}

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
                break

            if next_state is None:
                if verbose:
                    print(f"   -> No post state, terminating")
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

            # Termination conditions per design
            if next_state.state_id in visited:
                if verbose:
                    print("   -> Repeated state, terminating")
                break
            if next_state.ir_instruction_count == current_state.ir_instruction_count and reward_info["hybrid_reward_scaled"] == 0.0:
                # Allow 1 zero-effect but terminate after 2 consecutive?
                # For simplicity terminate if no change
                if verbose:
                    print("   -> No IR change, terminating")
                # Don't necessarily terminate immediately? For demo we terminate
                # break
                pass

            visited.add(next_state.state_id)
            current_state = next_state

            if next_state.ir_instruction_count == 0:
                break

        final_state = current_state
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
        }

    finally:
        env.close()

def compare_with_O_levels(benchmark_uri: str, reward_space: str = "IrInstructionCountO3"):
    """
    Quick evaluation vs default -O0, -O2, -O3 baselines using CompilerGym's observations
    We can approximate by checking IrInstructionCountO3 reward? For simplicity we just report.
    """
    try:
        import compiler_gym
    except ImportError:
        return None

    env = compiler_gym.make("llvm-v0")
    try:
        env.reset(benchmark=benchmark_uri, reward_space=reward_space)
        from scripts.extract_features import extract_features, MeasurementConfig
        meas = MeasurementConfig()
        s0 = extract_features(env, meas)
        # Try to get O3 reward? CompilerGym has special handling: reward is improvement over O3?
        # We'll just attempt env.commandline -O3? Simpler: use env.observation["IrInstructionCountO3"]?
        # For now just return s0
        return {"initial_ir": s0.ir_instruction_count}
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
    p.add_argument("--compare-with-o-levels", action="store_true", help="Also evaluate vs -O1/-O2/-O3 if possible")
    p.add_argument("--output", default=None, help="JSON output path for result")
    p.add_argument("--log-level", default="INFO")
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
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nSaved result to {args.output}")

    if args.compare_with_o_levels:
        print("\n[Comparison placeholder] To compare vs -O2/-O3, you would compile bitcode with clang -O2 -O3")
        print("and measure IR size/runtime. This requires clang toolchain in eval environment.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
