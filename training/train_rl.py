#!/usr/bin/env python3
"""
Phase 6 — RL Training (PPO / DQN / A2C)

Pipeline:
  State -> Next Pass -> New State -> Repeat
  RL learns ordering automatically.

We train on replay buffer from Phase 5: datasets/replay_buffer/rl_experiences.csv
  Each row: State, Action, Reward, Next State, Done

Design:
  - State: same extractor as SL (56 Autophase + core stats)
  - Action: ~27 curated passes + STOP (optional)
  - Reward: hybrid (0.6 RT + 0.3 IR + 0.1 Size) scaled x100
  - Episode terminates on no IR change / repeated state / max passes

RL Algorithms supported:
  1) DQN (if torch available): Q(s,a) approximator, experience replay, target network
  2) PPO (if stable-baselines3 available): policy gradient with SL prior as initial policy
  3) Fallback: imitation via supervised Q regression (sklearn HistGradientBoosting that predicts Q)

The fallback is sufficient for thesis/hackathon and runs without GPU/CUDA.

Saves:
  models/reinforcement/rl_agent.{pkl, joblib, pt}
  models/reinforcement/rl_config.json
  models/reinforcement/rl_metrics.json

The agent API expected by training/inference.py:
  - predict(state_features, sl_probs=None) -> (action_flag, q_values_dict)
  - predict_distribution(state, sl_probs) is optional
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

from training.common import get_feature_cols, safe_float, load_csv_rows  # noqa: E402

LOGGER = logging.getLogger("train_rl")
DEFAULT_RL_INPUT = PROJECT_ROOT / "datasets" / "replay_buffer" / "rl_experiences.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "reinforcement"

# -------------------------
# Fallback Q-learning with sklearn
# -------------------------
class SklearnDQNAgent:
    """
    Simplistic DQN-like agent using sklearn regressor for Q(s,a).
    Input: state_features + action_id  -> Q-value
    At inference: evaluate all actions, pick argmax (with SL prior mixing).
    """
    def __init__(self, feature_cols: List[str], action_vocab: Dict[str, int], model=None):
        self.feature_cols = feature_cols
        self.action_vocab = action_vocab
        self.inv_vocab = {v:k for k,v in action_vocab.items()}
        self.model = model
        self.pass_flags = list(action_vocab.keys())

    def _encode_state_action(self, state_row: Dict, action_flag: str) -> List[float]:
        feats = []
        for col in self.feature_cols:
            # Support both pre_ prefixed and raw col names in RL buffer
            # RL buffer stores pre_* columns directly
            v = safe_float(state_row.get(col, "") )
            if v is None:
                # Try without pre_? or try alt naming
                # For RL we have pre_autophase_ etc, but feature_cols already contains those
                v = 0.0
            feats.append(v)
        # action encoding
        aid = self.action_vocab.get(action_flag, 0)
        feats.append(float(aid))
        return feats

    def predict_q(self, state_row: Dict, action_flag: str) -> float:
        if self.model is None:
            return 0.0
        x = self._encode_state_action(state_row, action_flag)
        try:
            return float(self.model.predict([x])[0])
        except:
            return 0.0

    def predict(self, state_row: Dict, sl_probs: Optional[Dict[str, float]] = None, epsilon: float = 0.0) -> Tuple[str, Dict[str, float]]:
        """
        Returns best action flag and dict of q values for all actions.
        sl_probs: optional dict flag->prob from supervised model to bias selection
                  Hybrid:  q_hybrid = alpha*q + beta*sl_logit
        """
        q_values = {}
        for flag in self.pass_flags:
            q = self.predict_q(state_row, flag)
            # Mix with SL prior if provided
            if sl_probs and flag in sl_probs:
                # Weighted sum: 0.7*Q + 0.3*SL (normalized)
                # sl_probs expected to be 0..1 probability
                # Bring q to similar scale via tanh or scaling? Simplistic mixing
                q_mixed = 0.7 * q + 0.3 * (sl_probs[flag] * 10.0)  # scale SL prob x10 to match reward scale ~ percent
                q_values[flag] = q_mixed
            else:
                q_values[flag] = q

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            best = random.choice(self.pass_flags)
        else:
            best = max(q_values, key=lambda k: q_values[k])

        return best, q_values

    def save(self, path: Path):
        import joblib
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "action_vocab": self.action_vocab}, path)

    @staticmethod
    def load(path: Path) -> "SklearnDQNAgent":
        import joblib
        data = joblib.load(path)
        agent = SklearnDQNAgent(data["feature_cols"], data["action_vocab"], data["model"])
        return agent

def train_sklearn_dqn(rows: List[Dict[str,str]], feature_cols: List[str], action_vocab: Dict[str,int], args: argparse.Namespace):
    """Q-learning via fitted Q iteration using Bellman backups from replay buffer"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import numpy as np

    gamma = args.gamma

    # Build X, y for initial Q estimate = immediate reward
    # Then perform a few iterations of Bellman update: Q(s,a) = r + gamma * max_a' Q(s',a')
    # We need ability to lookup next state's max Q. We'll do iterative improvement.

    # First pass: gather rows with state and next state mapping
    # For simplicity, create arrays

    def featurize_row(r, action_flag):
        feats = []
        for col in feature_cols:
            v = safe_float(r.get(col, ""))
            feats.append(v if v is not None else 0.0)
        aid = action_vocab.get(action_flag, 0)
        feats.append(float(aid))
        return feats

    # Initial dataset: immediate hybrid_reward
    X = []
    y = []
    for r in rows:
        flag = r.get("pass_flag")
        if not flag:
            continue
        hr = safe_float(r.get("hybrid_reward"))
        if hr is None:
            hr = safe_float(r.get("raw_step_reward"))
        if hr is None:
            hr = 0.0
        X.append(featurize_row(r, flag))
        y.append(hr)

    LOGGER.info(f"[DQN] Initial training set {len(X)} samples")

    model = HistGradientBoostingRegressor(max_iter=args.q_iterations*100, max_depth=8, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    # Fitted Q iteration: refine targets using model itself for next state value
    # Need to be able to estimate V(s') = max_a' Q(s',a')
    # Our rows have pre_* and post_* features. For next state value we need its features.
    # post state features are prefixed post_... we need to map to pre_ equivalent for encoding

    # Build reverse mapping: post_ -> pre_? We'll create a helper that for next state prediction constructs feature dict from post columns
    pre_to_post = {}
    for col in feature_cols:
        # feature_cols come from RL: e.g., pre_autophase_TotalBlocks, so for next state we need post equivalent
        if col.startswith("pre_"):
            post_col = col.replace("pre_", "post_", 1)
            pre_to_post[col] = post_col
        else:
            pre_to_post[col] = col  # fallback

    for iteration in range(args.q_iterations):
        X_new: List[List[float]] = []
        y_new: List[float] = []
        non_done: List[Dict[str, str]] = []
        for r in rows:
            flag = r.get("pass_flag")
            if not flag:
                continue
            r_reward = safe_float(r.get("hybrid_reward"))
            if r_reward is None:
                r_reward = safe_float(r.get("raw_step_reward")) or 0.0
            done = r.get("done", "").lower() in ("true", "1", "yes")
            if done:
                X_new.append(featurize_row(r, flag))
                y_new.append(r_reward)
            else:
                non_done.append(r)

        if non_done:
            # Vectorized Bellman backup: V(s') = max_a' Q(s', a') via batched
            # predictions (one batch per action) instead of per-row predict
            # calls, which are dominated by sklearn call overhead.
            next_q_max: Optional[np.ndarray] = None
            for cand_flag in action_vocab.keys():
                X_next: List[List[float]] = []
                for r in non_done:
                    fake_next_state = {}
                    for pre_c, post_c in pre_to_post.items():
                        fake_next_state[pre_c] = r.get(post_c, "")
                    X_next.append(featurize_row(fake_next_state, cand_flag))
                q_next = np.asarray(model.predict(X_next), dtype=float)
                if next_q_max is None:
                    next_q_max = q_next
                else:
                    next_q_max = np.maximum(next_q_max, q_next)

            for r, qmax in zip(non_done, next_q_max):
                r_reward = safe_float(r.get("hybrid_reward"))
                if r_reward is None:
                    r_reward = safe_float(r.get("raw_step_reward")) or 0.0
                X_new.append(featurize_row(r, r.get("pass_flag", "")))
                y_new.append(r_reward + gamma * float(qmax))

        model.fit(X_new, y_new)
        avg_target = sum(y_new) / len(y_new) if y_new else 0.0
        LOGGER.info(
            f"[DQN] Iteration {iteration+1}/{args.q_iterations} "
            f"samples={len(y_new)} avg target {avg_target:.3f}"
        )

    return model

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 - Train RL Optimization Agent")
    p.add_argument("--input", default=str(DEFAULT_RL_INPUT), help="RL replay buffer CSV")
    p.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--model-type", default="dqn_sklearn", choices=["dqn_sklearn", "dqn_torch", "ppo"], help="RL algorithm")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    p.add_argument("--q-iterations", type=int, default=3, help="Fitted Q iterations")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        # Try alternative default sl dataset for demo
        raise FileNotFoundError(f"RL buffer not found: {input_path}. Generate via scripts/collect_rl_transitions.py first.")

    rows, fieldnames = load_csv_rows(input_path, max_rows=args.max_rows)
    LOGGER.info(f"Loaded {len(rows)} RL transitions, {len(fieldnames)} cols")

    # Determine feature columns: RL uses pre_* fields
    # reuse common logic: look for pre_ autophase and core
    pre_feature_cols = [c for c in fieldnames if c.startswith("pre_autophase_") or c in ("pre_ir_instruction_count","pre_object_text_size_bytes","pre_total_basic_blocks","pre_total_functions","pre_total_instructions","pre_total_memory_instructions")]
    if not pre_feature_cols:
        # fallback try without prefix
        from training.common import get_feature_cols
        pre_feature_cols = get_feature_cols(fieldnames, use_norm=False)
    LOGGER.info(f"Using {len(pre_feature_cols)} state features")

    # Action vocab
    action_vocab = {}
    for r in rows:
        flag = r.get("pass_flag")
        if flag and flag not in action_vocab:
            action_vocab[flag] = len(action_vocab)
    LOGGER.info(f"Action vocab size {len(action_vocab)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "dqn_sklearn":
        model = train_sklearn_dqn(rows, pre_feature_cols, action_vocab, args)
        agent = SklearnDQNAgent(pre_feature_cols, action_vocab, model)
        agent.save(output_dir / "rl_agent.joblib")
        LOGGER.info(f"Saved agent to {output_dir / 'rl_agent.joblib'}")

        # Save config
        config = {
            "model_type": args.model_type,
            "gamma": args.gamma,
            "feature_cols": pre_feature_cols,
            "action_vocab": action_vocab,
            "q_iterations": args.q_iterations,
        }
        (output_dir / "rl_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
        (output_dir / "rl_metrics.json").write_text(json.dumps({"transitions": len(rows), "actions": len(action_vocab)}, indent=2))

    elif args.model_type == "dqn_torch":
        # Placeholder: if torch not available, fallback
        try:
            import torch  # noqa
            LOGGER.info("Torch DQN not fully implemented in this scaffold, falling back to sklearn")
            model = train_sklearn_dqn(rows, pre_feature_cols, action_vocab, args)
            agent = SklearnDQNAgent(pre_feature_cols, action_vocab, model)
            agent.save(output_dir / "rl_agent.joblib")
        except ImportError:
            LOGGER.warning("Torch not available, using sklearn DQN")
            model = train_sklearn_dqn(rows, pre_feature_cols, action_vocab, args)
            agent = SklearnDQNAgent(pre_feature_cols, action_vocab, model)
            agent.save(output_dir / "rl_agent.joblib")

    else:  # ppo
        try:
            import stable_baselines3  # noqa
            LOGGER.info("SB3 PPO path not fully scaffolded, using sklearn fallback for now")
        except ImportError:
            LOGGER.warning("stable_baselines3 not available, using sklearn DQN")
        model = train_sklearn_dqn(rows, pre_feature_cols, action_vocab, args)
        agent = SklearnDQNAgent(pre_feature_cols, action_vocab, model)
        agent.save(output_dir / "rl_agent.joblib")

    print(f"[RL] Training complete. Output dir: {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
