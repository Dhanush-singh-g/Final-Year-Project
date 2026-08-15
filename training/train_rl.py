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

RL algorithm implemented:
  fitted-Q regression with sklearn HistGradientBoosting (works without GPU/CUDA).

Actions are ONE-HOT encoded (matching the SL encoding): a single numeric
action id would impose an artificial ordinal relationship between unrelated
LLVM passes (review fix). "dqn_torch" and "ppo" are NOT implemented and are
rejected by the CLI rather than silently falling back.

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

# Sentinel flag for the learned STOP action. It is a real member of the RL
# agent's action vocabulary: fitted-Q learns Q(state, STOP) from synthetic
# terminal transitions (reward 0, done=True), so the agent stops exactly when
# every available pass is expected to hurt more than stopping.
STOP_FLAG = "-stop"

# -------------------------
# Fitted Q-learning with sklearn
# -------------------------
def encode_state_action(
    state_row: Dict,
    action_flag: str,
    feature_cols: List[str],
    action_vocab: Dict[str, int],
    action_encoding: str = "one_hot",
) -> List[float]:
    """Encode (state, action) identically for training and inference.

    State features come from the raw ``pre_*`` columns; the action is one-hot
    over the (deterministically sorted) action vocabulary. The old "ordinal"
    encoding is kept for backward compatibility with pre-fix artifacts only.
    """
    feats: List[float] = []
    for col in feature_cols:
        v = safe_float(state_row.get(col, ""))
        feats.append(v if v is not None else 0.0)
    if action_encoding == "ordinal":
        feats.append(float(action_vocab.get(action_flag, 0)))
    else:  # one_hot
        one_hot = [0.0] * len(action_vocab)
        index = action_vocab.get(action_flag)
        if index is not None:
            one_hot[index] = 1.0
        feats.extend(one_hot)
    return feats


class SklearnDQNAgent:
    """
    DQN-like agent using a sklearn regressor for Q(s,a).
    Input: state features + one-hot action  -> Q-value
    At inference: evaluate all actions, pick argmax (with SL prior mixing).
    """
    def __init__(
        self,
        feature_cols: List[str],
        action_vocab: Dict[str, int],
        model=None,
        action_encoding: str = "one_hot",
    ):
        self.feature_cols = feature_cols
        self.action_vocab = action_vocab
        self.inv_vocab = {v: k for k, v in action_vocab.items()}
        self.model = model
        self.action_encoding = action_encoding
        self.pass_flags = list(action_vocab.keys())

    def _encode_state_action(self, state_row: Dict, action_flag: str) -> List[float]:
        return encode_state_action(
            state_row,
            action_flag,
            self.feature_cols,
            self.action_vocab,
            self.action_encoding,
        )

    def predict_q(self, state_row: Dict, action_flag: str) -> float:
        if self.model is None:
            return 0.0
        x = self._encode_state_action(state_row, action_flag)
        try:
            return float(self.model.predict([x])[0])
        except Exception:
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
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "action_vocab": self.action_vocab,
                "action_encoding": self.action_encoding,
            },
            path,
        )

    @staticmethod
    def load(path: Path) -> "SklearnDQNAgent":
        import joblib
        data = joblib.load(path)
        # Legacy artifacts predate the one-hot encoding; treat them as ordinal.
        return SklearnDQNAgent(
            data["feature_cols"],
            data["action_vocab"],
            data["model"],
            action_encoding=data.get("action_encoding", "ordinal"),
        )

def synthesize_stop_transitions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append one synthetic STOP transition per terminal state of the replay
    buffer.

    A fitted-Q agent can only learn Q(STOP) if STOP appears in the training
    data. Real episodes terminate on no-op / repeated-state / max-steps; the
    final transition of each episode is that terminal state. For every such
    row we add a ``-stop`` row with reward 0.0 and done=True, so the agent
    learns that stopping is worth ~0 (i.e. stop when all candidate passes are
    expected to be net-negative).
    """
    out: List[Dict[str, str]] = []
    terminal_rows = [r for r in rows if r.get("done", "").lower() in ("true", "1", "yes")]
    for r in terminal_rows:
        stop = dict(r)
        stop["pass_flag"] = STOP_FLAG
        stop["pass_name"] = STOP_FLAG
        stop["pass_id"] = ""
        stop["done"] = "True"
        stop["hybrid_reward"] = "0.0"
        stop["raw_step_reward"] = "0.0"
        # Terminal: post state is the pre state (no pass applied).
        for col in list(stop.keys()):
            if col.startswith("post_"):
                pre_col = "pre_" + col[len("post_"):]
                if pre_col in stop:
                    stop[col] = stop[pre_col]
        out.append(stop)
    return out


def fqi_target(
    reward: float,
    gamma: float,
    qmax_next: Optional[float],
    done: bool,
    is_self_loop: bool,
) -> float:
    """Bellman target for one transition (multi-step RL design, Phase 6).

    - terminal (``done=True``): ``y = r``
    - non-terminal with no usable next state or a self-loop (``s' == s``):
      ``y = r`` — never bootstrap through a state-identical transition
    - non-terminal: ``y = r + gamma * max_{a' in avail(s')} Q(s', a')``

    ``qmax_next`` must already include the STOP action (the option to stop,
    whose learned value is ~0), so the backup is
    ``r + gamma * max(0, max over available passes)``.
    """
    if done or is_self_loop or qmax_next is None:
        return reward
    return reward + gamma * qmax_next


def _parse_available_actions(r: Dict[str, str]) -> Optional[set]:
    """Unmasked candidate set at the transition's POST state, plus STOP.

    Returns None for legacy buffers without the column, meaning "fall back to
    the full action vocabulary" (the previous behaviour).
    """
    raw = (r.get("available_actions") or "").strip()
    if not raw:
        return None
    try:
        flags = json.loads(raw)
    except Exception:
        return None
    if not isinstance(flags, list) or not flags:
        return None
    return set(flags) | {STOP_FLAG}


def train_sklearn_dqn(rows: List[Dict[str,str]], feature_cols: List[str], action_vocab: Dict[str,int], args: argparse.Namespace):
    """Q-learning via fitted Q iteration with genuine Bellman backups.

    The runtime replay buffers collected by ``generate_rl_episodes.py`` are
    chained multi-step trajectories with real ``done`` flags (non-terminal
    transitions are ``done=False``), so the non-terminal branch of the backup
    is live and future value actually propagates:

        y = r                        if done / self-loop / no next state
        y = r + gamma * max_{a'} Q_prev(s', a')   otherwise

    The max ranges over the transition's ``available_actions`` (the unmasked
    candidate set at s') plus the STOP action, so the backup encodes the
    option to stop. Legacy one-step buffers (all ``done=True``) still train
    correctly as the myopic case: the bootstrap branch is simply empty.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    import numpy as np

    gamma = args.gamma
    action_encoding = getattr(args, "action_encoding", "one_hot")

    def featurize_row(r, action_flag):
        return encode_state_action(
            r, action_flag, feature_cols, action_vocab, action_encoding
        )

    def row_reward(r: Dict[str, str]) -> Optional[float]:
        v = safe_float(r.get("hybrid_reward"))
        if v is None:
            v = safe_float(r.get("raw_step_reward"))
        return v

    # Build X, y for the initial Q estimate = immediate reward (iteration 0
    # bootstrap is the myopic model; later iterations use the previous model).
    X: List[List[float]] = []
    y: List[float] = []
    usable: List[Dict[str, str]] = []
    for r in rows:
        flag = r.get("pass_flag")
        if not flag:
            continue
        reward = row_reward(r)
        if reward is None:
            continue
        X.append(featurize_row(r, flag))
        y.append(reward)
        usable.append(r)
    rows = usable

    LOGGER.info(f"[DQN] Initial training set {len(X)} samples")

    model = HistGradientBoostingRegressor(max_iter=args.q_iterations*100, max_depth=8, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    # Map pre_ feature columns to their post_ counterparts so the next state's
    # features can be reconstructed from a row's post_* columns.
    pre_to_post = {}
    for col in feature_cols:
        if col.startswith("pre_"):
            pre_to_post[col] = col.replace("pre_", "post_", 1)
        else:
            pre_to_post[col] = col

    for iteration in range(args.q_iterations):
        X_new: List[List[float]] = []
        y_new: List[float] = []
        # Non-terminal rows that need a bootstrapped target: (row, avail_set).
        pending: List[Tuple[Dict[str, str], Optional[set]]] = []
        for r in rows:
            flag = r.get("pass_flag")
            if not flag:
                continue
            reward = row_reward(r)
            if reward is None:
                continue
            done = r.get("done", "").lower() in ("true", "1", "yes")
            post_id = (r.get("post_state_id") or "").strip()
            is_self_loop = bool(post_id) and post_id == (r.get("pre_state_id") or "").strip()
            if done or is_self_loop or not post_id:
                X_new.append(featurize_row(r, flag))
                y_new.append(reward)
            else:
                pending.append((r, _parse_available_actions(r)))

        if pending:
            # Vectorized Bellman backup: for each candidate action, predict
            # Q_prev(s', a') only for the rows where that action is unmasked.
            qmax = np.full(len(pending), float("-inf"))
            candidates: set = set()
            for _, avail in pending:
                if avail is None:
                    candidates |= set(action_vocab.keys())
                else:
                    candidates |= avail
            for cand_flag in sorted(candidates):
                idxs: List[int] = []
                X_next: List[List[float]] = []
                for i, (r, avail) in enumerate(pending):
                    if avail is not None and cand_flag not in avail:
                        continue
                    fake_next_state = {}
                    for pre_c, post_c in pre_to_post.items():
                        fake_next_state[pre_c] = r.get(post_c, "")
                    X_next.append(featurize_row(fake_next_state, cand_flag))
                    idxs.append(i)
                if not X_next:
                    continue
                q_next = np.asarray(model.predict(X_next), dtype=float)
                for i, qi in zip(idxs, q_next):
                    if qi > qmax[i]:
                        qmax[i] = float(qi)

            for i, (r, _avail) in enumerate(pending):
                reward = row_reward(r)
                if reward is None:
                    continue
                qmax_i: Optional[float] = None if qmax[i] == float("-inf") else float(qmax[i])
                X_new.append(featurize_row(r, r.get("pass_flag", "")))
                y_new.append(fqi_target(reward, gamma, qmax_i, done=False, is_self_loop=False))

        model.fit(X_new, y_new)
        avg_target = sum(y_new) / len(y_new) if y_new else 0.0
        min_target = min(y_new) if y_new else 0.0
        max_target = max(y_new) if y_new else 0.0
        bootstrapped = len(pending)
        LOGGER.info(
            f"[DQN] Iteration {iteration+1}/{args.q_iterations} "
            f"samples={len(y_new)} bootstrapped={bootstrapped} "
            f"avg target {avg_target:.3f} min {min_target:.3f} max {max_target:.3f}"
        )

    return model

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 - Train RL Optimization Agent")
    p.add_argument("--input", default=str(DEFAULT_RL_INPUT), help="RL replay buffer CSV")
    p.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument(
        "--model-type",
        default="dqn_sklearn",
        choices=["dqn_sklearn", "dqn_torch", "ppo"],
        help="RL algorithm. Only 'dqn_sklearn' is implemented; dqn_torch/ppo are rejected.",
    )
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor (multi-step RL design)")
    p.add_argument("--q-iterations", type=int, default=20, help="Fitted Q iterations")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated override of the state feature columns (default: "
        "auto-derive pre_autophase_* + the core IR/size/block/function "
        "counts). Use for scale-free / derived state representations, e.g. "
        "--feature-cols=pre_autophase_TotalInsts,pre_ir_per_func.",
    )
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
    if args.feature_cols:
        override = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        missing = [c for c in override if c not in fieldnames]
        if missing:
            raise SystemExit(
                f"--feature-cols columns missing from buffer: {missing}"
            )
        pre_feature_cols = override
        LOGGER.info(f"Using --feature-cols override: {len(pre_feature_cols)} state features")
    else:
        LOGGER.info(f"Using {len(pre_feature_cols)} state features")

    # Action vocab: deterministically sorted so one-hot positions are stable
    # across runs and match the SL vocabulary ordering convention. STOP is a
    # real member of the vocabulary so the agent can learn when to stop.
    action_vocab = {}
    for r in rows:
        flag = r.get("pass_flag")
        if flag and flag not in action_vocab:
            action_vocab[flag] = len(action_vocab)
    action_vocab = {flag: action_vocab[flag] for flag in sorted(action_vocab)}
    if STOP_FLAG not in action_vocab:
        action_vocab[STOP_FLAG] = len(action_vocab)
    LOGGER.info(f"Action vocab size {len(action_vocab)} (includes {STOP_FLAG!r})")

    # STOP transitions: multi-step buffers collected by
    # scripts/generate_rl_episodes.py contain REAL terminal STOP rows (episodes
    # that actually ended by choosing STOP). Only legacy one-step buffers need
    # the synthetic augmentation so Q(state, STOP) is learnable.
    real_stop_count = sum(1 for r in rows if (r.get("pass_flag") or "") == STOP_FLAG)
    if real_stop_count:
        LOGGER.info(f"Buffer contains {real_stop_count} real STOP transitions; skipping synthesis")
    else:
        stop_rows = synthesize_stop_transitions(rows)
        if stop_rows:
            rows = rows + stop_rows
            LOGGER.info(f"Added {len(stop_rows)} synthetic STOP transitions")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type != "dqn_sklearn":
        # Review fix: these modes were advertised but only ever fell back to
        # the sklearn implementation. Refuse loudly instead of silently
        # training a different algorithm than requested.
        raise SystemExit(
            f"--model-type {args.model_type!r} is not implemented. "
            f"Only 'dqn_sklearn' is available; implement torch/PPO before "
            f"advertising them."
        )

    model = train_sklearn_dqn(rows, pre_feature_cols, action_vocab, args)
    agent = SklearnDQNAgent(pre_feature_cols, action_vocab, model, action_encoding="one_hot")
    agent.save(output_dir / "rl_agent.joblib")
    LOGGER.info(f"Saved agent to {output_dir / 'rl_agent.joblib'}")

    # Save config
    config = {
        "model_type": args.model_type,
        "gamma": args.gamma,
        "feature_cols": pre_feature_cols,
        "action_vocab": action_vocab,
        "action_encoding": "one_hot",
        "q_iterations": args.q_iterations,
    }
    (output_dir / "rl_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (output_dir / "rl_metrics.json").write_text(
        json.dumps(
            {
                "transitions": len(rows),
                "actions": len(action_vocab),
                "action_encoding": "one_hot",
            },
            indent=2,
        )
    )

    print(f"[RL] Training complete. Output dir: {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
