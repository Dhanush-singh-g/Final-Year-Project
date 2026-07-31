#!/usr/bin/env python3
"""
Phase 4 — Train Supervised Pass Predictor (Reward Regression)

Key refinement from the design doc:
  Do NOT train to predict a single "best pass" (classification).
  Instead train to estimate expected immediate reward (or rank) for each candidate pass
  given the current program state. During inference this produces a probability distribution.

Input: datasets/processed/hybrid_dataset.csv (from process_dataset.py)
  Each row: pre_state (56 Autophase + core stats) + action (pass_id/flag) + reward + post_state

Model architecture options:
  - Input: Program Features (pre_ + norm_pre_)
  - Output: P(GVN), P(LICM), P(DCE) ... = probability / expected reward per pass

Two implementation modes:
  1) Reward regression with action as feature: model([state_features + pass_id]) -> expected_reward
     At inference: score all 27 passes, softmax to distribution, ranked list.
  2) Multi-output: state -> vector of 27 rewards (requires grouping)

We implement mode 1 because it matches the dataset format (one pass per row) and is robust.

Models supported (fallback chain):
  - LightGBM (LGBMRegressor)
  - XGBoost
  - CatBoost
  - Sklearn HistGradientBoosting / RandomForest
  - Sklearn MLPRegressor

Saves:
  - models/supervised/sl_reward_model.pkl  (or .joblib)
  - models/supervised/sl_action_vocab.json
  - models/supervised/sl_feature_columns.json
  - models/supervised/sl_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.common import (  # noqa: E402
    get_feature_cols,
    safe_float,
    load_csv_rows,
    split_by_column,
)

LOGGER = logging.getLogger("train_sl")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "supervised"

def extract_xy(rows: List[Dict[str, str]], feature_cols: List[str], action_vocab: Dict[str, int]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Returns X_state, y_reward, X_meta for training.
    X = [state_features + one-hot or id encoded? we include action_id as numeric + vocab index]
    """
    X = []
    y = []
    meta = []  # for debugging: raw features
    for r in rows:
        feats = []
        skip = False
        for col in feature_cols:
            v = safe_float(r.get(col, ""))
            if v is None:
                v = 0.0
            feats.append(v)

        # Action encoding: use pass_id if numeric, else vocab index
        pass_id_raw = r.get("pass_id", "")
        pass_flag = r.get("pass_flag", "") or r.get("pass_name", "")
        try:
            pid = float(pass_id_raw)
        except:
            pid = float(action_vocab.get(pass_flag, 0))

        # Add action encoding as extra feature (simple)
        feats.append(pid)

        # Determine target reward: prefer step_reward, fallback to delta ir
        target = safe_float(r.get("step_reward"))
        if target is None:
            # Compute from delta_ir: positive reduction = positive reward
            delta = safe_float(r.get("delta_ir_instruction_count"))
            if delta is not None:
                target = -delta  # if delta -5 (reduced 5 instr), reward +5
            else:
                target = 0.0

        X.append(feats)
        y.append(target)
        meta.append([r.get("benchmark_uri",""), pass_flag, target])

    return X, y, meta

def get_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
        except ImportError:
            LOGGER.warning("LightGBM not available, falling back")
            model_type = "histgb"

    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
        except ImportError:
            LOGGER.warning("XGBoost not available, falling back")
            model_type = "histgb"

    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=500, depth=8, random_seed=42, verbose=False)
        except ImportError:
            LOGGER.warning("CatBoost not available, falling back")
            model_type = "histgb"

    if model_type in ("randomforest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

    if model_type in ("histgb", "gb", "hgb"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_iter=500, max_depth=10, learning_rate=0.05, random_state=42)
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    if model_type in ("mlp", "nn"):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(256,128,64), activation="relu", max_iter=300, random_state=42)

    # default
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

def evaluate_model(model, X_val, y_val) -> Dict[str, float]:
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        return {"mse": mse, "mae": mae, "r2": r2}
    except Exception as e:
        return {"error": str(e)}

def train(args: argparse.Namespace):
    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.output_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading {input_path}")
    rows, fieldnames = load_csv_rows(input_path, max_rows=args.max_rows)
    LOGGER.info(f"Loaded {len(rows)} rows, {len(fieldnames)} cols")

    splits = split_by_column(rows, split_col="dataset_split")
    LOGGER.info(f"Splits: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")

    # Use train split to build vocab and feature list
    feature_cols = get_feature_cols(fieldnames, use_norm=not args.no_norm)
    LOGGER.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    # Action vocab from all rows to ensure coverage
    action_vocab = {}
    for r in rows:
        key = r.get("pass_flag") or r.get("pass_name")
        if key and key not in action_vocab:
            action_vocab[key] = len(action_vocab)
    LOGGER.info(f"Actions vocab size: {len(action_vocab)} -> {list(action_vocab.keys())[:5]}")

    X_train, y_train, _ = extract_xy(splits["train"] or rows, feature_cols, action_vocab)
    X_val, y_val, _ = extract_xy(splits["validation"] or splits["train"][:100], feature_cols, action_vocab)
    X_test, y_test, _ = extract_xy(splits["test"] or splits["train"][:100], feature_cols, action_vocab)

    LOGGER.info(f"Training model type={args.model}")
    model = get_model(args.model)
    model.fit(X_train, y_train)

    metrics = {}
    if X_val:
        metrics["validation"] = evaluate_model(model, X_val, y_val)
        LOGGER.info(f"Val metrics: {metrics['validation']}")
    if X_test:
        metrics["test"] = evaluate_model(model, X_test, y_test)
        LOGGER.info(f"Test metrics: {metrics['test']}")

    # Also compute top-k accuracy proxy: among same benchmark pre-state group, does model rank correct pass high?
    # For simplicity, compute Pearson correlation

    # Save artifacts
    import joblib
    # Try joblib, fallback pickle
    try:
        joblib.dump(model, model_dir / "sl_reward_model.joblib")
        LOGGER.info(f"Saved joblib model to {model_dir / 'sl_reward_model.joblib'}")
    except Exception:
        with (model_dir / "sl_reward_model.pkl").open("wb") as f:
            pickle.dump(model, f)
        LOGGER.info(f"Saved pkl model to {model_dir / 'sl_reward_model.pkl'}")

    (model_dir / "sl_action_vocab.json").write_text(json.dumps(action_vocab, indent=2, sort_keys=True))
    (model_dir / "sl_feature_columns.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "uses_norm": not args.no_norm,
        "action_encoding": "pass_id_appended",
    }, indent=2))

    (model_dir / "sl_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    LOGGER.info(f"Saved vocab, feature cols, metrics to {model_dir}")

    # Also save a simple inference helper: given state, predict distribution over all passes
    # Precompute pass list for inference
    pass_list = list(action_vocab.keys())
    (model_dir / "sl_pass_list.json").write_text(json.dumps(pass_list, indent=2))

    # Example inference demo file
    example_code = '''
# Example usage for inference:
import json, joblib
import numpy as np

model = joblib.load("sl_reward_model.joblib")
vocab = json.loads(open("sl_action_vocab.json").read())
feature_cols = json.loads(open("sl_feature_columns.json").read())["feature_cols"]

def predict_pass_distribution(state_features_dict, pass_flags):
    # state_features_dict: dict of pre_* features
    # returns list of (flag, expected_reward)
    scores = []
    for flag in pass_flags:
        feats = [state_features_dict.get(c,0.0) for c in feature_cols]
        pid = vocab.get(flag, 0)
        feats.append(float(pid))
        score = model.predict([feats])[0]
        scores.append((flag, float(score)))
    # Softmax to probabilities
    vals = np.array([s for _,s in scores])
    # shift for numerical stability
    exps = np.exp(vals - np.max(vals))
    probs = exps / exps.sum()
    return sorted([(flag, float(score), float(prob)) for (flag,score),prob in zip(scores,probs)], key=lambda x: x[1], reverse=True)
'''
    (model_dir / "example_inference.py").write_text(example_code)

    print(f"[SL] Training complete. Model dir: {model_dir}")
    print(f"[SL] Metrics: {metrics}")

    return model_dir

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 - Train Supervised Reward Predictor")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="processed hybrid_dataset.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--model", default="histgb", choices=["lightgbm","xgboost","catboost","randomforest","rf","histgb","mlp"], help="Model type")
    p.add_argument("--max-rows", type=int, default=None, help="Limit rows for fast debug")
    p.add_argument("--no-norm", action="store_true", help="Don't use normalized features even if present")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    train(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
