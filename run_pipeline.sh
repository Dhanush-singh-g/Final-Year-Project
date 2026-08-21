#!/usr/bin/env bash
# NeuroCompiler end-to-end pipeline runner.
#
# Runs every phase in order. Safe by default: dataset generation resumes
# (completed transitions are skipped), so rerunning never destroys data.
# Set FRESH=1 to force a full re-census from scratch.
#
#   bash run_pipeline.sh
#   SL_MAX_BENCHMARKS=23 RL_EPISODES=200 bash run_pipeline.sh
#
set -euo pipefail
cd "$(dirname "$0")"

: "${DATASET:=cbench-v1}"
: "${SL_MAX_BENCHMARKS:=23}"
: "${SL_MAX_PASSES:=}"
: "${RL_MAX_BENCHMARKS:=23}"
: "${RL_EPISODES:=20}"
: "${RL_STEPS:=10}"
: "${MAX_STEPS:=10}"
: "${FRESH:=0}"
: "${SL_TARGET:=runtime_improvement_pct}"

RESUME_FLAG=""
if [ "$FRESH" = "1" ]; then
  RESUME_FLAG="--no-resume"
fi

SL_EXTRA=""
if [ -n "$SL_MAX_PASSES" ]; then
  SL_EXTRA="--max-passes $SL_MAX_PASSES"
fi

echo "==> Phase 3: SL transition dataset (curated passes)"
# shellcheck disable=SC2086
python scripts/generate_sl_dataset.py \
  --dataset "$DATASET" \
  --max-benchmarks "$SL_MAX_BENCHMARKS" \
  $SL_EXTRA \
  $RESUME_FLAG \
  --process

echo "==> Phase 4: Train supervised pass-quality predictor"
python training/train_sl.py \
  --input datasets/processed/hybrid_dataset.csv \
  --target "$SL_TARGET"

echo "==> Phase 5: Collect RL experiences (random episodes)"
# shellcheck disable=SC2086
python scripts/collect_rl_transitions.py \
  --dataset "$DATASET" \
  --max-benchmarks "$RL_MAX_BENCHMARKS" \
  --episodes-per-benchmark "$RL_EPISODES" \
  --max-steps-per-episode "$RL_STEPS" \
  $RESUME_FLAG

echo "==> Phase 6: Train RL agent (fitted-Q DQN, CPU-only)"
python training/train_rl.py \
  --input datasets/replay_buffer/rl_experiences.csv \
  --model-type dqn_sklearn \
  --gamma 0.90 \
  --q-iterations 3

echo "==> Phase 7: Hybrid SL + RL inference on one program"
python training/inference.py \
  --benchmark benchmark://cbench-v1/qsort \
  --max-steps "$MAX_STEPS" \
  --output results/hybrid_inference.json

echo "==> Evaluation on the held-out test split"
python evaluation/evaluate_benchmarks.py \
  --processed-csv datasets/processed/hybrid_dataset.csv \
  --max-steps "$MAX_STEPS"

echo "Pipeline complete. See models/ and results/."
