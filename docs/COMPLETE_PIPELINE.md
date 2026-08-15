# Complete Pipeline — Implementation Notes

> **Legacy documentation (pre-Aug 2026).** This describes the original
> IR-oriented pipeline. The current runtime-based pipeline (multi-state
> dataset, loop-pass SL scorer, runtime-trained RL) is documented in
> `README.md` and `EXECUTION_RUNBOOK_RUNTIME.md`.

This document maps the July 31 2026 design doc to actual implemented files.

## Objective Restated

> Given an unseen C/C++ program, automatically generate an LLVM optimization pipeline that produces better code than -O1/-O2/-O3.

Your final system is **not** trying to predict one optimization pass.

It generates:
```
Program → GVN → LICM → InstCombine → DCE → Loop Unroll → Optimized Program
```

Two learning components:
- Supervised Learning → immediate pass quality (expected reward)
- Reinforcement Learning → pass ordering and long-term optimization

## Phase Mapping

| Design Phase | Implemented File | Status |
|--------------|------------------|--------|
| Phase 1 Benchmark Collection | `benchmarks/README.md`, CompilerGym datasets (cbench-v1, anghabench-v1, etc) | ✅ |
| Phase 2 Pass Selection | `scripts/curated_passes.py` (27 passes) | ✅ |
| Phase 3 SL Dataset Generation | `scripts/generate_dataset.py` (base), `scripts/generate_sl_dataset.py` (wrapper with curated set) → raw CSV | ✅ Proven by cbench_runtime log (460 samples) |
| Phase 4 Train Supervised | `training/train_sl.py`, `training/common.py` → predicts expected reward for each pass, softmax distribution | ✅ NEW |
| Phase 5 RL Dataset Generation | `scripts/collect_rl_transitions.py` → episodes S0→S1→S2, replay buffer, termination on zero reward / no IR change / repeated state / max passes | ✅ NEW |
| Phase 6 RL Training | `training/train_rl.py` → DQN with fitted Q iteration (sklearn fallback), supports torch PPO if available | ✅ NEW |
| Phase 7 Hybrid Inference | `training/inference.py` → SL predicts probs, RL considers high-prob candidates + exploration, loop 10-20 steps | ✅ NEW |
| Evaluation | `evaluation/evaluate_benchmarks.py`, `scripts/evaluate.py` → beats -O3 evaluation | ✅ NEW |

## Key Refinement Implemented

As requested in design:

> I would **not** train the supervised model to predict a single "best pass" directly.
> Instead train it to estimate expected immediate reward (or rank) for each candidate pass given current program state.

Implemented in `training/train_sl.py`:
- Model is Regressor: `[state_features + pass_id] → expected_reward`
- At inference, score all 27 passes → softmax → probability distribution
- RL uses this distribution as prior: `Q_hybrid = 0.7*Q + 0.3*SL_prob*10`

## Reward Design

Implemented in `scripts/reward.py`:

```
Reward = 0.6*Runtime Improvement + 0.3*IR Reduction + 0.1*Code Size Reduction
```

If runtime unavailable (many cBench non-runnable), renormalize to IR+Size:
- Effective weights: ir_ratio = ir/(ir+size), etc.

## Dataset Size Targets

Implemented and configurable via CLI:

**SL:**
```
cBench 23 * 27 = 621
LLVM Test Suite 500 * 30 = 15000
PolyBench 30 * 30 = 900
AnghaBench 5000 * 30 = 150000
Total ≈166k supervised samples
```

**RL:**
```
500 benchmarks × 200 episodes = 100k episodes
10 transitions avg → 1M transitions replay buffer
```

Use `--max-benchmarks` and `--episodes-per-benchmark` to control for fast iteration.

## Final Repository Structure (as requested)

```
NeuroCompiler/
├── benchmarks/
│   ├── cbench/
│   ├── polybench/
│   ├── llvm_test_suite/
│   └── anghabench/
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── supervised/
│   └── replay_buffer/
├── scripts/
│   ├── extract_features.py
│   ├── generate_sl_dataset.py
│   ├── collect_rl_transitions.py
│   ├── process_dataset.py
│   └── evaluate.py
├── models/
│   ├── supervised/
│   └── rl/
├── training/
│   ├── train_sl.py
│   ├── train_rl.py
│   └── inference.py
└── results/
```

All present in this repo, plus additional helpers:
- `scripts/curated_passes.py` (pass selection single source)
- `scripts/reward.py` (hybrid reward)
- `training/common.py` (shared feature utils)
- `evaluation/evaluate_benchmarks.py`
- `run_pipeline.sh` (end-to-end)

## How to Run End-to-End (Locally with Conda)

```bash
conda activate neurocompiler

# Phase 3 SL raw + processed
python scripts/generate_sl_dataset.py --dataset cbench-v1 --process --max-benchmarks 23

# Phase 4 SL train
python training/train_sl.py --input datasets/processed/hybrid_dataset.csv --model histgb

# Phase 5 RL buffer
python scripts/collect_rl_transitions.py --dataset cbench-v1 --max-benchmarks 23 --episodes-per-benchmark 20 --max-steps-per-episode 10

# Phase 6 RL train
python training/train_rl.py --input datasets/replay_buffer/rl_experiences.csv

# Phase 7 Hybrid inference
python training/inference.py --benchmark benchmark://cbench-v1/qsort --max-steps 10

# Evaluation vs held-out test split
python evaluation/evaluate_benchmarks.py
```

## Tests

Existing tests cover Stage 1-4 invariants:

```bash
pytest -q
```

Tests use FakeEnv, no CompilerGym needed.

## Next Steps for Publication

1. Generate full AnghaBench 5k × 27 = 150k SL samples
2. Generate 100k RL episodes (or 1M transitions)
3. Train LightGBM SL + PPO RL with torch (install `stable-baselines3`)
4. Evaluate win-rate vs -O3 on test split, report avg IR reduction %, runtime speedup
5. Add plots in `results/` (IR reduction distribution, pass frequency in learned pipelines)

All scaffolding for this is present.
