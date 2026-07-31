# NeuroCompiler — Hybrid SL + RL LLVM Optimizer

**Goal:** Given an unseen C/C++ program, automatically generate an LLVM optimization pipeline that produces better code than default -O1/-O2/-O3.

Instead of predicting *one* pass, the system generates a sequence:

```
Program
  ↓
GVN
  ↓
LICM
  ↓
InstCombine
  ↓
DCE
  ↓
Loop Unroll
  ↓
Optimized Program
```

Two learning components:
- **Supervised Learning → learns immediate pass quality (expected reward per pass)**
- **Reinforcement Learning → learns pass ordering and long-term cumulative gain**

## Complete Pipeline

```
Benchmark Programs (cBench, PolyBench, LLVM Test Suite, AnghaBench)
        │
        ▼
CompilerGym + LLVM
        │
        ▼
Generate SL Transition Dataset (scripts/generate_sl_dataset.py)
        │
        ▼
Train Supervised Pass Predictor (training/train_sl.py)
        │
        ▼
Generate RL Experiences (scripts/collect_rl_transitions.py)
        │
        ▼
Train RL Optimization Agent (training/train_rl.py)
        │
        ▼
Hybrid Optimization System (training/inference.py)
        │
        ▼
Optimize Any New Program (beats -O3)
```

## Repository Structure

```
NeuroCompiler/
├── benchmarks/
│   ├── cbench/
│   ├── polybench/
│   ├── llvm_test_suite/
│   └── anghabench/
├── datasets/
│   ├── raw/                  # SL raw: pass_runtime_dataset.csv
│   ├── processed/            # SL processed: hybrid_dataset.csv + splits + normalization
│   ├── supervised/           # train-ready splits (optional)
│   └── replay_buffer/        # RL experiences: rl_experiences.csv
├── scripts/
│   ├── extract_features.py          # Stage 1: 56 Autophase + IR stats + object size
│   ├── run_passes.py                # Stage 2: Transition recording
│   ├── generate_dataset.py          # Stage 3 base (generic)
│   ├── generate_sl_dataset.py       # Phase 3 wrapper with curated 27 passes
│   ├── curated_passes.py            # Phase 2 pass selection (25-30 passes)
│   ├── reward.py                    # Hybrid reward: 0.6*RT + 0.3*IR + 0.1*Size
│   ├── collect_rl_transitions.py    # Phase 5: RL episodes -> replay buffer
│   ├── process_dataset.py           # Stage 4: clean, benchmark-split, normalize
│   └── evaluate.py                  # Wrapper for evaluation
├── models/
│   ├── supervised/   # sl_reward_model.joblib, sl_action_vocab.json, etc
│   ├── reinforcement/ # rl_agent.joblib, rl_config.json
│   └── hybrid/
├── training/
│   ├── common.py      # Feature utils
│   ├── train_sl.py    # Phase 4: reward regression -> probability distribution
│   ├── train_rl.py    # Phase 6: DQN/PPO with fitted Q iteration
│   └── inference.py   # Phase 7: Hybrid SL-guided RL inference
├── evaluation/
│   └── evaluate_benchmarks.py  # Test split evaluation vs baselines
└── results/
```

## Phase Details

### Phase 2 — LLVM Pass Selection
Do NOT use all 100+ passes. Use 27 that mutate IR:

**Scalar:** ADCE, DCE, EarlyCSE, GVN, NewGVN, InstCombine, AggressiveInstCombine, SROA, Reassociate, SimplifyCFG, ConstMerge, CorrelatedPropagation

**Loop:** LICM, LoopRotate, LoopUnroll, LoopVectorize, LoopDeletion, LoopUnswitch, LoopDistribute, IndVars

**Interprocedural:** Inline, PartialInliner, DeadArgElim, ArgPromotion, GlobalOpt, GlobalDCE, FunctionAttrs

**Memory:** DSE, MemcpyOpt

**Misc:** JumpThreading, TailCallElim

See `scripts/curated_passes.py`

### Phase 3 — Supervised Dataset Generation

For every benchmark:

```
Load Benchmark → Extract Initial Features S0 → Apply ONE Pass → Extract New Features S1 → Measure Reward → Save Transition → Reset → Next Pass
```

Every row: `State_before, Optimization_pass, Reward, State_after`

Feature Vector (72 dims + 56 Autophase):
- Instruction Count, Basic Blocks, Functions, Loops, Branches, PHI Nodes, Loads, Stores, Arithmetic, Memory, Call instructions
- CFG Statistics, IR Graph Statistics, Runtime, Compile Time, Object Size, Reward, IR Hash

### Phase 4 — Train Supervised Model (Key Refinement)

**Not single label classification.**

Instead train to estimate **expected immediate reward / rank** for each candidate pass given current state.

Input: Program Features
Output: P(GVN), P(LICM), P(DCE), ... probability distribution (via softmax over predicted rewards)

This distribution becomes a policy prior for RL.

Models: RandomForest, LightGBM, XGBoost, CatBoost, MLP (auto fallback)

### Phase 5 — RL Dataset Generation

No fixed CSV. RL agent creates experience:

```
Program -> State S0 -> Choose Pass -> State S1 -> Choose Pass -> State S2 -> Terminal
```

- State: Current LLVM IR → extract_features.py → Feature Vector (same extractor)
- Action: One LLVM Pass
- Environment: CompilerGym applies it
- Reward: 0.6*Runtime Improvement + 0.3*IR Reduction + 0.1*Code Size Reduction

Store replay buffer: State, Action, Reward, Next State, Done

Episode termination:
- Reward zero
- No IR change
- Repeated state
- Max passes (10/15/20)

Target: 500 benchmarks × 200 episodes = 100k episodes ≈1M transitions

### Phase 6 — RL Training

Train PPO/DQN/A2C (DQN implemented with fitted Q iteration + sklearn fallback to avoid GPU requirement)

RL learns ordering automatically.

### Phase 7 — Hybrid Inference

1. New program → LLVM IR → Extract Features
2. SL predicts: GVN 0.34, LICM 0.29, InstCombine 0.18, DCE 0.10
3. RL considers mainly high-probability candidates while retaining exploration
4. RL chooses GVN → LLVM applies
5. Extract features S1
6. SL predicts new distribution (LICM 0.41 now top)
7. RL chooses LICM
8. Repeat S0→GVN→S1→LICM→S2→InstCombine→S3→DCE→Final

## Installation

```bash
conda env create -f environment.yml
# or
conda create -n neurocompiler python=3.10
conda activate neurocompiler
pip install -r requirements.txt  # compiler_gym, torch, sklearn, lightgbm, pandas, etc

# Compile CompilerGym service (first run will download)
python scripts/extract_features.py --benchmark benchmark://cbench-v1/qsort
```

## Quickstart

### 1. Fast smoke test (2 benchmarks × 5 passes)

```bash
conda activate neurocompiler
python scripts/generate_sl_dataset.py \
  --dataset cbench-v1 --max-benchmarks 2 --max-passes 5 \
  --skip-object-text-size --no-resume --process
```

### 2. Full cBench census with curated 27 passes

```bash
python scripts/generate_sl_dataset.py \
  --dataset cbench-v1 --reward-space IrInstructionCountO3 --process
# -> datasets/processed/hybrid_dataset.csv (~690 rows)
```

### 3. Train SL reward predictor

```bash
python training/train_sl.py \
  --input datasets/processed/hybrid_dataset.csv \
  --model histgb
# -> models/supervised/sl_reward_model.joblib
```

### 4. Collect RL experiences (20 episodes per benchmark)

```bash
python scripts/collect_rl_transitions.py \
  --dataset cbench-v1 --max-benchmarks 10 --episodes-per-benchmark 20 \
  --max-steps-per-episode 10
# -> datasets/replay_buffer/rl_experiences.csv
```

### 5. Train RL agent

```bash
python training/train_rl.py --input datasets/replay_buffer/rl_experiences.csv
# -> models/reinforcement/rl_agent.joblib
```

### 6. Hybrid inference on new program

```bash
python training/inference.py --benchmark benchmark://cbench-v1/qsort --max-steps 10
```

Output example:
```
[Hybrid] Optimizing benchmark://cbench-v1/qsort
  Initial IR: 1894 -> Final: 1620 (reduction 14.4%)
  Sequence: -gvn -> -licm -> -instcombine -> -dce -> -sroa
```

### 7. Evaluate on test split (unseen programs)

```bash
python evaluation/evaluate_benchmarks.py --max-benchmarks 10 --max-steps 10
```

## Why Hybrid is Stronger

Standard "ML chooses an LLVM pass" predicts one pass → limited gain.

This project:
- SL provides **strong local heuristics** (which pass looks good now)
- RL discovers **effective sequences and ordering** for long-term cumulative reward
- Hybrid beats -O3 because it adapts to program features instead of using fixed pipeline

Easier to justify in research: supervised reward modeling + RL for sequential decision = principled division of labor.

## References

- CompilerGym (Facebook Research) - https://github.com/facebookresearch/CompilerGym
- Autophase - 56 static IR features
- cBench, AnghaBench, PolyBench

## License

MIT for training code. Benchmarks retain their original licenses.
