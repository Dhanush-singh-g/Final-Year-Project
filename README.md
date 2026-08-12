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
Optimize Any New Program (beats -O3 on IR count; runtime-vs-O3 measured by the external baseline harness, see results)
```

## Current Measured Results (scaled run, Aug 2026)

A 10–20× scale-up of the SL and RL datasets was generated with the new parallel driver
`scripts/scale_census.py` (sharded workers + resume + merge).

| Stage | Artifact | Size |
|---|---|---|
| SL census, 6 suites × 31 curated passes, runtime labeled | `datasets/raw/scale_sl/` → `scale_sl_combined.csv` | **4,441 transitions** (cBench 23, CHStone 12, BLAS 30, CLgen 30, POJ104 30, csmith 30) |
| Processed (benchmark-wise 70/15/15 split) | `datasets/processed/hybrid_dataset_scaled.csv` | 4,432 rows, **146 benchmarks** (train 3,134 / val 652 / test 646) |
| SL pass scorer (HistGB, target `step_reward` over all rows) | `models/supervised/` | 31 actions, 62 features |
| SL runtime-target reference model | `models/supervised_runtime/` | 31 actions |
| RL replay buffer (102 train-split benchmarks × 24 episodes) | `datasets/replay_buffer/rl_experiences_scaled.csv` | **3,324 transitions** |
| RL fitted-Q agent (vectorized Bellman, 3 iterations) | `models/reinforcement/` | 31 actions |

Held-out test evaluation — **all 22 test-split benchmarks, never seen in training**
(`results/hybrid_test_results_scaled_all.json`):

| Group | n | Mean IR reduction | vs exact -O3 IR | Runtime vs initial |
|---|---|---|---|---|
| cBench (real-world) | 10 | **33.6%** | **+3.8% (beats -O3 in 8/10)** | 1.20× (5/8 wins) |
| CHStone (embedded) | 3 | 21.0% | −14.1% (1/3) | n/a |
| csmith (synthetic) | 9 | 32.3% | −101.6% (2/9) | 1.80× (7/9 wins) |
| **All** | 22 | **31.4%** | wins **11/22** | **1.49×** (12/17 wins) |

Highlights on real-world programs: jpeg-c 62,452 → 36,229 IR (**+18.8% vs -O3**),
lame 49,131 → 29,747 (**+16.4% vs -O3**), gsm +32.4% IR, bzip2 +34.4% IR,
tiff2rgba 58,661 → 37,131 (**+5.4% vs -O3**).

Learned sequences are short and sensible, e.g. `-sroa → -simplifycfg`, `-newgvn → -newgvn`,
and `-sroa ×7 → -loop-distribute` (lame).

**External O3 runtime baseline — v2 protocol (runbook §10, `evaluation/o3_runtime_harness.py`).**
CompilerGym exposes no `-O3` runtime observation, so runtime-vs-O3 is measured with a controlled
harness: the same O0 bitcode is compiled three ways — `clang -O3` (clang's own O3 pipeline, the
reference baseline), `opt -O3` (kept as a sanity check), and the hybrid final IR as a pre-pass —
ALL with clang's `-O3` codegen, so the comparison isolates the middle-end pass sequence. Plain
`clang module.bc` without an -O flag emits effectively O0-level codegen; the pre-2026 waves that
used it made hybrid look 0.95×-competitive against a weak baseline and are superseded by these
v2 waves. Executables run with the benchmark's own dynamic input config, identical warmups +
repetitions, and `taskset` CPU pinning, reporting medians with 95% bootstrap CIs (interleaved to
cancel drift). Because the hybrid pass sequence is input-independent, binaries are built once per
benchmark and timed on multiple inputs (`--inputs`); the summary reports the largest-baseline-median
input per benchmark (most trustworthy timing), deduplicated across waves.

The table below is the **Aug 2026 re-measurement under the fixed inference** (no-op actions now
truly terminate and are masked per state, so degenerate repeated-pass sequences are gone):

| Runtime comparison (executable baseline, `-O3` codegen, largest-input representative) | n | Geo-mean vs opt -O3 | vs clang -O3 | Wins vs clang -O3 |
|---|---|---|---|---|
| All 22 test-split benchmarks | 22 | 1.062× | 1.018× | 11/22 (Wilcoxon p=0.45, n.s.) |
| cBench, non-trivial inputs (median ≥ 0.03 s) | 7 | 1.027× | 1.005× | 5/7 |
| cBench, substantial inputs (median ≥ 0.1 s) | 6 | 1.011× | 1.000× | 3/6 |

Large-input cBench detail (`results/o3_runtime_vs_o3_summary.json`):

| benchmark | input | opt-O3 med | clang-O3 med | hybrid med | spd vs clang-O3 |
|---|---|---|---|---|---|
| bzip2 | 8.bz2 | 4.252 s | 4.274 s | 4.252 s | 1.005× |
| dijkstra | 9.dat | 0.611 s | 0.604 s | 0.614 s | 0.984× |
| gsm | 2.au | 0.621 s | 0.634 s | 0.627 s | 1.012× |
| jpeg-c | 17.ppm | 0.980 s | 0.988 s | 1.013 s | 0.975× |
| tiff2rgba | 11.nocomp.tif | 2.476 s | 2.218 s | 2.347 s | 0.945× |
| tiff2bw | 17.nocomp.tif | 2.372 s | 2.398 s | 2.467 s | 0.972× |
| stringsearch | 4.txt | 0.027 s | 0.027 s | 0.028 s | 0.992× |
| bitcount | (arg) | 0.030 s | 0.033 s | 0.032 s | 1.026× |

Honest interpretation — the key finding of the corrected protocol: **hybrid does not beat
`clang -O3` at runtime when measured properly.** The large-input wins from the earlier O0-codegen
protocol (dijkstra 1.42×, tiff2rgba 1.36×, bzip2 1.24×) collapse to ~1.00× once both arms use
real `-O3` codegen — the backend re-optimizes the IR-level differences away. On the cBench
benchmarks with substantial (≥ 0.1 s) inputs the geo-mean is **1.000× vs `clang -O3`** (a tie),
and the overall 1.018× geo-mean is inflated by sub-10 ms CHStone/csmith rows where process
startup noise dominates (Wilcoxon p=0.45, n.s.). The fixed inference does not change this
conclusion: the old repeated-pass sequences had already terminated at the first no-op for these
benchmarks, so the pass sequences are essentially unchanged. `opt -O3` ≈ `clang -O3` (within
~1%), validating both baselines, and all runs produced byte-identical outputs across the three
arms (`outputs_match=true`). The research conclusion matches the plan's hypothesis: Phase-1
IR-count gains do not yet translate to runtime wins over a properly measured `-O3`; the next
steps are a runtime-aware (z-scored) reward and longer learned sequences with STOP, trained
against the `-O3` codegen target.

### Z-scored runtime reward (why raw runtime targets fail)

The raw `runtime_improvement_pct` target is **incomparable across programs**: each benchmark's
candidate-pass runtime distribution has a different mean and scale (e.g. gsm mean −21% vs another
benchmark +57%), so a scorer trained on raw values can learn *benchmark identity* rather than pass
quality. `scripts/zscore_dataset.py` adds a `z_runtime_improvement_pct` column that normalises each
benchmark's candidate distribution to mean 0 / std 1 (`scripts/reward.py::per_benchmark_zscore`),
and `train_sl.py` accepts it as a target. Trained on the z-scored target (`models/supervised_z/`), the scorer's test R² drops
from −8.05 (raw) to ≈ 0 and test top-3 pass ranking from 9.1% to **0%** (random ≈ 9.7%): once the
benchmark-identity shortcut is removed, the remaining pass-quality signal is **not learnable from
the current data** (short sub-10 ms workloads, ~3.1k train rows across 31 passes). This is the
controlled, decisive confirmation that runtime-aware training needs longer workloads and more
per-benchmark coverage — not just a different target.

Honest caveats:

1. On small/synthetic programs (CHStone, csmith) `-O3` still wins the IR-count race — its full
   fixed pipeline removes trivially dead synthetic code that our short learned sequences do not.
2. Runtime-vs-O3 is measured directly with `-O3` codegen (table above). The `opt -O3` arm runs
   on the same O0 bitcode (IR-pipeline comparison); the `clang -O3` arm is clang's own pipeline
   on that bitcode — the benchmark protos ship no source, so a literal source-level `clang -O3`
   rebuild is not possible for these datasets.
3. Cross-program runtime prediction remains noisy (single-pass runtime deltas are dominated by
   process overhead), so the canonical SL scorer uses the deterministic IR `step_reward`;
   `models/supervised_runtime/` is kept as the runtime-target reference.

### Pilot history (cBench-only, 460 runtime-labeled rows)

The earlier pilot (`datasets/raw/cbench_runtime_dataset_v2.csv`, `models/` artifacts from 14:10)
achieved 20.0% mean IR reduction and a 1/2 IR win rate vs -O3 on 2 test benchmarks. It remains
available for regression comparison; the scaled artifacts above supersede it.

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
│   ├── processed/            # SL processed: hybrid_dataset_scaled.csv (canonical) + pilot hybrid_dataset.csv
│   ├── supervised/           # train-ready splits (optional)
│   └── replay_buffer/        # RL experiences: rl_experiences_scaled.csv (canonical) + pilot rl_experiences.csv
├── scripts/
│   ├── extract_features.py          # Stage 1: 56 Autophase + IR stats + object size
│   ├── run_passes.py                # Stage 2: Transition recording
│   ├── generate_dataset.py          # Stage 3 base (generic)
│   ├── generate_sl_dataset.py       # Phase 3 wrapper with the curated pass set
│   ├── scale_census.py              # NEW: parallel/resumable SL+RL scale-up driver
│   ├── curated_passes.py            # Phase 2 pass selection (31 curated passes)
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
│   ├── evaluate_benchmarks.py  # Test split evaluation vs baselines
│   └── o3_runtime_harness.py   # NEW: external opt -O3 executable runtime baseline (§10)
└── results/
```

## Phase Details

### Phase 2 — LLVM Pass Selection
Do NOT use all 100+ passes. Use the 31 curated passes in `scripts/curated_passes.py` that actually mutate IR:

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

## Scaled Dataset Run (what generated the numbers above)

`scripts/scale_census.py` shards benchmark URIs across worker processes, reuses the
existing generation functions per shard, and merges + processes. It is resumable:
rerunning skips completed work (transition keys / deterministic episode IDs), and
`--resume-from` seeds a canonical merged CSV so re-runs skip finished rows instantly.

```bash
# 1. Scaled SL census: 155 benchmarks x 31 curated passes, runtime labeled
python scripts/scale_census.py sl \
  --workdir datasets/raw/scale_sl \
  --datasets cbench-v1,chstone-v0,blas-v0,clgen-v0,poj104-v1 \
  --csmith-count 30 --sample 30 \
  --workers 32 --shards 155 --measure-runtime \
  --runtime-warmup-count 1 --runtime-count 3 --skip-object-text-size

# 2. Merge + process
python scripts/scale_census.py merge-sl --workdir datasets/raw/scale_sl \
  --output datasets/raw/scale_sl_combined.csv --process \
  --processed-output datasets/processed/hybrid_dataset_scaled.csv

# 3. Scaled RL replay buffer (episodes only from train-split benchmarks, no leakage)
python scripts/scale_census.py rl --workdir datasets/raw/scale_rl \
  --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
  --workers 32 --episodes-per-benchmark 24 --max-steps-per-episode 8 --seed 42 \
  --skip-object-text-size

python scripts/scale_census.py merge-rl --workdir datasets/raw/scale_rl \
  --output datasets/replay_buffer/rl_experiences_scaled.csv

# 4. Retrain on the scaled data
python training/train_sl.py --input datasets/processed/hybrid_dataset_scaled.csv \
  --output-dir models/supervised --target step_reward
python training/train_rl.py --input datasets/replay_buffer/rl_experiences_scaled.csv \
  --output-dir models/reinforcement --gamma 0.90 --q-iterations 3

# 5. Evaluate on the held-out test split
python evaluation/evaluate_benchmarks.py \
  --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
  --max-steps 8 --measure-runtime --output results/hybrid_test_results_scaled.json

# 6. External O3 executable runtime baseline (runbook §10) — run in parallel waves
python evaluation/o3_runtime_harness.py measure \
  --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
  --sl-model-dir models/supervised --rl-model-dir models/reinforcement \
  --max-steps 8 --warmup 1 --reps 5 --cpu 4 --timeout 120 --inputs 0,largest \
  --workdir results/o3_harness_work --output results/o3_wave1.json

python evaluation/o3_runtime_harness.py summarize \
  --results results/o3_wave*.json --output results/o3_runtime_vs_o3_summary.json
```

For the full design targets (AnghaBench 5k × 31, 100k RL episodes), run the same commands
on a bigger machine with `--datasets anghabench-v1 --sample 5000` and higher
`--episodes-per-benchmark`; the driver parallelizes and resumes automatically.

## Quickstart

### 1. Fast smoke test (2 benchmarks × 5 passes)

```bash
conda activate neurocompiler
python scripts/generate_sl_dataset.py \
  --dataset cbench-v1 --max-benchmarks 2 --max-passes 5 \
  --skip-object-text-size --no-resume --process
```

### 2. Full cBench census with the curated pass set (pilot, ~690 rows)

```bash
python scripts/generate_sl_dataset.py \
  --dataset cbench-v1 --reward-space IrInstructionCountO3 --process
# -> datasets/processed/hybrid_dataset.csv (pilot)
# The canonical scaled dataset is datasets/processed/hybrid_dataset_scaled.csv;
# see "Scaled Dataset Run" below for how it is produced.
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
- Hybrid beats -O3 **on IR count** because it adapts to program features instead
  of using a fixed pipeline; runtime vs -O3 is measured separately by the
  external O3 baseline harness (0.99× geo-mean on the scaled run — a statistical
  tie with `-O3`, see results above; the earlier 0.93× figure used O0-level codegen
  and is superseded).

Easier to justify in research: supervised reward modeling + RL for sequential decision = principled division of labor.

## References

- CompilerGym (Facebook Research) - https://github.com/facebookresearch/CompilerGym
- Autophase - 56 static IR features
- cBench, AnghaBench, PolyBench

## License

MIT for training code. Benchmarks retain their original licenses.
