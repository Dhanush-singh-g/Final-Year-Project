# NeuroCompiler Runtime-First End-to-End Execution Runbook

## What this correction fixes

1. Supervised learning now defaults to `runtime_improvement_pct`, not
   `IrInstructionCountO3` step reward.
2. LLVM actions are one-hot encoded consistently in training and inference.
3. Tree-model training uses raw pre-state features by default, eliminating the
   previous online normalized-feature/all-zero mismatch.
4. Training writes elapsed time, completed boosting iterations, regression
   metrics, and per-state top-1/top-3 pass-ranking metrics.
5. Inference refuses to silently use a random policy when no SL model exists.
6. Hybrid inference reports runtime before/after and exact IR comparison against
   CompilerGym's `IrInstructionCountO3` baseline.
7. RL episode IDs are deterministic, making replay-buffer resume effective.
8. Evaluation uses runtime targets rather than instruction-count reward.

## Scientific boundary

CompilerGym exposes the exact `-O3` IR cost but does not directly expose an
`-O3` runtime observation. The corrected evaluation reports:

- hybrid runtime vs initial no-pass runtime;
- hybrid IR instruction count vs exact `-O3` IR instruction count.

Do not claim runtime superiority over `-O3` until a separate executable O3
baseline harness is implemented and measured using identical inputs, warmups,
CPU affinity, and repetitions.

## 0. Install the correction

Extract the correction archive from the WSL home directory. Its top-level
`NeuroCompiler/` paths merge into the existing project:

```bash
cd ~
unzip -o NeuroCompiler_runtime_training_fix.zip
cd ~/NeuroCompiler
```

Or copy the corrected Python files to their matching paths.

## 1. Activate and verify dependencies

```bash
cd ~/NeuroCompiler
conda activate neurocompiler
python --version
python - <<'PY'
import compiler_gym, numpy, sklearn, joblib
print('CompilerGym:', compiler_gym.__version__)
print('NumPy:', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('joblib:', joblib.__version__)
PY
```

Preserve NumPy 1.26.4. If training dependencies are missing:

```bash
python -m pip install 'numpy==1.26.4' 'scikit-learn==1.3.2' 'joblib==1.3.2'
```

## 2. Generate the raw runtime census

Use the previously approved 20-pass set. This command creates 23 x 20 = 460
attempted independent transitions.

```bash
python ./scripts/generate_dataset.py \
  --dataset cbench-v1 \
  --passes=-adce,-aggressive-instcombine,-argpromotion,-constmerge,-correlated-propagation,-dce,-deadargelim,-dse,-early-cse,-globaldce,-globalopt,-gvn,-indvars,-inline,-instcombine,-jump-threading,-licm,-loop-unroll,-loop-vectorize,-sroa \
  --measure-runtime \
  --require-runtime \
  --runtime-warmup-count 3 \
  --runtime-count 10 \
  --skip-object-text-size \
  --reward-space IrInstructionCountO3 \
  --timeout 600 \
  --output datasets/raw/cbench_runtime_dataset_v2.csv \
  --no-resume \
  --fsync
```

Do not delete files while the command is running. If interrupted, rerun without
`--no-resume`.

Verify:

```bash
test -s datasets/raw/cbench_runtime_dataset_v2.csv
wc -l datasets/raw/cbench_runtime_dataset_v2.csv
```

Expected maximum: 461 lines (header + 460 rows).

## 3. Process the runtime dataset

```bash
python ./scripts/process_dataset.py \
  --input datasets/raw/cbench_runtime_dataset_v2.csv \
  --output datasets/processed/cbench_runtime_hybrid_dataset.csv \
  --require-runtime
```

Verify:

```bash
python - <<'PY'
import csv, json
from pathlib import Path
p = Path('datasets/processed/cbench_runtime_hybrid_dataset.csv')
with p.open() as f:
    rows = list(csv.DictReader(f))
print('accepted rows:', len(rows))
for split in ('train','validation','test'):
    print(split, sum(r['dataset_split'] == split for r in rows))
for col in ('pre_runtime_median_sec','post_runtime_median_sec',
            'runtime_improvement_pct','runtime_speedup'):
    assert col in rows[0], col
print('runtime schema OK')
PY
```

## 4. Train the runtime supervised model

The default HistGradientBoosting model has up to 400 boosting iterations with
early stopping. These are not neural-network epochs.

```bash
/usr/bin/time -v python ./training/train_sl.py \
  --input datasets/processed/cbench_runtime_hybrid_dataset.csv \
  --output-dir models/supervised \
  --model histgb \
  --target runtime_improvement_pct \
  --seed 42
```

Do not add `--use-normalized` for the tree model. Expected artifacts:

```text
models/supervised/sl_reward_model.joblib
models/supervised/sl_action_vocab.json
models/supervised/sl_feature_columns.json
models/supervised/sl_pass_list.json
models/supervised/sl_metrics.json
```

Inspect metrics:

```bash
python -m json.tool models/supervised/sl_metrics.json
```

Important metrics:

- test MAE/RMSE/R2;
- test top-1 and top-3 pass-ranking accuracy;
- mean oracle regret;
- fit_seconds and iterations_completed.

With only 23 programs, treat these results as a pilot, not publication-grade
proof of generalization.

## 5. Run SL-only sequential inference

Use a benchmark assigned to the test split:

```bash
TEST_BENCHMARK=$(python - <<'PY'
import csv
with open('datasets/processed/cbench_runtime_hybrid_dataset.csv') as f:
    rows = list(csv.DictReader(f))
print(next(r['benchmark_uri'] for r in rows if r['dataset_split']=='test'))
PY
)

echo "$TEST_BENCHMARK"
python ./training/inference.py \
  --benchmark "$TEST_BENCHMARK" \
  --max-steps 10 \
  --measure-runtime \
  --output results/sl_only_test_result.json
```

Before an RL agent exists, inference uses the trained SL pass scorer. It no
longer silently substitutes a random model if SL artifacts are absent.

## 6. Collect RL transitions without test leakage

Generate episodes only from benchmarks assigned to the training split:

```bash
mapfile -t TRAIN_BENCHMARKS < <(python - <<'PY'
import csv
with open('datasets/processed/cbench_runtime_hybrid_dataset.csv') as f:
    rows = list(csv.DictReader(f))
print('\n'.join(sorted({r['benchmark_uri'] for r in rows if r['dataset_split']=='train'})))
PY
)

BENCHMARK_ARGS=()
for benchmark in "${TRAIN_BENCHMARKS[@]}"; do
  BENCHMARK_ARGS+=(--benchmark "$benchmark")
done

python ./scripts/collect_rl_transitions.py \
  --dataset cbench-v1 \
  "${BENCHMARK_ARGS[@]}" \
  --passes=-adce,-aggressive-instcombine,-argpromotion,-constmerge,-correlated-propagation,-dce,-deadargelim,-dse,-early-cse,-globaldce,-globalopt,-gvn,-indvars,-inline,-instcombine,-jump-threading,-licm,-loop-unroll,-loop-vectorize,-sroa \
  --episodes-per-benchmark 5 \
  --max-steps-per-episode 8 \
  --seed 42 \
  --measure-runtime \
  --runtime-warmup-count 1 \
  --runtime-count 5 \
  --skip-object-text-size \
  --output datasets/replay_buffer/rl_experiences.csv \
  --no-resume
```

This is a pilot collection. Increase episodes only after verifying the buffer.
If interrupted, rerun without `--no-resume`; deterministic episode IDs now make
resume effective.

Verify:

```bash
wc -l datasets/replay_buffer/rl_experiences.csv
```

## 7. Train the implemented RL agent

The current implemented algorithm is fitted-Q regression using sklearn. It is
not PPO or a neural DQN. PPO and Torch branches in the original repository were
placeholders.

```bash
/usr/bin/time -v python ./training/train_rl.py \
  --input datasets/replay_buffer/rl_experiences.csv \
  --output-dir models/reinforcement \
  --model-type dqn_sklearn \
  --gamma 0.90 \
  --q-iterations 3
```

Expected:

```text
models/reinforcement/rl_agent.joblib
models/reinforcement/rl_config.json
models/reinforcement/rl_metrics.json
```

## 8. Run hybrid SL + fitted-Q inference

```bash
python ./training/inference.py \
  --benchmark "$TEST_BENCHMARK" \
  --max-steps 10 \
  --measure-runtime \
  --output results/hybrid_test_result.json
```

Expected output includes:

- selected ordered pass sequence;
- initial and final runtime;
- runtime speedup and improvement percentage;
- initial and final IR count;
- exact hybrid-vs-O3 IR percentage;
- cumulative hybrid reward.

## 9. Evaluate on the held-out benchmark split

```bash
python ./evaluation/evaluate_benchmarks.py \
  --processed-csv datasets/processed/cbench_runtime_hybrid_dataset.csv \
  --target runtime_improvement_pct \
  --max-benchmarks 10 \
  --max-steps 10 \
  --measure-runtime \
  --output results/hybrid_test_results.json
```

Report:

1. SL test MAE/RMSE/R2.
2. Pass-ranking top-1 and top-3 accuracy.
3. Runtime geometric-mean speedup vs initial state.
4. Runtime win rate vs initial state.
5. Mean hybrid-vs-O3 IR improvement and IR win rate.
6. Dataset size, benchmark split, runtime repetitions, CPU, LLVM, and
   CompilerGym versions.

## 10. Required next research change

To make the final claim "beats -O3 on runtime," add a controlled external O3
baseline harness that:

1. compiles the same benchmark and input with LLVM 10 `opt -O3`;
2. preserves the cBench dynamic run configuration;
3. pins execution to the same CPU core;
4. uses identical warmups and repetitions;
5. compares medians and confidence intervals.

Until then, the defensible claims are runtime improvement vs the initial state
and IR instruction-count comparison vs exact -O3.
