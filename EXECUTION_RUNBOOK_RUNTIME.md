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

## Scientific boundary (guardrail — now active)

CompilerGym exposes the exact `-O3` IR cost but does not directly expose an
`-O3` runtime observation. The evaluation reports:

- hybrid runtime vs initial no-pass runtime (CompilerGym Runtime observation);
- hybrid IR instruction count vs exact `-O3` IR instruction count;
- hybrid runtime vs `opt -O3` runtime, measured ONLY by the external
  executable O3 baseline harness (section 10, `evaluation/o3_runtime_harness.py`)
  using identical inputs, warmups, CPU affinity, and repetitions.

Guardrail: never claim runtime superiority over `-O3` from CompilerGym
Runtime-vs-initial numbers. Any such claim must cite the harness output
(`results/o3_runtime_vs_o3_summary.json`). As of the Aug 2026 corrected
protocol run the measured result is a geometric-mean **0.99×** speedup vs
`clang -O3` over all 22 test benchmarks (wins 12/22, Wilcoxon p=0.50) — a
statistical tie with `-O3` on runtime (and the earlier 0.95× "hybrid looks
competitive" figure used O0-level codegen and is superseded, see section 10);
the harness is the instrument for closing that gap.

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
placeholders and are now REJECTED by the CLI ("--model-type dqn_torch/ppo are
not implemented") instead of silently training a different algorithm. Actions
are one-hot encoded (a single numeric action id imposed an artificial ordinal
relationship between unrelated passes); the committed rl_agent.joblib was
retrained with this encoding.

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

- selected ordered pass sequence (no-op actions are masked per state, so a
  pass is never repeated in an unchanged state);
- termination reason (`max_steps`, `repeated_state`, `no_effect`,
  `all_actions_tried`, `stop`, `zero_ir`);
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

## 10. External O3 executable runtime baseline — IMPLEMENTED

`evaluation/o3_runtime_harness.py` implements the controlled external O3
baseline. It satisfies all five requirements:

1. compiles the same benchmark and input three ways from the same O0 bitcode
   (`Benchmark.proto.program.contents`, identical to the environment's start
   state): `clang -O3` (clang's own full O3 pipeline — the reference
   baseline), `opt -O3` (kept as a sanity check), and the hybrid final IR as a
   pre-pass; ALL arms are then built with clang's `-O3` codegen so the
   comparison isolates the middle-end pass sequence;
2. preserves the benchmark dynamic run configuration — native builds reuse the
   benchmark's `build_cmd` template (`$CC` -> bundled clang, `$IN` -> bitcode)
   and executions reuse `pre_run_cmd` / `run_cmd` (including cBench input setup
   such as `echo 1 >_finfo_dataset`);
3. pins execution to the same CPU core — every run goes through
   `taskset -c <cpu> /bin/sh -c ...`, and the three arms are interleaved to
   cancel thermal/load drift;
4. uses identical warmups and repetitions for all arms (`--warmup 1
   --reps 5`; the Aug 2026 v2 waves used `--reps 3`);
5. compares medians and 95% bootstrap confidence intervals, plus paired
   Wilcoxon signed-rank tests against both baselines and an output-hash
   equality check across the three binaries.

The hybrid final IR is dumped to bitcode by `training/inference.py`
(`hybrid_optimize_benchmark(..., dump_bitcode_to=...)`).

Run (one process per wave, disjoint `--benchmarks` subsets for parallelism):

```bash
python evaluation/o3_runtime_harness.py measure \
  --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
  --sl-model-dir models/supervised --rl-model-dir models/reinforcement \
  --max-steps 8 --warmup 1 --reps 3 --cpu 4 --timeout 120 \
  --inputs 0,largest \
  --workdir results/o3_harness_work --output results/o3_wave1.json
python evaluation/o3_runtime_harness.py summarize \
  --results results/o3_wave1.json results/o3_wave2.json \
  --output results/o3_runtime_vs_o3_summary.json
```

Protocol note (why the v2 waves replaced the earlier ones): plain
`clang module.bc` without an -O flag emits effectively O0-level codegen
(empirically ~25.7 KB of asm vs ~17.5 KB for `-O2`/`-O3` on dijkstra), so the
pre-2026 waves that built with default codegen compared two O0-codegen
binaries — a weak baseline that let IR-level differences show up as large
runtime wins that do not survive real `-O3` codegen. Since the v2 protocol, the
harness always compiles with `-O3` codegen and adds the `clang -O3` arm.

Measured result (v2 protocol, Aug 2026, `results/o3_runtime_vs_o3_summary.json`):

- All 22 held-out test benchmarks (including ispell/lame via
  `--include-fallback`), largest-baseline-median input per benchmark,
  deduplicated across waves.
- Geo-mean speedup hybrid vs `opt -O3`: **0.993×** (wins 12/22, Wilcoxon
  p = 0.34); vs `clang -O3`: **0.990×** (wins 12/22, Wilcoxon p = 0.50) — a
  statistical tie with `-O3`.
- On cBench benchmarks with clang-O3 medians ≥ 0.1 s hybrid wins 4/6: gsm
  2.au **1.164×** (0.74 s), tiff2bw 17.nocomp.tif 1.013×, bzip2 8.bz2 1.007×,
  dijkstra 9.dat 1.004×; losses jpeg-c 17.ppm 0.968× and tiff2rgba
  11.nocomp.tif 0.945×.
- `opt -O3` and `clang -O3` agree within ~1%, validating both baselines.
- All runs byte-identical across the three arms (`outputs_match=true`).

Conclusion: with the current short IR-focused learned sequences the hybrid
optimizer does NOT beat the full `-O3` pipeline on runtime when measured
correctly — the earlier large-input wins (dijkstra 1.42×, tiff2rgba 1.36×,
bzip2 1.24×) collapse to ~1.00× under `-O3` codegen because the backend
re-optimizes the IR-level differences away, and the overall geo-mean is a tie
(0.99×). The one benchmark that changed direction under the corrected protocol
is gsm (0.44× loss -> **1.164× win**), where the learned `-sroa`-heavy
sequence helps the backend. The IR-count advantage from section 8 still does
not carry over to runtime; the harness is the controlled, reproducible
instrument for the next research step (runtime-aware z-scored reward and
longer learned sequences with STOP, evaluated against this `-O3` codegen
target).

Harness robustness notes (Aug 2026): `resolve_input` swaps the whole numbered
dataset family so multi-file benchmarks (stringsearch: `1.txt` + `1.s.txt`)
stay consistent, and `largest` selection preserves the chosen file exactly
(tiff2rgba: `1.tif` -> `17.nocomp.tif`, not `17.tif`). `summarize`
deduplicates benchmarks measured in both default-input and large-input waves.
Input indices index the lexicographically-sorted numeric files in the
benchmark's data directory (so `--inputs 9` is NOT `9.dat` for cBench data
sets — verify with the printed `input_file`). A benchmark's "largest" input by
file size can be impractically slow (`dijkstra` 20.dat: minutes per run) —
pass explicit `--inputs` indices to pick a larger-but-bounded input instead.

Defensible claims as of this run:

- runtime improvement vs the initial no-pass state (CompilerGym Runtime
  observation);
- IR instruction-count comparison vs exact -O3;
- runtime-vs-O3 measured by the harness above (currently 0.99× vs both
  baselines — a statistical tie with `-O3`, not a win).
