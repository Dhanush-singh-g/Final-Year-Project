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
protocol run the measured result is a geometric-mean **~1.0×** speedup vs
`clang -O3` over all 22 test benchmarks (wins 11/22, Wilcoxon p=0.45) — a
statistical tie with `-O3` on runtime (the earlier 0.95× "hybrid looks
competitive" figure used O0-level codegen and the first 0.99× figure is
superseded by the fixed-inference re-measurement, see section 10); the
harness is the instrument for closing that gap.

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
  --max-steps 15 --warmup 1 --reps 3 --cpu 4 --timeout 120 \
  --inputs 0,largest \
  --workdir results/o3_harness_work --output results/o3_wave1.json
python evaluation/o3_runtime_harness.py summarize \
  --results results/o3_wave1.json results/o3_wave2.json \
  --output results/o3_runtime_vs_o3_summary.json
```

To train the runtime-aware scorer on a comparable target, z-score the runtime
column per benchmark first (sub-10 ms rows are noise-dominated; the z-scored
target removes the benchmark-identity shortcut, see below):

```bash
python scripts/zscore_dataset.py \
  --input datasets/processed/hybrid_dataset_scaled.csv \
  --output datasets/processed/hybrid_dataset_scaled_z.csv
python training/train_sl.py \
  --input datasets/processed/hybrid_dataset_scaled_z.csv \
  --target z_runtime_improvement_pct \
  --output-dir models/supervised_z
```

Protocol note (why the v2 waves replaced the earlier ones): plain
`clang module.bc` without an -O flag emits effectively O0-level codegen
(empirically ~25.7 KB of asm vs ~17.5 KB for `-O2`/`-O3` on dijkstra), so the
pre-2026 waves that built with default codegen compared two O0-codegen
binaries — a weak baseline that let IR-level differences show up as large
runtime wins that do not survive real `-O3` codegen. Since the v2 protocol, the
harness always compiles with `-O3` codegen and adds the `clang -O3` arm.

Measured result (v2 protocol, **re-measured Aug 2026 under the fixed
inference**, `results/o3_runtime_vs_o3_summary.json`):

- All 22 held-out test benchmarks (including ispell/lame via
  `--include-fallback`), largest-baseline-median input per benchmark,
  deduplicated across waves. Re-measured with the post-review inference
  (no-op actions truly terminate and are masked per state) — the pass
  sequences are essentially unchanged from the first v2 waves because those
  runs had already terminated at the first no-op.
- Geo-mean speedup hybrid vs `opt -O3`: **1.062×** (wins 9/22, Wilcoxon
  p = 0.27); vs `clang -O3`: **1.018×** (wins 11/22, Wilcoxon p = 0.45) —
  still a statistical tie with `-O3`. On cBench benchmarks with substantial
  (≥ 0.1 s) inputs the geo-mean vs `clang -O3` is **1.000×**; the overall
  geo-mean is inflated by sub-10 ms CHStone/csmith rows dominated by process
  startup noise (e.g. lame 1.39×, csmith/24 1.47× at ~1–2 ms medians).
- Large-input cBench: bzip2 8.bz2 1.005×, gsm 2.au 1.012×, bitcount 1.026×,
  dijkstra 9.dat 0.984×, jpeg-c 17.ppm 0.975×, tiff2bw 17.nocomp.tif 0.972×,
  tiff2rgba 11.nocomp.tif 0.945×, stringsearch 4.txt 0.992× (all ≈1.0×).
- `opt -O3` and `clang -O3` agree within ~1%, validating both baselines.
- All runs byte-identical across the three arms (`outputs_match=true`).

Conclusion: with the current short IR-focused learned sequences the hybrid
optimizer does NOT beat the full `-O3` pipeline on runtime when measured
correctly — the earlier large-input wins (dijkstra 1.42×, tiff2rgba 1.36×,
bzip2 1.24×) collapse to ~1.00× under `-O3` codegen because the backend
re-optimizes the IR-level differences away, and the overall geo-mean is a tie
(~1.0×; the earlier 0.99× figure is superseded by this re-measurement). The
IR-count advantage from section 8 still does not carry over to runtime; the
harness is the controlled, reproducible instrument for the next research step
(runtime-aware z-scored reward evaluated against this `-O3` codegen target).

STOP is now a learned RL action (Aug 2026): the agent's action vocabulary
includes `-stop` (`training/train_rl.py::synthesize_stop_transitions`
augments the replay buffer with synthetic terminal STOP rows — reward 0,
done=True — so fitted-Q learns Q(state, STOP)). Inference uses the learned
Q(STOP) by default when the agent is loaded; the harness `--max-steps`
default is 15.

Large-input pass-quality pipeline (full sweep, Aug 2026):
`scripts/generate_large_input_dataset.py --benchmark benchmark://cbench-v1/gsm
--inputs 11 --output datasets/processed/gsm_large_input_passes.csv` builds
all 31 curated pass variants natively and times them on the chosen input
(~2 min/benchmark at ~0.6-1 s workloads; use `--warmup 0 --runs 3` and pick a
bounded input — bzip2's largest is 4.3 s/run, tiff's largest is 143 MB). The
full 8-benchmark sweep (gsm 2.au, dijkstra 9.dat, jpeg-c 17.ppm, bzip2 30.bz2,
tiff2rgba/tiff2bw 15.nocomp.tif, bitcount, stringsearch 4.txt) confirms a real
and stable per-pass runtime ordering (best `-loop-unroll` +4.4% mean vs O0,
worst `-sroa` −2.0%; 24/31 passes beat O0) — this is the review-demanded
*global-best-pass* baseline, now measured.

STRUCTURAL RESULT: a leave-one-benchmark-out scorer trained on the other 7
benchmarks gets test top-3 = **0% on every held-out benchmark** (mean R²
−0.005). Verified root cause: all 31 rows of a benchmark share one pre-state
signature, so pre-state features cannot discriminate passes within a
benchmark, and no pass is positive on all 8 (no cross-benchmark signal). The
raw-target model's earlier "signal" was benchmark identity, which z-scoring
removes. The dataset design fix is a multi-state transition dataset (pre-state
features varying across rows, e.g. relabeling the RL replay buffer's unique
states with large-input runtime) — the committed scorer should NOT be
retrained on the current single-shot data (it would be a 0%-ranking model).

Next steps from here: (a) add a fixed-sequence arm to the harness
(`-loop-unroll -loop-vectorize -argpromotion …` as the review-demanded
fixed-curated-sequence baseline) and measure it on the 8 large-input
benchmarks; (b) build the multi-state dataset (replay buffer states timed
natively on large inputs) and retrain the scorer on it.

**Longer-horizon experiment (Aug 2026, do not re-run casually):** relaxing
the first-no-op termination (`no_op_limit=max_steps`) let the learned policy
emit longer sequences (dijkstra 4 → 15 passes, IR 450→264) but runtime vs
`clang -O3` got worse (0.957× vs 0.984× for the short sequence) — the tail
was wasted no-op budget and the backend re-optimizes the extra IR away.
First-no-op termination therefore remains the harness's measured
configuration; re-enable longer horizons only after the scorer is retrained
on data that justifies continuing past a no-op.

Z-scored runtime reward (step 4 of the plan, implemented Aug 2026): raw
`runtime_improvement_pct` is cross-program-incomparable — each benchmark's
candidate-pass distribution has a different mean/scale (gsm mean −21% vs
another benchmark +57%), so a scorer on raw values can learn benchmark
identity instead of pass quality. `scripts/zscore_dataset.py` adds a
`z_runtime_improvement_pct` column (per-benchmark z-score via
`scripts/reward.py::per_benchmark_zscore`), and `train_sl.py` accepts it as a
target. Trained on the z-scored target, the scorer's test top-3 pass ranking
drops from 9.1% (raw, ≈ random 9.7%) to **0%** with test R² ≈ 0: once the
benchmark-identity shortcut is removed, the pass-quality signal is not
learnable from the current data (short sub-10 ms workloads, ~3.1k train rows
across 31 passes). This is the controlled confirmation that runtime-aware
training needs longer workloads and more per-benchmark coverage, not just a
different target.

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

Fixed-sequence baseline arm (implemented + measured Aug 2026):
`measure --sequence=-loop-unroll,-loop-vectorize,-loop-deletion,
-argpromotion,-globaldce` applies the static list to the same O0 bitcode and
times it as a 4th arm (`fixed`) with `speedup_fixed_vs_*` and
`speedup_hybrid_vs_fixed` fields + summary aggregates. Measured on the 8
large-input cBench benchmarks (same inputs as the sweep, 5 interleaved reps):

- fixed geo-mean vs clang -O3 = **1.015×** (first positive runtime result;
  driven by tiff2rgba 1.197× with non-overlapping 95% CIs);
- **hybrid is beaten by the fixed list**: geo-mean hybrid-vs-fixed 0.977×
  (split 4-4) — the IR-based scorer's `-sroa`/`-newgvn` picks are re-optimized
  by the backend while the loop transforms actually move runtime on tiff2rgba;
- the fixed list, not the learned policy, is now the baseline to beat.
  `results/o3_runtime_fixed_arm_summary.json`, `results/o3_wave_fixed_*.json`.

REPLICATION (Aug 2026, 15 reps x 3 inputs on tiff2rgba,
`results/o3_wave_fixed_tiff2rgba_rep.json`): the 1.197x was partly baseline
noise. Fixed vs clang -O3: **1.109x on 15.nocomp.tif** (non-overlapping CIs),
1.009x on 11.tif, 1.013x on 23.nocomp.tif (geo-mean ~1.04x); hybrid loses on
every input (0.86x-0.97x). The advantage concentrates where pixel-loop work
dominates (uncompressed input).

MULTI-STATE DATASET + LOOP-FOCUSED SCORER (max-scale, Aug 2026):
`scripts/generate_multistate_dataset.py` builds the transition structure the
single-shot design lacked — O0 plus states from IR-reducing scalar prefixes,
with the 8-pass loop subset timed natively at each state. State acceptance
requires BOTH a new bitcode signature AND a feature-vector distance >= 0.05
from every accepted state (autophase proportions + relative IR). The
signature check alone is insufficient: -memcpyopt changes the bitcode while
leaving model-visible features identical (distance 0.0000), which silently
duplicates rows — the diversity guard rejects those. SCALED to EVERY
runnable benchmark in the environment: 14 cBench (real inputs) + 12 CHStone
+ 9 csmith (both via `--fallback`: build ./a.out, run with no inputs) = 35
benchmarks x 3 distinct states x 8 passes = 840 rows, per-(benchmark,state)
z-scored (`datasets/processed/multistate_combined_z.csv`, gitignored).
CHStone/csmith runtimes are 1-10 ms (startup noise); they add diversity, not
runtime signal. Findings:

- in-distribution ranking (state-level split): top-1 0.154, top-3 0.615
  (random 0.125/0.375);
- 35-fold leave-one-benchmark-out: top-1 0.029 (1/35), top-3 0.343 ~= random
  (0.375), in both suites (cBench/CHStone 0.346, csmith 0.333) —
  cross-benchmark transfer is definitively absent at the maximum achievable
  scale (34 train benchmarks / 102 states);
- harness comparison (8 large-input cBench benchmarks, 5 interleaved reps,
  committed 840-row model, `results/o3_runtime_loop840_summary.json`): loop
  scorer (SL-only) 1.009x vs clang -O3, fixed top-5 loop list 1.010x,
  scorer-vs-fixed 0.999x (3-5) — a tie, both beating clang -O3 on average
  (tiff2rgba 1.085x). In-distribution only; on unseen benchmarks the scorer
  ranks ~randomly, so the fixed list is the defensible general policy;
- ANGHABENCH BOUNDARY (verified): anghabench-v1, blas/clgen/poj104/npb are
  function-level datasets — no main, no inputs, no dynamic run config — so a
  runtime-measuring pipeline cannot process them. The dataset also does not
  fit the sandbox disk (a failed install filled it — ENOSPC; cleaned). The
  35-benchmark set is the complete runnable universe here. Further scaling
  needs runnable-program corpora (SPEC/PolyBench) or synthesized
  call-harnesses for function-level code (separate build).
- retrained artifact at `models/supervised_loop_multistate/` (8 loop actions,
  target z_runtime_improvement_pct) is an IN-DISTRIBUTION ranker only; do
  NOT use it for unseen benchmarks. Fixed loop list remains the defensible
  general policy.

INPUT-INDEX TRAP (hit again Aug 2026): `--inputs` indexes the
lexicographically-sorted same-suffix files, NOT the dataset number — dijkstra
`--inputs 9` resolves to **18.dat (~28 s/run)** and jpeg-c `--inputs 17` to
7.ppm; dijkstra 9.dat is index **19**, jpeg-c 17.ppm is index **8**. Always
verify with the printed `input_file` before launching a wave.

Defensible claims as of this run:

- runtime improvement vs the initial no-pass state (CompilerGym Runtime
  observation);
- IR instruction-count comparison vs exact -O3;
- runtime-vs-O3 measured by the harness above (hybrid currently ~1.0× vs
  `clang -O3` — a statistical tie, not a win);
- a fixed top-5 loop-pass sequence is **1.015× vs `clang -O3`** (8 large-input
  cBench benchmarks) and beats the learned hybrid (0.977× hybrid-vs-fixed),
  pending replication on more reps/inputs.
