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

RUNTIME-LABELED RL (SL -> RL end-to-end, Aug 2026):

The RL side was made consistent with the runtime-based SL side. Old state:
`models/reinforcement/` was trained on IR rewards (reward_space
IrInstructionCountO3; `runtime_improvement` column all zeros in the scaled
buffer) on a 31-action space that does not match the loop scorer's 8 actions.

New pipeline (committed):

1. `scripts/generate_multistate_dataset.py --emit-replay` reuses the exact
   multi-state measurement protocol (large-input native timing, output
   matched, diversity-guarded states) and ALSO writes RL-schema rows with
   POST-state features + `runtime_improvement` — same pass application, same
   timing, zero extra measurement passes.
2. `scripts/zscore_dataset.py --group-col state_id --value-col
   runtime_improvement --sync-to hybrid_reward` converts the raw delta to a
   per-(benchmark, state) z-score and writes it into `hybrid_reward`, the
   column the RL trainer consumes (matching how the SL data is z-scored).
3. `training/train_rl.py` retrains fitted-Q on the **9-action space (8 loop
   passes + learned `-stop`)**, one-hot encoding — same pass set as the SL
   loop scorer. Artifact: `models/reinforcement_runtime/` (384 transitions,
   9 actions). The old IR-trained 31-action agent is preserved untouched at
   `models/reinforcement/` as the ablation baseline.
4. Harness waves with the fused path (`--sl-model-dir
   models/supervised_loop_multistate --rl-model-dir
   models/reinforcement_runtime`): SL ranks candidates -> RL Q-values decide
   (learned Q(STOP) can terminate) -> apply -> time. Verified per-episode
   trace: masking is per-(state_id, action); a repeated pass only fires after
   a genuine state change, and consecutive no-ops terminate via `no_effect`.
   Wave artifacts: `results/o3_wave_rlfused_*.json` (fused) +
   `results/o3_wave_slonly_*.json` (SL-only, `--rl-model-dir results/no_rl`)
   + summaries `results/o3_runtime_rl_fused_summary.json` /
   `results/o3_runtime_sl_only_summary.json`.

FOUR-ARM COMPARISON (8 large-input cBench, 5 interleaved reps, same wave
protocol for all arms, geo-mean speedup vs `clang -O3`):

| arm | geo-mean vs clang-O3 | wins/losses vs clang-O3 |
|---|---|---|
| fixed top-5 loop list | 1.018x | 5-3 |
| SL-only loop scorer | 1.022x | 4-4 |
| SL + runtime-RL (fused) | 1.020x | 5-3 |

- All three learned/fixed arms beat `clang -O3` on average; the differences
  BETWEEN arms (scorer-vs-fixed 1.004x fused / 1.016x slonly, 4-4
  hybrid-vs-fixed) are within run-to-run noise on ms-scale benchmarks.
- The RL layer is now active in the measured pipeline and neither helps nor
  hurts on average vs SL-only — it selects the same top passes (Q argmax over
  the SL candidates) with a learned STOP.
- tiff2rgba remains the standout for all arms (fixed 1.147x, SL-only 1.163x,
  fused 1.133x); bzip2 and jpeg-c are slight losses for the learned arms.
- SCALED TO THE FULL 35-BENCHMARK BUFFER (Aug 2026): the same collector
  ran over all 14 cBench + 12 CHStone + 9 csmith (--fallback), merged with
  the original 8-benchmark buffer, deduplicated on (benchmark, state, pass)
  -> 840 transitions (`datasets/replay_buffer/rl_experiences_runtime_full_z.csv`,
  gitignored), z-scored per (benchmark, state). Retrained agent
  `models/reinforcement_runtime_full/` (9 actions incl. -stop, one-hot,
  3 Q-iterations; the 8-buffer agent stays at `models/reinforcement_runtime/`
  as the smaller-data ablation). Re-measured wave
  `results/o3_wave_rlfused_full_*.json` + summary
  `results/o3_runtime_rl_fused_full_summary.json`:

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | fixed top-5 loop list | 1.015x | -- |
  | SL-only loop scorer | 1.022x | 4 |
  | SL + RL (8-buffer) | 1.020x | 5 |
  | SL + RL (35-buffer) | **1.023x** | **6** |

  The 35-buffer agent is the best arm measured to date; gains over the
  8-buffer agent come from RL training diversity (SL scorer unchanged across
  both fused waves). CAVEATS (stated, not optimized away): harness
  benchmarks were in the training set (in-distribution adaptation case);
  ms-scale noise on 6 of 8 benchmarks; single wave per arm. The claim is:
  the fused SL->runtime-RL pipeline executes end-to-end and, trained on the
  full 840-transition runtime buffer, matches/edges the fixed list and
  SL-only, with the tiff2rgba signal present in all arms.

- OUT-OF-DISTRIBUTION HOLDOUT (Aug 2026): to separate in-distribution
  adaptation from generalization, retrained the fitted-Q agent on
  CHStone+csmith ONLY (504 runtime-labeled transitions, 21 benchmarks x 3
  states, zero cBench rows -> `datasets/replay_buffer/rl_experiences_runtime_ood_z.csv`,
  gitignored) into `models/reinforcement_runtime_ood/` (9 actions incl.
  -stop, one-hot, 3 Q-iterations), then measured the fused pipeline on the
  SAME 8 large-input cBench with the SAME protocol
  (`results/o3_wave_rlood_*.json` + `results/o3_runtime_rl_ood_summary.json`):

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | fixed top-5 loop list | 1.025x | 6 |
  | SL-only loop scorer | 1.022x | 4 |
  | SL + RL (35-buffer, in-dist) | 1.023x | 6 |
  | SL + RL (CHStone/csmith-trained, OOD) | **0.990x** | **3** |

  RESULT: the OOD agent is the worst arm measured — below clang-O3 on
  average. Its Q-prior (learned from CHStone/csmith reward patterns) opens
  with -loop-distribute on 4/8 benchmarks where it is a no-op/harmful, and
  misses tiff2rgba's -indvars opportunity entirely (0.993x vs 1.16-1.19x
  for the other arms). CONCLUSION: the RL layer's measured gain is
  IN-DISTRIBUTION ADAPTATION, not generalization — consistent with the SL
  LOBO result (top-3 ~ random on unseen benchmarks). The fixed loop list
  remains the defensible general policy; the learned pipeline's value is
  adapting to a seen benchmark's state distribution, not transferring across
  families. To fix OOD transfer, the state/action representation itself must
  change (features that normalize away benchmark identity, or a policy
  trained jointly on runnable corpora) — not more rows from the same
  families.

- OOD REPRESENTATION ABLATION (Aug 2026): the fitted-Q state is 62 features
  (6 absolute IR/size/block/function counts + 56 autophase proportions).
  Diagnostics BEFORE the change: (1) 43-50% of cBench values fall outside
  the CHStone+csmith training range on the 6 absolute counts (covariate
  shift); (2) family classifier: 100% in-sample / 60% LOBO vs 33% chance;
  (3) the raw OOD Q-function does NOT extrapolate wildly on cBench
  (|z|>3 ~ 0 per action) — it confidently applies training-family patterns.
  CHANGE: replaced the 6 absolute counts with 5 per-state ratios
  (`training/common.py::derive_ratio_features`: pre_ir_per_func,
  pre_mem_frac, pre_size_per_inst, pre_blocks_per_func, pre_insts_per_block),
  which cut the cBench out-of-range fraction to 0-7%. Trainer gained
  `--feature-cols` (comma-separated override; default auto-derivation
  unchanged); inference injects the ratio columns online when the agent
  expects them (no normalization statistics needed at test time). Retrained
  OOD agent -> `models/reinforcement_runtime_ood_rel/` (61 features),
  same CHStone+csmith -> cBench eval (`results/o3_wave_rloodrel_*.json` +
  `results/o3_runtime_rl_ood_rel_summary.json`):

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | SL + RL (OOD, raw 62 features) | 0.990x | 3 |
  | SL + RL (OOD, scale-free 61) | **1.006x** | **5** |

  RESULT: scale-free representation removed the degenerate
  -loop-distribute-everywhere behavior (per-benchmark picks: -loop-rotate
  gsm, -loop-unroll dijkstra/bitcount, -licm bzip2/tiff-family,
  -loop-deletion stringsearch) and moved the OOD arm above clang-O3, but
  does NOT achieve transfer: tiff2rgba's -indvars is still missed (0.997x)
  and the arm trails the fixed list. CONCLUSION: BOTH causes are real —
  spurious program-size scale in the state representation contributed to
  the OOD failure (now removed), and insufficient training diversity is the
  DOMINANT remaining bottleneck (the state->pass->reward mapping is
  family-specific). Next lever: cross-family data / policy change, not
  another feature tweak. Raw-feature baseline preserved at
  `models/reinforcement_runtime_ood/` (0.990x).

- TRAINING-DIVERSITY EXPERIMENT (Aug 2026): only 3 families exist (cBench,
  CHStone, csmith — verified; mibench/NPB/AnghaBench unavailable). The
  controlled treatment, keeping EVERYTHING else identical (scale-free 61
  features, 9 actions, z-scored reward, same train/eval commands, seeds
  42): training set = CHStone + csmith + the 6 NON-EVAL cBench benchmarks
  (patricia, qsort, sha, susan, tiffdither, tiffmedian) -> 648 transitions,
  27 benchmarks; the 8 eval benchmarks are excluded from training
  (verified: no leakage; the only 'gsm'/'sha' rows in training are the
  CHStone variants, different URIs). Buffer:
  `datasets/replay_buffer/rl_experiences_runtime_diverse_rel_z.csv`
  (gitignored); agent `models/reinforcement_runtime_diverse_rel/`;
  eval `results/o3_wave_rldiverse_*.json` + `results/o3_runtime_rl_diverse_summary.json`:

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | SL + RL (OOD, raw 62) — preserved | 0.990x | 3 |
  | SL + RL (OOD, scale-free 61) — preserved | 1.006x | 5 |
  | SL + RL (diverse, scale-free 61) | **1.022x** | **5** |

  Per-benchmark vs scale-free OOD: tiff2rgba 0.997 -> **1.147x** (now picks
  -indvars, learned from tiffmedian/tiffdither in training — the exact
  opportunity the OOD agents missed), jpeg-c 0.978 -> 1.000, bitcount
  0.972 -> 1.028, stringsearch 1.009 -> 1.054; gsm 1.020 -> 0.996, dijkstra
  1.010 -> 1.000, tiff2bw 1.053 -> 0.957 (ms-scale noise band; tiff2bw
  fluctuates 0.92-1.07 across waves). Repeated passes verified legal
  (state-change-gated; e.g. tiff2rgba -indvars x2, IR 58661->58688 then
  no-op termination). CONCLUSION: **H1 supported — insufficient training
  diversity is the remaining bottleneck; more cross-family data (incl.
  same-suite programs held out of eval) transfers (-indvars) and ties the
  fixed list on tiff2rgba.** Win count unchanged (5/8); the gain is
  concentrated in tiff2rgba and the added data includes same-family
  benchmarks, so this is diversity-helps, not family-agnostic transfer.
  Frozen baselines untouched (`models/reinforcement_runtime_ood/`,
  `models/reinforcement_runtime_ood_rel/` + waves).

- STRICT LEAVE-ONE-FAMILY-OUT (Aug 2026): corpus audit first — only 3
  runnable families exist. MiBench-v1 IS installed (40 programs) but NOT
  runnable: bitcode links against the ASTEX instrumentation runtime
  (__astex_fopen, __astex_memalloc, ...) which ships nowhere in the install
  (verified: `clang -O3 bitcount-1.bc -lm` -> undefined references).
  JotaiBench/AnghaBench/BLAS/CLgen/NPB are function/kernel-level; SPEC and
  PolyBench are not in the CompilerGym registry (PolyBench would need manual
  C->bitcode dataset integration; disk 90% full). So LOO ran across the 3
  runnable families, methodology identical (61 scale-free features, 9
  actions, z-reward, fitted-Q, seed 42, SL scorer fixed):

  | held-out (eval) | train families | geo-mean vs clang-O3 | W/L/T | fixed geo-mean |
  |---|---|---|---|---|
  | cBench (8, large-input, clean) | CHStone + csmith | 1.006x | 5/3/0 | ~1.02x |
  | csmith (9, ms fallback) | CHStone + cBench | 0.996x | 4/5/0 | 1.057x |
  | CHStone (12, ms fallback) | cBench + csmith | 0.968x | 4/8/0 | 0.976x |

  Buffers `rl_experiences_runtime_loo_{csmith,chstone}_z.csv` (gitignored,
  verified zero leakage), agents
  `models/reinforcement_runtime_loo_{csmith,chstone}/`, waves
  `results/o3_wave_loo_{csmith,chstone}.json` + summaries
  `results/o3_runtime_loo_*_summary.json`. RESULT: the learned policy is
  neutral-to-negative on completely unseen families (csmith 0.996x,
  CHStone 0.968x; CHStone is ms-scale noise ~1.0 and the fixed list is
  ~0.98x there too, so neither arm helps CHStone at fallback scale). The
  earlier 1.006->1.022x gain is therefore SAME-FAMILY exposure, not
  cross-family transfer. CONCLUSION: the RL layer adapts within a seen
  family's state distribution; the state->pass->reward mapping does NOT
  generalize across families. Strongest defensible claim: within-family
  adaptation on seen families (tiff2rgba -indvars, 1.147x) with no
  demonstrated generalization to unseen families and no runtime advantage
  over the fixed loop list outside the in-distribution case.

- MULTI-STEP RL REDESIGN (Phases 5-6, Aug 2026): the old runtime replay
  buffers were one-step (every row done=True, never chained), so fitted-Q
  never bootstrapped and the "RL" was a myopic reward regressor. The
  redesign makes it genuine multi-step: `scripts/generate_rl_episodes.py`
  collects chained episodes (one env per episode, `env.step` mutates the
  SAME state, `done=False` on non-terminal transitions, STOP only as the
  final action, 70/30 O0/deeper starts, per-(state,action) masking),
  `training/train_rl.py` now applies the real Bellman target
  `y = r + gamma * max_{a' in avail(s')} Q(s',a')` for non-terminal
  transitions (max includes the always-available STOP) and `y = r` for
  terminal/self-loop/no-next-state rows. Verification: `scripts/inspect_rl_buffer.py`
  proves `post(t)==pre(t+1)` chaining (0 violations on the real buffer),
  `scripts/verify_split_leakage.py` proves the 8 eval programs absent,
  `tests/test_multistep_rl.py` covers the 7 required properties + an
  end-to-end check that future value actually propagates (all 71 tests
  pass). Buffer: 258 episodes / 1670 transitions over the 27 clean training
  benchmarks (6 non-eval cBench + 12 CHStone + 9 csmith), 84.6%
  non-terminal, 127 real STOP endings, z-scored per benchmark
  (`datasets/replay_buffer/rl_experiences_multistep_z.csv`, gitignored).
  Agent: `models/reinforcement_multistep/` (61 scale-free features, 9
  actions, one-hot, **gamma 0.9, 3 Q-iters** — a CONFIG DEVIATION: the
  approved design specified gamma 0.95 / 20 Q-iters, but the first run
  mirrored the frozen diverse agent's hyperparameters for comparability;
  the approved config was retrained afterward, see below; 452/1670
  bootstrapped rows; avg target 0.105->0.141 across iterations). Measured
  on the same clean protocol (8 held-out cBench, input 0, warmup 1, reps
  5, -O3 codegen, fixed top-5 arm; `results/o3_wave_msrl_*.json` +
  `results/o3_runtime_msrl_summary.json`):

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | fixed top-5 loop list (same wave) | 1.0026x | 4 |
  | SL + multi-step RL | **1.0067x** | **4** |
  | hybrid-vs-fixed | 1.0041x | 3 |

  Per-benchmark: gsm 1.119x / bitcount 1.046x / tiff2rgba 1.052x / jpeg-c
  1.030x wins; bzip2 0.903x / stringsearch 0.967x / dijkstra 0.967x /
  tiff2bw 0.985x losses; Wilcoxon p=0.37 (n.s.). The policy is now genuinely
  multi-pass and state-conditional (gsm -licm->-loop-unswitch->-indvars x3,
  jpeg-c -licm x3, tiff2bw -indvars->-licm->-indvars x3, tiff2rgba -indvars
  x2) instead of the one-step agent's uniform -indvars x2. CONCLUSION: the
  multi-step formulation changes the policy but NOT the result class —
  1.0067x vs clang-O3 is within noise of SL-only (1.006x, 6/8) and the
  one-step fused (1.0032x, 4/8), still below the fixed list; the bottleneck
  is the transferable state->pass->runtime mapping, not temporal credit
  assignment. Fixed list remains the defensible general policy.

- MULTI-STEP FQI-CONFIG FOLLOW-UP (gamma 0.95 / 20 Q-iters, approved
  config; Aug 2026): the first multi-step agent above used gamma 0.9 / 3
  Q-iters (mirroring the frozen diverse agent) instead of the approved
  gamma 0.95 / 20. Retrained the IDENTICAL 1,670-transition buffer
  (`datasets/replay_buffer/rl_experiences_multistep_z.csv`) with the
  approved config: 61 scale-free features, 9 actions incl. -stop, one-hot,
  gamma 0.95, 20 fitted-Q iterations, seed 42, same z-rewards, same train
  set -> `models/reinforcement_multistep_g095_it20/`. Every iteration
  bootstraps the same 452 non-terminal transitions (the other 1,218 rows
  are terminal STOP / self-loop / no-next-state and use the myopic target
  y=r); avg Bellman target rises monotonically 0.163 -> 0.798 across the
  20 iterations (min -5.803 -> -4.978, max 4.287 -> 14.225) — genuine
  future-value propagation under the approved discount. `train_rl.py` log
  line extended with min/max target (logging only, no algorithm change).
  Measured with the identical clean protocol (8 held-out cBench, input 0,
  warmup 1, reps 5, -O3 codegen, fixed top-5 arm;
  `results/o3_wave_msrl_g095_*.json` +
  `results/o3_runtime_msrl_g095_summary.json`):

  | arm | geo-mean vs clang-O3 | wins/8 |
  |---|---|---|
  | fixed top-5 loop list (same wave) | 0.9652x | 3 |
  | SL + multi-step RL, gamma 0.95/20 | **0.9790x** | **4** |
  | SL + multi-step RL, gamma 0.95/20, hybrid-vs-fixed | 1.0144x | 6 |

  Per-benchmark vs clang-O3: bitcount 1.066x / bzip2 1.062x / dijkstra
  1.088x / tiff2bw 1.038x wins; gsm 0.983x / jpeg-c 0.937x / stringsearch
  0.778x / tiff2rgba 0.922x losses; Wilcoxon p=0.63 (n.s.). The gamma
  0.95/20 policy is qualitatively different — it converges on
  -loop-rotate-heavy sequences (gsm/jpeg-c/bzip2/stringsearch open with
  -loop-rotate x2) and loses the gamma 0.9 agent's targeted wins on gsm
  (1.119x -> 0.983x), tiff2rgba (1.052x -> 0.922x) and jpeg-c (1.030x ->
  0.937x), while stringsearch collapses to 0.778x. RESULT: the approved
  FQI configuration does NOT improve over gamma 0.9/3, the one-step fused
  (1.0032x) or SL-only (1.006x) — it is worse (0.9790x, below clang-O3 on
  average). 20-iteration FQI overfits the buffer's reward structure: deeper
  backups amplify the dominant loop passes of the training distribution and
  that bias does not transfer to the held-out programs. Sequential-RL
  configuration is not the lever; the bottleneck remains the transferable
  state->pass->runtime mapping. All prior artifacts preserved
  (`models/reinforcement_multistep/` and the msrl waves are untouched;
  the gamma 0.95/20 model + waves are new).

Defensible claims as of this run:

- runtime improvement vs the initial no-pass state (CompilerGym Runtime
  observation);
- IR instruction-count comparison vs exact -O3;
- runtime-vs-O3 measured by the harness above (hybrid currently ~1.0× vs
  `clang -O3` — a statistical tie, not a win);
- a fixed top-5 loop-pass sequence is **1.015-1.018× vs `clang -O3`** (8
  large-input cBench benchmarks) and beats the learned hybrid on some waves
  (0.977-1.004× hybrid-vs-fixed) — pending replication on more reps/inputs;
- the full SL -> runtime-RL pipeline (State -> SL ranks -> RL selects/STOP ->
  apply -> measure runtime -> next state) is implemented, measured, and
  matches the fixed list within noise on the 8-benchmark set.
