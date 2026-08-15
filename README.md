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
steps are a runtime-aware (z-scored) reward trained against the `-O3` codegen target.

### Fixed-sequence baseline arm (the first positive runtime result)

The reviews demanded a *fixed curated sequence* baseline, and the large-input
sweep (section below) finally supplied the measured ranking to build one.
`evaluation/o3_runtime_harness.py` now takes `--sequence` (comma-separated
passes): it applies the static list to the same O0 bitcode and times it as a
fourth arm (`fixed`) alongside `o3` / `clang_o3` / `hybrid`, with
`speedup_fixed_*` and `speedup_hybrid_vs_fixed` fields and summary aggregates.

Measured on the same 8 large-input cBench benchmarks as the sweep, using the
top-5 single-pass ranking (`-loop-unroll,-loop-vectorize,-loop-deletion,
-argpromotion,-globaldce`), 5 interleaved reps, `-O3` codegen on every arm
(`results/o3_runtime_fixed_arm_summary.json` + `results/o3_wave_fixed_*.json`):

| arm | geo-mean vs opt -O3 | vs clang -O3 | best | worst |
|---|---|---|---|---|
| **hybrid** (learned) | 0.977× | 0.992× | gsm 1.027× | tiff2rgba 0.940× |
| **fixed** (top-5 loop passes) | 1.000× | **1.015×** | **tiff2rgba 1.197×** | tiff2bw 0.968× |

| benchmark | input | hybrid vs clang-O3 | fixed vs clang-O3 | hybrid vs fixed |
|---|---|---|---|---|
| bitcount | (arg) | 1.006× | 1.002× | 1.005× |
| bzip2 | 30.bz2 | 1.001× | 0.988× | 1.013× |
| dijkstra | 9.dat | 0.994× | 1.004× | 0.991× |
| gsm | 2.au | 1.027× | 0.981× | 1.047× |
| jpeg-c | 17.ppm | 0.956× | 0.997× | 0.960× |
| stringsearch | 4.txt | 0.988× | 1.002× | 0.986× |
| tiff2bw | 15.nocomp.tif | 1.025× | 0.968× | 1.059× |
| tiff2rgba | 15.nocomp.tif | 0.940× | **1.197×** | 0.785× |

Interpretation — this is the project's **first measured runtime win**: a static
top-5 middle-end sequence is **1.015× faster than `clang -O3`** on average over
these 8 benchmarks (driven by tiff2rgba). Two conclusions follow. (1) The
learned hybrid is **beaten by the dumb fixed list** (0.977× hybrid-vs-fixed
geo-mean, split 4-4) — the IR-count-based scorer picks `-sroa`/`-newgvn`-type
passes that the backend re-optimizes away, while the loop transforms in the
fixed list actually move runtime on tiff2rgba. This is exactly the review's
demand to compare against a fixed curated sequence, and it sharpens the
thesis: pass selection should target runtime, and the loop-pass subset is the
promising part of the action space. (2) The fixed list, not the learned
policy, is the baseline to beat going forward.

**Replication (15 reps × 3 inputs, `results/o3_wave_fixed_tiff2rgba_rep.json`):**
the original 1.197× was partly baseline noise. Fixed vs `clang -O3` is
**1.109× on 15.nocomp.tif** (non-overlapping CIs), **1.009× on 11.tif** and
**1.013× on 23.nocomp.tif** (geo-mean ≈1.04× across inputs). The advantage is
input-dependent and concentrated where pixel-loop work dominates (the
uncompressed input); the hybrid arm loses on every input (0.86×–0.97×).

### Multi-state dataset (the dataset-design fix) and loop-focused scorer

The single-pass-from-O0 design is structurally incapable of teaching pass
selection, so `scripts/generate_multistate_dataset.py` builds the
*transition* structure instead: each benchmark contributes O0 plus states
built by applying IR-reducing scalar passes one at a time, and the 8-pass
**loop subset** (`-licm -loop-rotate -loop-unroll -loop-vectorize
-loop-deletion -loop-unswitch -loop-distribute -indvars`) is timed natively
at every state on the large input.

**State-diversity guard** (added after the first version's verified failure
mode): a candidate state is accepted only if its bitcode signature is new AND
its feature vector (autophase proportions + relative IR) is at least 0.05
from *every* accepted state. Without this, near-duplicate states pass the
signature check — `-memcpyopt` changes the bitcode while leaving the
model-visible features identical (measured distance 0.0000 vs 0.25+ for
genuine states), silently tripling duplicate rows. Audit of the scaled build:
global minimum pairwise state distance **0.0657** — no near-duplicates.

**Scaled build (Aug 2026): every runnable benchmark in the environment** —
14 cBench (real inputs) + 12 CHStone (fallback `./a.out`) + 9 csmith
(fallback) = **35 benchmarks × 3 genuinely distinct states × 8 loop passes =
840 rows**, z-scored per (benchmark, state)
(`datasets/processed/multistate_combined_z.csv`, gitignored). CHStone/csmith
run at 1–10 ms (startup-dominated), so their z-scores are noise — they add
benchmark/state diversity, not runtime signal.

Findings (35-fold leave-one-benchmark-out — the maximum power this
environment allows):

- **In-distribution ranking works**: state-level split gives **top-1 0.154,
  top-3 0.615** (random 0.125 / 0.375).
- **It does NOT transfer across benchmarks**: LOBO gives **top-1 0.029
  (1/35), top-3 0.343 ≈ random (0.375)**, in both suites (cBench/CHStone
  0.346, csmith 0.333). Even with 34 training benchmarks × 102 diverse
  states, no transferable state→pass-quality mapping emerges.
- **AnghaBench-scale is not reachable in this environment, and not for
  runtime at all**: AnghaBench (and BLAS/CLgen/POJ104/NPB here) are
  function-level — no `main`, no inputs, no dynamic run config — so a
  runtime-measuring pipeline cannot process them; the dataset also does not
  fit on this sandbox's disk (a failed install filled it; cleaned up). The
  35-benchmark set above is the complete runnable universe. Scaling further
  requires runnable-program corpora (SPEC/PolyBench) or synthesizing
  call-harnesses for function-level code — a separate build.
- **Harness comparison (in-distribution, committed 840-row model,
  `results/o3_runtime_loop840_summary.json`):** the retrained loop scorer
  (SL-only, 8 actions) vs the fixed top-5 loop list vs `clang -O3` on the 8
  large-input cBench benchmarks, 5 interleaved reps: **loop scorer 1.009× vs
  `clang -O3`**, fixed list **1.010×**, scorer-vs-fixed **0.999×** (3-5) —
  a tie, with both arms beating `clang -O3` on average (tiff2rgba 1.085× for
  the scorer, picking `-indvars`). On *unseen* benchmarks the scorer ranks
  loop passes ~randomly (LOBO 0.343 ≈ random 0.375), so the fixed list
  remains the defensible general policy.

The retrained artifact is committed at
`models/supervised_loop_multistate/` (8 loop actions, 62 features, target
`z_runtime_improvement_pct`): an in-distribution loop-pass ranker only — do
not use it for unseen benchmarks. Scaling to AnghaBench-scale benchmarks with
the same guard is the path to a generalizable runtime-aware scorer.

### STOP is now a learned RL action (longer horizons)

The RL agent's action vocabulary now includes a real `-stop` action
(`training/train_rl.py::synthesize_stop_transitions`): fitted-Q learns
Q(state, STOP) from synthetic terminal transitions (reward 0, done=True) so
the agent stops exactly when every available pass is expected to be
net-negative — no more fixed 10%-chance or stop-prior hacks. Inference's
`select_action` uses the learned Q(STOP) by default when the agent is loaded
(`--stop-prior` remains only as an SL-only fallback). The harness's
`--max-steps` default is 15. Verified: the retrained agent (32 actions,
2,227 synthetic STOP transitions) predicts Q(STOP) < 0 on normal states and
stops cleanly when continuation is worse.

### Runtime-trained RL (SL → RL end-to-end, Aug 2026)

`models/reinforcement_runtime/` is a fitted-Q agent trained on the SAME
runtime objective as the SL loop scorer, closing the IR↔runtime mismatch in
the RL layer:

- **Runtime-labeled replay buffer.** `scripts/generate_multistate_dataset.py
  --emit-replay` reuses the multi-state measurement protocol (large-input
  native timing, output-matched, diversity-guarded states) and also emits
  RL-schema rows with post-state features and `runtime_improvement` — same
  pass application, same timing, no extra measurement passes.
  `scripts/zscore_dataset.py --group-col state_id --value-col
  runtime_improvement --sync-to hybrid_reward` z-scores per (benchmark,
  state) and writes into `hybrid_reward`, the column the RL trainer consumes
  (mirrors how the SL data is z-scored). Buffer:
  `datasets/replay_buffer/rl_experiences_runtime_z.csv` (192 rows).
- **9-action space matching the SL scorer**: the 8 loop passes + a learned
  `-stop`, one-hot encoded — compatible with `models/supervised_loop_multistate`
  (the old 31-action IR-trained agent stays at `models/reinforcement/` as an
  ablation baseline).
- **Fused inference verified end-to-end**: State → SL ranks candidates → RL
  Q-values decide (learned Q(STOP) can terminate) → apply → measure runtime →
  next state. Per-episode traces confirm masking is per-(state_id, action)
  and consecutive no-ops terminate (`no_effect`).

Measured 4-arm comparison (8 large-input cBench, 5 interleaved reps,
`results/o3_runtime_rl_fused_summary.json` + `results/o3_runtime_sl_only_summary.json`):

| arm | geo-mean vs `clang -O3` | wins/losses |
|---|---|---|
| fixed top-5 loop list | 1.018× | 5-3 |
| SL-only loop scorer | 1.022× | 4-4 |
| **SL + runtime-RL (fused)** | **1.020×** | **5-3** |

All arms beat `clang -O3` on average; arm-to-arm differences are within
noise on ms-scale workloads (scorer-vs-fixed 4-4 both ways). The fused
pipeline executes genuinely — SL proposes, RL disambiguates with a learned
STOP — and matches the fixed list within noise. Honest caveats: the harness
benchmarks were in the training set (in-distribution adaptation case), and
tiff2rgba (1.13–1.16× all arms) drives most of the average.

**Scaled to the full 35-benchmark buffer (Aug 2026).** The same collector
ran over all 14 cBench + 12 CHStone + 9 csmith (`--fallback`), merged with
the original 8-benchmark buffer and deduplicated on
(benchmark, state, pass) → **840 transitions**
(`datasets/replay_buffer/rl_experiences_runtime_full_z.csv`, gitignored),
z-scored per (benchmark, state) exactly like the SL data. Retrained agent:
`models/reinforcement_runtime_full/` (9 actions, one-hot, fitted-Q, 3 iters;
the old 8-buffer agent stays at `models/reinforcement_runtime/` as the
smaller-data ablation). Re-measured 4-arm comparison (same 8 large-input
cBench, 5 interleaved reps, `results/o3_runtime_rl_fused_full_summary.json`):

| arm | geo-mean vs `clang -O3` | wins/losses |
|---|---|---|
| fixed top-5 loop list | 1.015× | — |
| SL-only loop scorer | 1.022× | 4-4 |
| SL + RL (8-buffer) | 1.020× | 5-3 |
| **SL + RL (35-buffer)** | **1.023×** | **6-2** |

The 35-buffer agent is the best arm measured to date (6/8 wins vs
`clang -O3`, best per-benchmark on gsm/dijkstra/jpeg-c/bitcount); the gains
over the 8-buffer agent come from RL training diversity, not the SL scorer
(unchanged across both fused waves). Caveats unchanged: in-distribution
only, ms-scale noise, single wave — the fixed list remains the defensible
general policy.

**Out-of-distribution holdout (Aug 2026): the RL layer's gain does not
survive OOD.** To test whether the RL Q-function transfers across benchmark
families, the fitted-Q agent was retrained on **CHStone+csmith only** (504
runtime-labeled transitions, 21 benchmarks × 3 states, no cBench rows —
`datasets/replay_buffer/rl_experiences_runtime_ood_z.csv`, gitignored) into
`models/reinforcement_runtime_ood/` (9 actions, one-hot, fitted-Q, 3 iters),
then evaluated on the same 8 large-input cBench benchmarks with the same
5-interleaved-rep protocol (`results/o3_wave_rlood_*.json` +
`results/o3_runtime_rl_ood_summary.json`):

| arm | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| fixed top-5 loop list | 1.025× | 6 |
| SL-only loop scorer | 1.022× | 4 |
| SL + RL (35-buffer, in-dist) | 1.023× | 6 |
| **SL + RL (CHStone/csmith-trained, OOD)** | **0.990×** | **3** |

The OOD agent is the **worst arm measured** — below `clang -O3` on average
and below the fixed list. The cause is visible in its pass sequences: it
opens with `-loop-distribute` on 4/8 benchmarks (gsm, jpeg-c, tiff2rgba,
tiff2bw), a no-op/harmful pass there, and it misses tiff2rgba's `-indvars`
opportunity entirely (0.993× vs 1.16–1.19× for the other arms). CHStone/csmith
reward patterns taught the Q-function a different prior that does not
transfer. Interpretation: the RL layer's measured gain was
**in-distribution adaptation, not generalization** — matching the SL LOBO
result (top-3 ≈ random on unseen benchmarks). This is the experiment that
separates the two claims: the fixed loop list remains the defensible general
policy, and the learned pipeline's value is in *adapting to a seen
benchmark's state distribution*, not transferring across families.

**OOD representation ablation (Aug 2026): part of the failure was scale
leakage; the rest is diversity.** The fitted-Q state is 62 features: 6
absolute IR/size/block/function counts + 56 autophase proportions. Diagnostics
before any change: (1) 43–50% of cBench feature values fall **outside the
CHStone+csmith training range** for the 6 absolute counts (covariate shift);
(2) a family classifier reaches 100% in-sample / 60% leave-one-benchmark-out
vs 33% chance — features carry family signal; (3) the raw OOD Q-function does
NOT extrapolate wildly on cBench (|z|>3 ≈ 0 for every action) — it
confidently applies training-family patterns (e.g. Q(`-loop-distribute`)
rises 0.07→0.17 on cBench). To distinguish scale leakage from insufficient
diversity, the OOD agent was retrained on a **scale-free representation**:
the 6 absolute counts replaced by 5 per-state ratios
(`pre_ir_per_func`, `pre_mem_frac`, `pre_size_per_inst`, `pre_blocks_per_func`,
`pre_insts_per_block` — `training/common.derive_ratio_features`), which drop
the cBench out-of-range fraction to **0–7%** while keeping the autophase
shape signal. Same CHStone+csmith → cBench evaluation
(`models/reinforcement_runtime_ood_rel/`, `results/o3_wave_rloodrel_*.json`
+ `results/o3_runtime_rl_ood_rel_summary.json`):

| arm | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| SL + RL (OOD, raw 62 features) | 0.990× | 3 |
| **SL + RL (OOD, scale-free 61)** | **1.006×** | **5** |

The scale-free representation removed the degenerate `-loop-distribute`-
everywhere behavior (per-benchmark picks now: `-loop-rotate` gsm,
`-loop-unroll` dijkstra/bitcount, `-licm` bzip2/tiff-family, `-loop-deletion`
stringsearch) and moved the OOD arm from below `clang -O3` to above it.
But it does NOT achieve transfer: tiff2rgba's `-indvars` opportunity is still
missed (0.997× vs 1.10–1.19× for fixed/in-dist arms) and the arm still trails
the fixed list (1.006× vs 1.015–1.025×). Conclusion: **both causes are real —
spurious program-size scale in the state representation contributed to the
OOD failure and is now removed; insufficient training diversity remains the
dominant bottleneck**, i.e. the state→pass→reward mapping itself is
family-specific and needs cross-family data (or a fundamentally different
policy/objective), not another feature tweak. The raw-feature OOD agent and
its waves are preserved as the 0.990× baseline (`models/reinforcement_runtime_ood/`).

**Training-diversity experiment (Aug 2026): more cross-family training data
closes part of the OOD gap — H1 supported.** Only 3 benchmark families exist
in the dataset (cBench, CHStone, csmith — verified; mibench/NPB/AnghaBench
are unavailable in this environment), so the diversity treatment is the
maximum the data permits: train the fitted-Q agent (identical scale-free 61
features, 9 actions, same z-scored reward, same procedure/seeds) on
CHStone + csmith **+ the 6 non-eval cBench benchmarks** (patricia, qsort,
sha, susan, tiffdither, tiffmedian) → 648 transitions, 27 benchmarks, with
the 8 eval benchmarks excluded from training
(`datasets/replay_buffer/rl_experiences_runtime_diverse_rel_z.csv`,
gitignored; `models/reinforcement_runtime_diverse_rel/`). Same 8-benchmark
eval (`results/o3_wave_rldiverse_*.json` +
`results/o3_runtime_rl_diverse_summary.json`):

| arm | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| SL + RL (OOD, raw 62) — preserved | 0.990× | 3 |
| SL + RL (OOD, scale-free 61) — preserved | 1.006× | 5 |
| **SL + RL (diverse, scale-free 61)** | **1.022×** | **5** |

Per-benchmark vs the scale-free OOD arm: tiff2rgba **0.997→1.147×** (the
agent now selects `-indvars` — the tiff-family opportunity it previously
missed — because tiffmedian/tiffdither in the training set taught it),
jpeg-c 0.978→1.000, bitcount 0.972→1.028, stringsearch 1.009→1.054;
gsm 1.020→0.996, dijkstra 1.010→1.000, tiff2bw 1.053→0.957 (ms-scale
noise band). Repeated passes verified legal (state-change-gated masking).
Interpretation: **the remaining OOD gap is largely insufficient training
diversity — seeing programs similar to the eval family (same suite, held
out of eval) transfers (`-indvars`), and the diverse agent ties/edges the
fixed list on tiff2rgba.** Caveat: the win count is unchanged (5/8); the
geo-mean gain is concentrated in tiff2rgba, and the added data includes
same-family (cBench) benchmarks, so this is diversity-helps, not
family-agnostic transfer. The frozen baselines (`models/reinforcement_runtime_ood/`,
`models/reinforcement_runtime_ood_rel/` + their wave JSONs) are untouched.

**Strict leave-one-family-out (Aug 2026): no genuine cross-family
generalization.** Corpus audit first: only 3 runnable families exist in this
environment. **MiBench-v1 is installed (40 programs) but NOT runnable** — its
bitcode is instrumented with the ASTEX runtime (`__astex_fopen`,
`__astex_memalloc`, …) and no runtime library ships, so linking fails
(verified at the link level). JotaiBench/AnghaBench/BLAS/CLgen/NPB are
function- or kernel-level (no `main`, no run config); SPEC/PolyBench are not
in the CompilerGym registry and would require manual dataset integration;
disk is 90% full. So the strict LOO runs across the 3 runnable families,
identical methodology (61 scale-free features, 9 actions, z-scored reward,
fitted-Q, seeds 42, SL scorer `models/supervised_loop_multistate` fixed):

| held-out family (eval) | training families | geo-mean vs `clang -O3` | wins/losses/ties | fixed-list geo-mean |
|---|---|---|---|---|
| cBench (8, large inputs, clean timing) | CHStone + csmith | **1.006×** | 5/3/0 | ~1.02× |
| csmith (9, ms-scale fallback) | CHStone + cBench | **0.996×** | 4/5/0 | 1.057× |
| CHStone (12, ms-scale fallback) | cBench + csmith | **0.968×** | 4/8/0 | 0.976× |

Training buffers (`rl_experiences_runtime_loo_*_z.csv`, gitignored) verified
leakage-free; agents `models/reinforcement_runtime_loo_csmith/` and
`models/reinforcement_runtime_loo_chstone/`; waves + summaries
`results/o3_wave_loo_*.json`, `results/o3_runtime_loo_*_summary.json`.
Interpretation: the learned policy is **neutral-to-negative on completely
unseen families** (0.996× csmith, 0.968× CHStone — the latter is ms-scale
noise around 1.0, and the fixed list is also ≈1.0 there, so neither arm
helps CHStone at fallback scale). The earlier 1.006→1.022× gain therefore
came from **same-family exposure, not cross-family transfer**: the RL layer
adapts within a seen family's state distribution and does not generalize the
state→pass→reward mapping across families. Combined with the SL 35-fold LOBO
(top-3 ≈ random) and this LOO evidence, the strongest defensible claim is:
**the learned pipeline shows within-family adaptation on seen benchmark
families (tiff2rgba `-indvars` transfer, 1.147×), no demonstrated
generalization to unseen families, and no runtime advantage over the fixed
loop list outside the in-distribution case.**

### Multi-step RL — genuine Bellman backups (Phases 5–6, the RL redesign)

**The old RL layer was not really RL.** All previous runtime replay buffers
were one-step: every row carried `done=True` and post-states were never
chained, so fitted-Q never bootstrapped and the "agent" was a myopic reward
regressor. The redesign (approved design, then implemented + measured Aug
2026) makes the training genuinely multi-step:

```
S0 --A0--> S1 --A1--> S2 --A2--> ... --STOP--> terminal
```

- **Phase 5 — collector** (`scripts/generate_rl_episodes.py`): one
  CompilerGym env per episode, `env.step` mutates the SAME LLVM state that
  the next step reads (never regenerated from S0). `state_id` is the
  `IrSha1` observation (identical identity to inference masking).
  Non-terminal transitions carry `done=False`; STOP is an explicit action
  that only ever appears as the final transition (127 real STOP rows, none
  synthesized). Start distribution 70% O0 / 30% deeper diversity-guarded
  states; per-(state, action) masking keeps no-ops out of later steps.
- **Phase 6 — trainer** (`training/train_rl.py::train_sklearn_dqn`,
  `fqi_target`): the Bellman branch is now live —
  `y = r + γ·max_{a'∈avail(s')} Q(s', a')` for non-terminal transitions
  (max over the transition's recorded `available_actions` plus the
  always-available STOP), `y = r` for terminal, self-loop (`s'==s`), and
  missing-next-state rows. Legacy one-step buffers still train as the myopic
  special case (bootstrap branch empty).
- **Verification tooling**: `scripts/inspect_rl_buffer.py` proves
  `post_state_id(t) == pre_state_id(t+1)` for every episode (0 violations),
  `scripts/verify_split_leakage.py` proves the 8 eval programs are absent,
  and `tests/test_multistep_rl.py` (8 tests) covers the terminal /
  non-terminal / STOP / self-loop / chaining / leakage properties plus an
  end-to-end check that fitted-Q actually propagates future value
  (Q(s0,a0) ≈ 1 + γ·1 > 1). `pytest` 71 passed.

**Buffer** (27 clean training benchmarks — 6 non-eval cBench + 12 CHStone +
9 csmith; the 8 eval benchmarks excluded, verified zero leakage): 258
episodes, **1,670 transitions**, avg episode length 6.5, **84.6%
non-terminal**, 127 episodes ended by real STOP / 131 by max-steps, 183
O0 / 75 deeper starts, z-scored per benchmark
(`datasets/replay_buffer/rl_experiences_multistep_z.csv`, gitignored).

**Agent**: `models/reinforcement_multistep/` — configuration deviation from
the approved design, noted explicitly: it mirrored the frozen diverse agent
(61 scale-free features, 9 actions incl. `-stop`, one-hot, **γ=0.9, 3
Q-iterations**, seed 42) on the chained buffer. 452 of 1,670 transitions got
genuine bootstrap targets; the average target rises 0.105 → 0.141 across
iterations (future value propagating). The approved configuration (γ=0.95,
20 Q-iterations) was retrained later on the identical buffer — see the
FQI-configuration follow-up below.

**Fused inference verified**: State → SL ranks candidates → multi-step RL
selects/STOP → apply → next state. The learned policy emits genuinely
multi-pass sequences, e.g. gsm `-licm → -loop-unswitch → -indvars ×3`,
jpeg-c `-licm ×3`, tiff2bw `-indvars → -licm → -indvars ×3`, tiff2rgba
`-indvars ×2` — the one-step agent's uniform `-indvars ×2` is replaced by
state-conditional adaptation (tiff2rgba's `-indvars` opportunity is still
found, learned from tiffmedian/tiffdither in training).

**Measured result** (same clean protocol as the frozen clean experiment: 8
held-out cBench, input 0, warmup 1, reps 5, `-O3` codegen, fixed top-5 loop
list arm, `results/o3_wave_msrl_*.json` + `results/o3_runtime_msrl_summary.json`;
all arms byte-identical, `outputs_match=true`):

| arm | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| fixed top-5 loop list (same wave) | 1.0026× | 4 |
| **SL + multi-step RL (this experiment)** | **1.0067×** | **4** |
| SL + multi-step RL, hybrid-vs-fixed | 1.0041× | 3 |

Per-benchmark vs `clang -O3`: gsm 1.119×, bitcount 1.046×, tiff2rgba
1.052×, jpeg-c 1.030× are the wins; bzip2 0.903×, stringsearch 0.967×,
dijkstra 0.967×, tiff2bw 0.985× are the losses. Wilcoxon p = 0.37 vs
`clang -O3` (n.s.).

**Comparison against the frozen clean arms (identical protocol):** SL-only
1.006× (6/8), SL + one-step RL 1.0032× (4/8), fixed ~1.010×. The multi-step
agent lands between SL-only and the one-step agent — **within wave-to-wave
noise of both, and still below the fixed list on average.**

**Conclusion: the multi-step redesign is implemented and verified, and it
changes the policy qualitatively (multi-pass, state-conditional sequences)
but does not close the runtime gap.** On the clean held-out set the
multi-step fused arm is 1.0067× vs `clang -O3` — a statistical tie, matching
SL-only (1.006×) and the one-step fused (1.0032×) within noise, and trailing
the fixed loop list (~1.01×). Combined with the earlier evidence (SL
35-fold LOBO ≈ random; strict LOO RL ≈ 1.0), this says the bottleneck is
not temporal credit assignment: even a correctly bootstrapped multi-step
policy cannot learn a transferable state→pass→runtime mapping from this
corpus. The fixed loop list remains the defensible general policy; the
learned pipeline's value remains in-distribution adaptation. (Caveats: all
arms are ms-scale at default inputs, single wave per arm, 1670-transition
buffer with modest per-benchmark episode counts.)

**FQI-configuration follow-up (γ=0.95, 20 iterations, approved config):**
Because the first multi-step agent reused the frozen diverse baseline's
hyperparameters (γ=0.9, 3 Q-iters) instead of the approved γ=0.95 / 20
Q-iterations, the identical 1,670-transition buffer was retrained with the
approved configuration: same 61 scale-free features, 9 actions incl.
`-stop`, one-hot, γ=0.95, 20 fitted-Q iterations, seed 42, same z-scored
rewards, same train set. All 20 iterations bootstrap the same 452
non-terminal transitions (the other 1,218 rows are terminal STOP / self-loop
/ no-next-state and use the myopic target); the average Bellman target rises
monotonically 0.163 → 0.798 across iterations (min −5.803 → −4.978, max
4.287 → 14.225), confirming genuine future-value propagation under the
approved discount. Artifact `models/reinforcement_multistep_g095_it20/`
(γ=0.95, 20 Q-iters; the γ=0.9/3-iter model is preserved untouched).

Measured with the identical clean protocol (same 8 held-out cBench, input 0,
warmup 1, reps 5, `-O3` codegen, fixed top-5 arm,
`results/o3_wave_msrl_g095_*.json` + `results/o3_runtime_msrl_g095_summary.json`;
all arms byte-identical, `outputs_match=true`):

| arm | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| fixed top-5 loop list (same wave) | 0.9652× | 3 |
| **SL + multi-step RL, γ=0.95/20 (this experiment)** | **0.9790×** | **4** |
| SL + multi-step RL, γ=0.95/20, hybrid-vs-fixed | 1.0144× | 6 |

Per-benchmark vs `clang -O3`: bitcount 1.066×, bzip2 1.062×, dijkstra
1.088×, tiff2bw 1.038× are the wins; gsm 0.983×, jpeg-c 0.937×,
stringsearch 0.778×, tiff2rgba 0.922× are the losses. Wilcoxon p = 0.63 vs
`clang -O3` (n.s.).

**Result: the approved γ=0.95 / 20-iteration configuration does NOT improve
over the γ=0.9 / 3-iteration agent or the SL/one-step baselines — it is
worse.** The γ=0.95/20 policy is qualitatively different (it converges on
`-loop-rotate`-heavy sequences — gsm/jpeg-c/bzip2/stringsearch open with
`-loop-rotate ×2` — and loses the γ=0.9 agent's targeted wins on gsm
(1.119× → 0.983×), tiff2rgba (1.052× → 0.922×) and jpeg-c (1.030× → 0.937×),
while stringsearch collapses to 0.778×). The 20-iteration FQI appears to
overfit the buffer's reward structure: the deeper backups amplify the
value of the dominant loop passes in the training distribution, and that
bias does not transfer to the held-out programs. Sequential-RL
configuration is therefore not the lever that closes the gap; the
bottleneck remains the transferable state→pass→runtime mapping.

| arm (clean protocol, all frozen or measured this turn) | geo-mean vs `clang -O3` | wins/8 |
|---|---|---|
| fixed top-5 loop list (msrl γ=0.9 wave) | 1.0026× | 4 |
| SL-only loop scorer | 1.006× | 6 |
| SL + one-step RL | 1.0032× | 4 |
| SL + multi-step RL, γ=0.9, 3 iters (preserved) | 1.0067× | 4 |
| SL + multi-step RL, γ=0.95, 20 iters (this turn) | 0.9790× | 4 |

### Large-input runtime signal (full 8-benchmark sweep)

`scripts/generate_large_input_dataset.py` builds each curated pass variant
natively (via the O3 harness's input-resolution/timing machinery) and measures
runtime on an explicit **large** input, emitting the standard `pre_*` feature
columns so the SL trainer can consume it directly. Full sweep (Aug 2026,
`datasets/processed/*_large_input_passes*.csv` + `large_input_combined_z.csv`,
gitignored):

| benchmark | input | O0 med | per-pass spread | signal quality |
|---|---|---|---|---|
| gsm | 2.au | 0.645 s | −0.6% … +4.1% | clean |
| dijkstra | 9.dat | 0.641 s | −1.8% … +2.8% | clean |
| jpeg-c | 17.ppm | 1.02 s | −1% … +2% | clean |
| bzip2 | 30.bz2 | 1.06 s | −7.4% … +2.8% | clean |
| tiff2rgba | 15.nocomp.tif | 0.25 s | −5.6% … +9.6% | decent |
| tiff2bw | 15.nocomp.tif | 52 ms | −10% … +8.4% | noisy |
| bitcount | (fixed arg) | 32 ms | −11% … +2% | noisy |
| stringsearch | 4.txt | 27 ms | −7% … +7% | noisy |

**Finding 1 — the pass ordering is real, and stable.** At large inputs the
fixed ranking of single passes has a wide spread (mean effect vs O0:
`-loop-unroll` +4.4%, `-loop-vectorize` +3.6%, `-loop-deletion` +3.3%,
`-argpromotion` +3.2%, `-globaldce` +3.0% … `-newgvn` +0.4%, `-sroa` −2.0%;
24/31 passes beat O0 on average). This is the review-demanded
*global-best-pass* baseline, now measured rather than speculative.

**Finding 2 — the single-pass-from-O0 dataset design is structurally
incapable of teaching pass *selection*.** All 31 rows of a benchmark share
one pre-state (verified: exactly 1 distinct pre-state signature per
benchmark), so the model cannot discriminate between passes within a
benchmark — and no pass is positive on all 8 benchmarks (or negative on
all 8), so there is no cross-benchmark signal to grab either. A
leave-one-benchmark-out scorer trained on the other 7 benchmarks gets
**test top-3 = 0% on every single held-out benchmark** (mean R² −0.005). The
earlier raw-target "signal" was entirely benchmark identity (per-benchmark
mean runtime), which z-scoring correctly removes. The dataset design fix is
multi-state transitions — rows whose pre-state features actually vary (the RL
replay-buffer structure), which is the next build.

**Measured result (Aug 2026): longer horizons do not help with the current
scorer.** An experiment relaxed the first-no-op termination
(`no_op_limit=max_steps`) so the STOP/longer-horizon policy could act.
Sequences grew (dijkstra 4 → 15 passes, IR 450→264) but the tail was wasted
no-op budget, and runtime vs `clang -O3` got *worse* (0.957× vs 0.984× for the
short sequence) — the backend re-optimizes the extra IR away. First-no-op
termination therefore remains the harness's measured configuration; the
longer-horizon machinery is in place and can be re-enabled once the scorer is
trained on data that justifies continuing past a no-op.

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
  --max-steps 15 --measure-runtime --output results/hybrid_test_results_scaled.json

# 6. External O3 executable runtime baseline (runbook §10) — run in parallel waves
python evaluation/o3_runtime_harness.py measure \
  --processed-csv datasets/processed/hybrid_dataset_scaled.csv \
  --sl-model-dir models/supervised --rl-model-dir models/reinforcement \
  --max-steps 15 --warmup 1 --reps 5 --cpu 4 --timeout 120 --inputs 0,largest \
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
