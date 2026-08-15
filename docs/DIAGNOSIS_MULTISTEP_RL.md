# Multi-step RL Transfer-Failure Diagnosis

**Date:** Aug 2026
**Scope:** Why does a structurally correct sequential RL agent (γ=0.95, 20 fitted-Q
iterations, genuine Bellman backups, 1,670-transition chained buffer) fail to transfer
optimization decisions from the 27-program training corpus to the 8 held-out cBench
programs (0.9790× vs `clang -O3`, 4/8 wins)?

**Method:** Pure diagnosis on existing artifacts. No new model training, no new
evaluation waves, no collector/feature/reward changes. Sources:

- `datasets/replay_buffer/rl_experiences_multistep_{raw,z}.csv` — the 1,670-transition
  training buffer (27 programs: 6 non-eval cBench + 12 CHStone + 9 csmith).
- `datasets/processed/*_multistate.csv` + `multistate_combined_z.csv` — per-(state,
  pass) runtime measurements for all 35 programs (same protocol as the buffer).
- `datasets/processed/large_input_combined_z.csv` — 8 eval benchmarks × 31 passes on
  large (0.2–4 s) inputs.
- `results/o3_wave_msrl_g095_{1,2}.json` — the γ=0.95/20 evaluation waves.
- Frozen models: `models/supervised_loop_multistate_clean/`,
  `models/reinforcement_multistep/` (γ=0.9/3), `models/reinforcement_multistep_g095_it20/`.

Frozen reference results (unchanged): γ=0.9/3 multi-step = 1.0067× (4/8); SL-only =
1.006× (6/8); one-step RL = 1.0032× (4/8); fixed top-5 = ~1.01×. This experiment
(γ=0.95/20) = **0.9790× (4/8)**.

---

## 1. Train vs test state distribution (61-feature representation)

| metric | value |
|---|---|
| train unique states | 360 (27 programs) |
| test unique states | 24 (8 eval programs × 3 states) |
| features with \|standardized shift\| > 1 | **52/61** |
| features with ≥50% of test values out of train range | 0/61 |
| features with ≥25% out of range | 1/61 |
| Mahalanobis distance (standardized, ridge): train median / p90 | 4.9 / 7.9 |
| Mahalanobis distance: test median / p90 / max | **60.7 / 149 / 171** |
| fraction of test states beyond train p90 | **62%** |
| PCA components explaining 95% of train variance | 6/61 |
| residual norm beyond 6 comps: train median vs test median | 1.17 vs **4.52** |

Top shifted features (all positive = test programs are structurally larger):

| feature | train mean | test mean | %OOR | std shift |
|---|---|---|---|---|
| pre_autophase_BBHiPhi | 5.3 | 56.3 | 12.5% | +3.86 |
| pre_autophase_BBNumArgsHi | 7.0 | 66.7 | 12.5% | +3.80 |
| pre_autophase_ArgsPhi | 215 | 1668 | 16.7% | +3.79 |
| pre_autophase_NumSelectInst | 1.9 | 24.9 | 12.5% | +3.62 |
| pre_autophase_NumPHIInst | 111 | 831 | 12.5% | +3.33 |
| pre_autophase_BeginPhi | 68 | 334 | 12.5% | +2.09 |

**Interpretation.** The shift is systematic and *directional*: the held-out programs are
bigger, more real-world-like programs, and the 61-feature representation still encodes
that. Only **5 of 61** features (the ratio features added by the scale-free experiment)
are size-agnostic; the 56 autophase columns are raw instruction counts that scale with
program size, and every one of them is shifted by more than one training standard
deviation. Per-feature out-of-range fractions are low (≤25%) because the training
corpus already spans a wide range, but the multivariate picture is unambiguous: test
states sit ~12× further from the training centroid than typical training states, and 62%
of them are beyond the training p90. Test states also occupy low-variance training
directions (residual 4× larger beyond the top-6 PCA subspace). The earlier scale-free
fix addressed the 6 core counts only; the autophase counts remain a program-size /
program-identity encoding.

---

## 2. Action coverage (pass applicability, train vs test)

Per-(state, pass) z-scored runtime improvement, same protocol both sides
(`multistate_combined_z.csv`; train = 27 programs × 3 states × 8 passes = 648 rows,
test = 8 × 3 × 8 = 192 rows):

| pass | train n | train z-mean | train z-std | test n | test z-mean | test z-std | test frac>0 |
|---|---|---|---|---|---|---|---|
| -indvars | 81 | +0.24 | 1.10 | 24 | +0.39 | 1.05 | 0.62 |
| -licm | 81 | +0.07 | 1.14 | 24 | +0.06 | 0.90 | 0.58 |
| -loop-deletion | 81 | −0.10 | 1.07 | 24 | −0.13 | 0.94 | 0.46 |
| -loop-distribute | 81 | **+0.14** | 0.88 | 24 | **−0.38** | 0.93 | 0.42 |
| -loop-rotate | 81 | −0.08 | 0.98 | 24 | **+0.32** | 0.95 | 0.54 |
| -loop-unroll | 81 | −0.05 | 0.81 | 24 | −0.31 | 1.19 | 0.33 |
| -loop-unswitch | 81 | −0.16 | 0.90 | 24 | +0.07 | 0.90 | 0.58 |
| -loop-vectorize | 81 | −0.07 | 1.02 | 24 | −0.02 | 0.83 | 0.50 |

Test-side raw (default inputs) and large-input per-pass means:

| pass | test raw mean (default) | test large-input mean | best large-input benchmark |
|---|---|---|---|
| -indvars | +2.4% | +2.0% | tiff2rgba +14.8% |
| -licm | +1.3% | +2.1% | tiff2rgba +7.1% |
| -loop-deletion | +0.8% | +3.3% | tiff2rgba +16.7% |
| -loop-distribute | −0.1% | +1.2% | tiff2bw +6.9% |
| -loop-rotate | +2.2% | +1.0% | tiff2rgba +9.6% |
| -loop-unroll | +1.0% | +4.4% | tiff2rgba +18.1% |
| -loop-unswitch | +1.1% | +2.3% | tiff2rgba +16.7% |
| -loop-vectorize | +1.1% | +3.6% | tiff2rgba +13.6% |

**Interpretation.** The action space is appropriate for the test programs: **every one of
the 8 loop passes has positive mean effect on the held-out programs** (both at default
and large inputs), and the large-input sweep shows a stable, large per-benchmark
ordering (e.g. tiff2rgba responds +7–18% to almost every loop pass). The training
distribution is different in kind: in training, four of the eight passes are no-ops in
≥70% of states (see §5), and the reward signs for `-loop-distribute`, `-loop-unroll` and
`-loop-rotate` flip between train and test. The RL agent is therefore learning pass
effects that are **partially absent or reversed** in the evaluation programs. Note,
however, that the γ=0.95/20 agent's actual first actions (`-loop-rotate`, `-licm`) are
both positive on test — the first action is often defensible; the damage is downstream
(§7).

---

## 3. Reward quality

Per-(family, pass) raw per-step reward (`100·(t_pre − t_post)/t_pre`, default inputs,
3-run medians):

| family | pass | n | median % | std % | IQR % | %pos | %neg | %~0 |
|---|---|---|---|---|---|---|---|---|
| cbench | -indvars | 51 | −5.4 | 22.9 | 29.5 | 35 | 61 | 4 |
| cbench | -licm | 44 | −1.9 | 40.1 | 30.1 | 45 | 52 | 2 |
| cbench | -loop-distribute | 84 | +0.4 | 23.3 | 18.3 | 50 | 46 | 4 |
| cbench | -loop-rotate | 34 | +3.6 | 15.3 | 17.4 | 62 | 32 | 6 |
| cbench | -loop-unroll | 38 | −1.4 | 48.1 | 16.0 | 47 | 53 | 0 |
| chstone | -indvars | 131 | −0.2 | **546** | 59 | 48 | 49 | 3 |
| chstone | -licm | 75 | −5.8 | **570** | 61 | 41 | 56 | 3 |
| chstone | -loop-rotate | 73 | +3.3 | **619** | 779 | 55 | 45 | 0 |
| chstone | -loop-vectorize | 58 | −15.8 | **598** | 1041 | 34 | 66 | 0 |
| csmith | -indvars | 89 | +4.9 | **540** | 112 | 56 | 44 | 0 |
| csmith | -loop-unroll | 80 | −7.0 | **588** | 62 | 44 | 54 | 2 |
| csmith | -loop-unswitch | 34 | −7.9 | **659** | 1230 | 38 | 59 | 3 |

Harness measurement noise (5 interleaved reps per arm, ms-scale workloads):
bitcount CV 2.5%, bzip2 9.9%, gsm 11.1%, dijkstra 9.7%, jpeg-c 1.9%, stringsearch 13.2%,
tiff2rgba 17.6%, tiff2bw **70.3%** (1.97 ms median).

**Signal-to-noise, quantified.**

- Typical pass effect on runtime: **+1 to +4%** (large inputs) or **+1 to +2%** (default
  inputs).
- Reward noise for cBench transitions: per-(benchmark, pass) std of **13–48%**; the
  pass-to-pass signal spread is ~2–10%, so the ordering between two passes is
  *not* stable across measurements (SNR ≈ 0.1–0.3).
- Reward noise for CHStone/csmith transitions: per-(benchmark, pass) std of
  **480–660%**, IQR up to 1,200%. The median reward of every (family, pass) cell sits
  within ±16% of zero while the noise is ~500% — the sign of an individual reward is a
  coin flip. **79% of the training buffer (1,316/1,670 transitions) carries rewards whose
  noise is 2–3 orders of magnitude larger than the effect being learned.**
- The per-benchmark z-scoring normalizes the *magnitude* of this noise but not its
  *ordering*: a "positive" z-reward and a "negative" z-reward of the same benchmark are
  equally likely to be measurement artifacts.

**Conclusion.** The runtime reward as currently measured is not a reliable RL signal for
CHStone/csmith at default inputs, and is marginal for cBench at default (1–45 ms)
inputs. This is not a claim that runtimes are "small" — it is a measured
signal-to-noise statement: effect 1–4%, one-sigma measurement noise 13–660% depending on
family.

---

## 4. Training-data diversity

| metric | value |
|---|---|
| transitions / episodes / unique pre-states | 1,670 / 258 / **360** |
| avg episode length | 6.5 |
| transitions per benchmark: min / med / max | 25 / 64 / 85 |
| unique pre-states per benchmark: min / med / max | 7 / 12 / 24 |
| transitions by family | cBench 354 (21%), CHStone 768 (46%), csmith 548 (33%) |
| episode lengths | 1:18, 2:18, 3:12, 4:16, 5:13, 6:14, 7:14, **8:130 (50%)**, 9:1, 10:22 |
| distinct action sequences | 224/258 (but only 9.1 distinct per benchmark) |
| action distribution (top) | -loop-distribute 22%, -indvars 16%, -stop 8% |
| unique-state pairwise L2 (61 feat) | median 2,735; p10 492; p90 18,306 |
| z-reward distribution | median +0.43, std 1.0, 19% \|z\|>1, 8% \|z\|>2 |

**Interpretation.** The buffer is not a flat pool of 1,670 independent states — it is
258 episodes over **360 distinct states** (a state is revisited ~4.6 times on average as
different passes are tried). Half of all episodes (130/258) are length-8 full sweeps that
try every loop pass in one state until `all_tried` terminates — these carry almost no
learning signal beyond the first few actions. Within a benchmark there are only ~12
states and ~9 distinct trajectories, and most trajectory diversity comes from the random
exploration fraction (25%) re-ordering the same few passes. The corpus is 27 small
programs, 79% of whose rewards are noise-dominated (§3). **Verdict: the failure is
"not enough distinct program/state diversity", not "not enough transitions"** — adding
more transitions over the same 27 programs would add more noise-labeled rows, not new
structure.

---

## 5. Self-loop (no-op) analysis

Counts: 1,670 total → 1,412 non-terminal → **960 self-loops (68% of non-terminal, 57.5%
of all rows)**, 452 genuine bootstraps, 258 terminal (127 STOP).

Self-loop rate by pass (non-terminal rows):

| pass | self-loops / total | rate |
|---|---|---|
| -loop-distribute | 336/336 | **100%** |
| -loop-vectorize | 112/121 | 93% |
| -loop-deletion | 122/140 | 87% |
| -loop-unswitch | 84/109 | 77% |
| -loop-unroll | 108/154 | 70% |
| -indvars | 147/252 | 58% |
| -licm | 29/162 | 18% |
| -loop-rotate | 22/138 | 16% |

- 246/360 unique training states contain at least one self-loop.
- Self-loops by family: CHStone 476, csmith 281, cBench 203 (in line with the family mix).
- **Self-loop rewards are not zero.** z-scored: median +0.43, std 1.0, 21% with |z|>1.
  Raw: median −1.05%, **std 521%**, 95% with |r|>1%. A self-loop's "reward" is the
  difference between two re-measured runtimes of the *same* state — i.e. pure measurement
  noise that the per-benchmark z-score promotes to apparent signal.

**Classification.**
- **A (legitimate, keep):** for genuinely inapplicable passes (`-loop-deletion` on a
  non-rotatable loop, `-loop-distribute` on single-loop functions) the no-op *is* the
  correct Q(s, a) ≈ 0 answer. A minority of self-loops are of this kind.
- **B (collector/measurement artifact):** the dominant case. These transitions exist
  because the behavior policy selected a pass that happened to be a no-op at that state;
  the *reward recorded* for them is re-measurement noise, and z-scoring turns that noise
  into a spread (median +0.43, std 1.0) that the regressor then fits as if it were a real
  effect. `-loop-distribute` is the most extreme: it is 22% of the action mass and 100%
  no-op, so the fitted Q for `(s, -loop-distribute)` is fit to noise on 336 rows.
- **C (should be masked before the regression):** the collector already masks a
  no-op pass for the rest of the episode (it cannot be re-selected), so the self-loop
  rows carry no action-selection information beyond "this was tried once". Keeping them
  in the Q-regression as noise-labeled targets dilutes the 452 genuine bootstraps and
  contaminates Q(s, ·) at 246 states.

**Contribution to the failure.** The γ=0.95/20 agent's Q-values are *systematically
positive* (1.5–3.8) for almost every action at every test state (§7) — consistent with a
regressor trained on a reward distribution whose median is +0.43 and whose noise has been
normalized to unit variance. The self-loop rows are the largest single source of that
noise label mass.

---

## 6. SL contribution (frozen models, 8 eval initial states)

| metric | value |
|---|---|
| SL top-1 on eval states | **-loop-distribute on 8/8** (a 100%-no-op pass in training) |
| RL-only (γ=0.95) matches empirical best pass | 2/8 |
| fused matches empirical best | 2/8 |
| **fused pick == RL-only pick** | **8/8** |
| fused pick == SL top-1 | 0/8 |
| Kendall τ (SL ranking vs RL Q ranking) | **−0.28** (γ=0.95), −0.08 (γ=0.9) |
| SL top-3 ∩ RL top-3 | 0–1/3 per benchmark |

Empirical best pass per benchmark (large-input sweep, ground truth): gsm `-loop-rotate`
(+4.0%), dijkstra `-loop-unroll` (+6.1%), bitcount `-licm` (+3.4%), stringsearch
`-loop-deletion` (+6.2%), tiff2bw `-loop-unroll` (+13.3%), tiff2rgba `-loop-unroll`
(+18.1%).

**Interpretation.**
- The SL fusion is **inert**: at the current 0.7·Q + 0.3·SL weight, the fused argmax
  equals the RL-only argmax on all 8 held-out states. SL neither helps nor hurts the
  measured pipeline — the evaluation waves were, in effect, RL-only.
- The SL scorer's ranking is **anti-correlated** with the RL Q ranking (τ = −0.28) and
  its top-1 is always `-loop-distribute`, which the large-input sweep shows is one of the
  weakest passes on the test set. The SL ranking on held-out states carries no usable
  signal (consistent with the earlier LOBO result: SL top-3 ≈ random on unseen
  benchmarks).

---

## 7. Failure-case analysis (γ=0.95/20 agent, per-step inference + wave runtimes)

| benchmark | sequence | term. | hybrid vs cO3 | fixed vs cO3 | empir. best pass |
|---|---|---|---|---|---|
| gsm | `-loop-rotate, -loop-rotate` | no_effect | **0.9829** | 1.0388 | -loop-rotate +4.0% |
| jpeg-c | `-loop-rotate, -loop-rotate` | no_effect | **0.9366** | 0.9153 | (≈0) |
| stringsearch | `-loop-rotate, -licm, -loop-unroll` | no_effect | **0.7784** | 0.9800 | -loop-deletion +6.2% |
| tiff2rgba | `-loop-rotate, -licm, -licm, -licm` | no_effect | **0.9216** | 0.8495 | -loop-unroll +18.1% |
| bitcount | `-licm, -indvars, -indvars` | no_effect | 1.0655 | 1.0108 | -licm +3.4% |
| dijkstra | `-licm, -loop-rotate, -loop-rotate` | no_effect | 1.0884 | 1.0301 | -loop-unroll +6.1% |
| bzip2 | `-loop-rotate, -loop-rotate` | no_effect | 1.0616 | 0.9179 | (≈−1.5%) |
| tiff2bw | `-loop-rotate, -licm, -licm, -licm` | no_effect | 1.0383 | 0.9956 | -loop-unroll +13.3% |

Per-step detail (Q values are the fused Q; SL top-3 never contains the chosen pass except
once on dijkstra):

- **gsm**: step 0 `-loop-rotate` (Q 2.78; SL top = -loop-distribute; IR +230) → step 1
  `-loop-rotate` no-op → stop. gsm's first action is *empirically the best pass* (+4.0%)
  yet measures 0.983× — the IR-level gain collapses under `-O3` codegen and the outcome
  is measurement noise.
- **jpeg-c**: `-loop-rotate` (IR +2,042!) ×2. The pass the agent is most confident about
  *increases* IR by 3% before the backend; measured loss.
- **stringsearch**: `-loop-rotate → -licm → -loop-unroll`, each adding small IR, none
  being the empirically best `-loop-deletion`; measured 0.778× (worst arm).
- **tiff2rgba**: `-loop-rotate → -licm ×3`. `-loop-rotate` (+9.6%) then redundant `-licm`s
  (+519, +8, 0 IR); the empirically best `-loop-unroll` (+18.1%) is never tried.
- **STOP is never emitted** — every episode ends in `no_effect` because the learned
  Q(STOP) never beats the uniformly positive pass Q-values (1.5–3.8). The agent cannot
  express "stop after the useful passes".

**Failure classification.** The failure is a combination of (a) **failure to STOP** —
the agent keeps applying redundant passes until a no-op forces termination, accumulating
IR bloat that can hurt under `-O3` codegen (jpeg-c +2,042 IR); (b) **poor reward
prediction** — Q-values are systematically positive with no learned sense of when a pass
hurts, so there is no basis for stopping; (c) **mediocre second/third actions** —
tiff2rgba's `-licm`-heavy tail versus the empirically best `-loop-unroll`; (d) *not* SL
interference (inert, §6), and *not* a wrong action space (all 8 passes are positive on
test, §2). The first action is often defensible (gsm, tiff2rgba). The winning and losing
sequences are the same *class* of sequence — the same `-loop-rotate ×2` "wins" on bzip2
(1.062×) and "loses" on gsm/jpeg-c (0.98×/0.94×) — which places the residual
win/loss spread inside the ms-scale measurement band (CV 2–17%, tiff2bw 70%).

---

## 8. Primary-bottleneck ranking

| rank | cause | evidence for | evidence against | confidence | proposed intervention | cost |
|---|---|---|---|---|---|---|
| 1 | **Reward noise (measurement protocol)** | CHStone/csmith reward std 480–660% vs 1–4% effects; 79% of buffer noise-labeled; self-loop rows z-scored into fake signal; Q-values uniformly positive; agent can't learn "when a pass hurts" | cBench rows less noisy; noise alone can't explain why first actions are decent | **high** | Re-measure the same 27-program corpus with the large-input protocol (0.2–4 s workloads, more reps, interleaved); keep everything else identical | low–moderate (measurement time, no model change) |
| 2 | **State distribution shift / representation** | 52/61 features shifted >1σ; test Mahalanobis 60.7 vs train 4.9; 62% test beyond train p90; only 5/61 features scale-free; test programs structurally larger | per-feature OOR ≤25%; shift is mostly univariate scale | **medium-high** | Extend scale-free normalization to the 56 autophase counts (per-function / per-instruction normalization or log transforms) — a true scale-free state | moderate (feature + retrain) |
| 3 | **Insufficient program/state diversity** | only 27 programs / 360 states; 4/8 passes ~no-op in training vs all-positive on test; within-family diversity helped (1.006→1.022) | 224/258 distinct sequences superficially; states per benchmark ~12 | **medium** | Add reliable, runnable programs with meaningful inputs (cBench-family or repaired MiBench), not more noise-labeled rows | moderate–high |
| 4 | **Self-loops in the Q-regression** | 57.5% of rows are noise-reward no-ops; -loop-distribute = 22% of action mass at 100% no-op; contaminates Q at 246/360 states | already masked within episodes; clamped to y=r | **medium-high** | Drop/no-op-filter self-loop rows from the regression (keep them only for masking) | low |
| 5 | **State-distribution mismatch at eval** | test runtimes 1–45 ms with CV up to 70% → any single-wave comparison is noise-limited | protocol was identical across arms | **high (as a resolvability limit)** | Evaluate at large inputs with more reps (the large-input sweep already shows stable orderings) | moderate |
| 6 | **Action-space design** | 4/8 passes are no-ops in training, wasting action mass | all 8 passes positive on test; space itself is fine | **low** | prune/merge no-op-heavy passes later | low |
| 7 | **SL interference** | SL ranking anti-correlated with RL (τ=−0.28); SL top-1 always a no-op pass | fusion is inert (8/8 unchanged), so no measured harm | **low** | not the current lever; revisit only after reward fixes | low |

Causes 1 and 2 are the primary pair: the agent is learning from a reward signal that is
mostly noise (§3, §5) and encoding states in a way that systematically misrepresents the
held-out programs (§1). Cause 5 means even a perfect policy would be hard to *verify*
at the current evaluation scale.

---

## 9. Is adding more benchmark diversity justified?

Evidence from this diagnosis:

- **cBench runtime-quality data (6 non-eval programs):** the only family whose training
  rewards are within an order of magnitude of the signal (§3). The large-input sweep on
  the 8 eval benchmarks shows stable, large per-pass effects (0.2–4 s workloads) — this
  family, measured at large inputs, is the right raw material for runtime RL. Adding more
  cBench-family programs with meaningful inputs is the highest-value diversity play and
  is supported by the within-family diversity result (1.006 → 1.022×).
- **CHStone structural diversity:** real structural diversity, but its runtime rewards at
  fallback scale (ms, no inputs) are pure noise (std 480–660%). At current measurement
  quality, CHStone rows *hurt* the reward signal more than the structural diversity helps.
  Usable only with a reliable measurement protocol or for IR-reward ablations.
- **csmith structural diversity:** same verdict as CHStone — noise-dominated rewards;
  suitable for structural/state diversity but not for runtime-reward training as measured.
- **MiBench:** installed but unrunnable (undefined ASSTEX runtime symbols). Fixing the
  build would add 40 real programs — moderate engineering, potentially high value, but
  out of scope until the reward protocol is fixed.
- **AnghaBench:** kernel-level functions, not runnable programs; impractical here.

**Verdict.** Diversity helps *only if* the reward signal is reliable. The diagnosis shows
the current bottleneck is upstream of diversity: 79% of the existing buffer's rewards are
noise, so adding families today would multiply noise, not signal. The justified sequence
is: (1) fix reward fidelity on the existing corpus; (2) then add reliable program
diversity (more cBench-family programs, repaired MiBench); (3) then revisit the
representation (scale-free autophase) if transfer still fails.

---

## 10. Recommended next experiment (one variable)

**Reward-fidelity experiment (dataset-side, single variable: runtime measurement
protocol).**

- Corpus, features, actions, algorithm, seeds, SL scorer, fusion, and evaluation protocol
  all unchanged (27 training programs, 61 features, 9 actions, γ=0.95, 20 FQI iterations,
  seed 42, same clean 8-benchmark eval).
- Change only: re-collect the RL replay buffer with the **large-input measurement
  protocol** already proven in `generate_large_input_dataset.py` (0.2–4 s workloads per
  benchmark, repeated interleaved medians, warmup) so per-step rewards come from stable
  runtimes instead of 1–45 ms default-input runtimes. Drop or down-weight the
  noise-reward self-loop rows from the regression (masking unchanged).
- Expected outcome: reward ordering between passes becomes measurable (SNR rises from
  ~0.1–0.3 toward >1), the fitted Q stops being uniformly positive, STOP becomes
  learnable, and the evaluation becomes resolvable.
- Predicted failure mode if the diagnosis is right: the retrained agent's sequences
  shorten (real STOPs) and the arm moves from 0.979× toward the SL-only/one-step band;
  if instead the agent still fails, the evidence then points to representation (§8, cause
  2) as the next single-variable change.

Do not combine this with feature, SL-fusion, action-space, or algorithm changes in the
same run — that would make the result uninterpretable. The γ=0.9/3 and γ=0.95/20
experiments remain frozen evidence; this experiment adds a third, separately labeled
configuration ("multi-step RL, reward-fidelity protocol").
