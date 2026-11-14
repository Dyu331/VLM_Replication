# Steering, redesigned (v4 follow-up 2)

Replaces the first steering run (`directions.py`, `steer.py`, `steer_bagel.py`, 2026-11-07). Design agreed with the user
2026-11-12. Everything below is fixed **before** any new result exists. A change after results come in is
written into the change log at the end, with the reason.

Visual explainer: https://claude.ai/artifact/E4nXySXkhSfLRkUKcnQcnv

## Why redo it

The first run found that a push along B's direction with its "say left" part removed makes Qwen and BAGEL give B's
answer at 16–32× B's gap. That result cannot tell apart two readings:

- the later layers act on B's content, or
- the push only overlaps, by accident, the general channel that moves any answer ("leak").

Removing "say left" does not settle it. "Say left" is a correlational direction (mean state when the model says left
minus right). The models say A's answer 82–90% of the time, so it is mostly A's direction. Removing one such vector
leaves an unknown amount that still moves the answer. Isotropic random directions are a weak control: they point where
states never go, likely why they break Gemma.

## Two questions, kept apart

1. **Natural use.** At its natural size, does B's information move the answer?
2. **Capacity.** If B's content is amplified, do the later layers act on it, beyond the generic answer channel?

## Definitions

- **Lean**: log-odds of "left" against "right" for the first answer word, LSE(left, Left) − LSE(right, Right).
- **g** (per layer L): the gradient of the final lean with respect to the hidden state at layer L, at the token before
  the answer, averaged over training questions; unit length. The direction in which a small push at L moves the final
  answer most.
- **G** (per layer L): the top principal components of the per-question gradients, enough for 90% of their variance,
  **capped at 10** to start. The cap is revisited after the per-question check (below), not before.
- **B's direction**: difference of means within A's answer, averaged (as before):
  ½[(LL − LR) + (RL − RR)], first letter A's answer, second B's answer; training rooms only.
- **A's direction**: the same with the roles swapped (split by B's answer).
- **B ⊥ G**: B's direction with its components along G removed, then unit length.
- **Random ⊥ G**: drawn from the training states' covariance (data-shaped), G removed, unit length. 10 seeds.
  If one or two dimensions dominate the variance (outlier dimensions), they are set aside before sampling and this
  is reported.

## Phase 0: freeze this plan (done 2026-11-12)

## Phase 1: directions (CPU, from saved training states)

1. **Reliability**: split the 25 training rooms into two halves (fixed seed), build B's and A's directions in each
   half, cosine between halves, every layer.
2. **Decide on denoising from reliability**:
   - split-half cosine ≥ 0.9 at the headline layer and its neighbours: keep the raw direction;
   - lower: rebuild the direction inside the top 256 principal components of the training states (as the readers
     use), and re-measure reliability.
3. **Gap check**: plain gap |d| against the split-half estimate √(d₁·d₂). Within 10%: use the plain gap and report the
   check. Further apart: use the split-half gap.
4. **Outlier dimensions**: share of the training-state variance in the top 1–2 dimensions; decides the random draw.
5. **Headline layer**, chosen on training rooms only: the **earliest layer whose cross-validated B-reader score is
   within 0.01 of the maximum** (the cross-validated curve is flat near its top, so its argmax is decided by noise;
   the earliest plateau layer leaves the most layers after it). From `analysis.json`: Qwen L25, Gemma L38, BAGEL L21
   (test-set peaks were L29, L43, L23).
6. **Quality report**: balanced accuracy, within A's answer, of projecting test questions onto B's direction. Expected
   below the reader (the reader also discounts unrelated variation); near 50% is a problem.

**Gate 1**: B's direction is reliable at the headline layer and its neighbours (split-half cosine ≥ 0.9, after
denoising if needed). Otherwise stop and rethink before any GPU work.

## Phase 2: gradients (GPU)

1. One forward and backward pass per question gives the gradient of the final lean with respect to the hidden state
   at every layer at once (gradients with respect to activations only, no weight gradients).
2. **Finite-difference check** on a few questions: a small push along g moves the lean by what g predicts.
3. Run on training questions (to build g and G) and test questions (for the per-question check). Qwen, then Gemma.
4. **BAGEL**: regular gradients through its packed forward pass; if that does not work, estimate the gradient by finite
   differences along the top principal components.

**Gate 2**:
- the finite-difference check passes;
- for each test question, using its own gradient, the predicted first-order effect of a push along B ⊥ G and along
  random ⊥ G is near zero (small next to the effect of the same push along g). If not, raise k and record it.

## Phase 3: Part 1, natural use (CPU)

- B's overlap with g, against 1,000 data-shaped random directions (the null distribution), with A's direction as the
  reference for information the model uses.
- Predicted effect of a 1× push: B's gap × overlap with g.
- **Decision rule**: B is "used at natural size" only if its overlap with g is above the 95th percentile of the null.

## Phase 4: Part 2, capacity (GPU)

- **Directions**, per pushed layer: B ⊥ G (the test), random ⊥ G ×10 (the null), g (positive control), B full
  (links to the first run).
- **Pushes**: 1, 4, 8, 16, 32× B's gap, the same length for every direction, toward each question's B answer, at
  the token before the answer, one layer at a time. Also reported as a share of the hidden state's length.
- **Layers**: the headline layer, plus every second layer for the layer-by-layer figure.
- **Questions**: all test questions.
- **Smoke checks** on 8 questions before the full run:
  - the cached step matches the full forward pass;
  - B ⊥ G and random ⊥ G move the lean by about zero at 1×;
  - the real 1× push along B matches Part 1's prediction.
- **First-order check at large pushes**, headline layer only: recompute g at an 8× pushed state on a subset of
  questions and report its cosine with g.
- **Decision rule**: B's content is "acted on" only if, at the headline layer, B ⊥ G moves the lean more than every
  one of the 10 random seeds at 16× and at 32×, its 95% interval clears the random mean, and the cell is under 10% broken.

## Measures

- **Primary**: shift in the lean toward the target word, averaged over all test questions, with photo-bootstrap
  intervals (1,000 resamples). Flip and no-flip questions are shown separately if their patterns differ.
- **Secondary**: flip rate on flip questions (share giving B's answer).
- **Guardrail**: broken rate (less than half the probability on left/Left/right/Right). Cells over 10% broken are
  flagged and not interpreted.

## Outcomes

| Part 1 | Part 2 | Conclusion |
|---|---|---|
| Like random | Like random | Computed but inert: the later layers ignore it even amplified |
| Like random | Above random | Present but suppressed: acted on only when strong |
| Above random | either | Used at natural size but outweighed; to reconcile with the lens result |

## Order

Qwen end to end first, then Gemma, then BAGEL.

## Files

New outputs go to `$V4_RESULTS/<model>/steer2/`, so the first run's files (used by the current report and by
`lens.py`) stay as they are until the report is rewritten.

| File | Role |
|---|---|
| `repro/v4/steer2_dirs.py` | CPU. `reliability` (Phase 1), then `build` after gradients (G, B ⊥ G, random ⊥ G, Part 1) |
| `repro/v4/grads.py` | GPU. Gradients of the final lean, Qwen and Gemma |
| `repro/v4/grads_bagel.py` | GPU. The same for BAGEL (finite differences if autograd fails) |
| `repro/v4/steer2.py`, `steer2_bagel.py` | GPU. Part 2 pushes; reuse the cached-prefix step from `steer.py` / `steer_bagel.py` |
| `repro/v4/steer2_common.py` | numpy only: layers, deltas, the lean measure, summaries, bootstrap |
| `slurm/v4_steer2_dirs.sbatch`, `v4_grads.sbatch`, `v4_grads_bagel.sbatch`, `v4_steer2.sbatch`, `v4_steer2_bagel.sbatch` | jobs |

The first run's scripts are removed once the report no longer reads their outputs (git keeps them).

## Change log

- 2026-11-12, after Phase 1 (`steer2/reliability.json`):
  - **Gate 1 corrected.** The split-half cosine compares two directions each built from half the training data, so it
    understates how reliable the full-data direction is. The gate now uses the estimated cosine of the full direction
    with the true signal, √(2ρ/(1+ρ)) for split-half cosine ρ, which must be ≥ 0.9. At the headline layers:
    Qwen 0.94 (ρ 0.80), Gemma 0.91 (ρ 0.70), BAGEL 0.91 (ρ 0.71): pass. The estimate agrees with the measured gap
    inflation (plain ÷ split-half gap 1.06 / 1.11 / 1.10).
  - **No denoising.** The direction inside the top 256 principal components matches the raw one to ±0.01 at every
    layer: the noise lies in the same high-variance directions, so this cannot remove it. Raw directions are used.
  - **Outlier rule made concrete.** Where the top two dimensions of the training states hold more than 20% of the
    variance (Gemma L14–21, BAGEL L2–4 and L28; at most 5% at the headline layers), those dimensions are left out
    when drawing the data-shaped random directions only. B, A, g, G and B ⊥ G keep every dimension.
  - **Push unit: the plain gap** |d|, not the split-half estimate (user's call: the 6–11% overstatement is small next
    to the push sizes). Reported alongside the split-half gap.
  - **Other positions (natural use only).** Phase 2 also keeps the gradients at the last mention of X and of Y (one
    backward pass gives every position), so Part 1 runs there as well, from saved states, with no extra GPU.
    Steering at those positions is left as a later extension.
- 2026-11-12, after Phase 2:
  - **Precision.** The lean is scored in float32 from the final hidden state (bf16 logits round it to steps of
    0.125–0.25). Real pushes are added in bf16, which leaves a noise floor of about 0.02 in the lean per question.
  - **g and G from the realistic part of the gradients.** Only 25% (Qwen L25, BAGEL L21) and about 0% (Gemma L38)
    of each gradient lies in the directions along which the hidden states vary (99% of their variance, about 350–410
    directions); the rest points where no question's state ever goes. B and the random controls only point where
    states go, so only that part can act on them. Each gradient is first projected onto the top principal directions
    of the training states holding 99% of their variance; g and G are built from the projected gradients. Robustness:
    the same at 95% and 99.9%.
  - **Two more directions, with G removed.** Depth rank (farthest minus middle, averaged over the four A × B cells):
    structured but unrelated to left/right, so a positive Part 2 result for B must beat it too (specificity).
    A's direction: a reference only (the model uses A linearly, so that use lies inside G).
  - **Decision rule for Part 2 extended:** B ⊥ G must also beat depth rank ⊥ G. A null result is reported as
    "no route found at this token", not as proof that the information is inert.
- 2026-11-12, Phase 3 (Part 1):
  - The overlap test is two-sided (|B · g| against |random · g|): a negative overlap would also be a route, toward
    the other word. The sign is reported.
  - Flip and no-flip questions differed in the predicted shift, so their difference got its own null from the same
    1,000 random directions (added after seeing that they differ; the plan said to show them separately if so).
- 2026-11-14, Phase 4 (Part 2) results (`steer2/steer.json`, copies in `local_data/v4/results_steer2/`):
  - Fixes during the run: Gemma's 76-row batch ran out of memory (now 2 chunks of 38); BAGEL's scores were computed
    in bf16 under autocast (now float32). BAGEL's Phase 2 lean was also bf16 under autocast; its gradients are exact
    for that lean and the finite-difference checks used the same function, so Phases 2–3 stand.
  - Decision rule (headline layer, 16x and 32x): **fails in all three models.** Gemma passes at 32x only (+0.64 logits,
    above all 10 random seeds), not at 16x; BAGEL nothing.
  - Qwen, not passing but noted: B ⊥ G moves the lean toward B's answer at L24–26 (+2.3 to +2.9 at 16x, +3.6 to +6.6
    at 32x, same on flip and no-flip), ranking 8–9 of 10 random seeds; data-shaped random directions swing the lean by
    ±2–4 (16x) and ±4–5 (32x) in both directions, so B ⊥ G sits inside their spread.
