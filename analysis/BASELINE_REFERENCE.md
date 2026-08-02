# Baseline reference — the numbers this repo is measured against

Copied out of `synthetic_CECT/analysis/` when this fork was made, because the rest
of that directory (15 runs' worth of one-off comparisons) was dropped and every
claim in `DIFFUSION_PLAN.md` §7 is relative to these values.

**These are results of record. Do not recompute them here — this repo has no
deterministic-baseline runs of its own.** If a number below has to be re-derived,
it comes from `../synthetic_CECT`, on the same frozen split.

---

## 1. The model to beat — `l1_organ_groupnorm`, 3 seeds

`--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay
--generator_norm group`, 20 test cases, HU window [-200, 400].

| seed | featHU | PSNR | SSIM | oPSNR | oSSIM | oMAE | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s42 | 13.88 | 30.57 | 0.9486 | 24.77 | 0.9698 | 0.0288 | 0.841 | 0.0021 | 0.0065 | 1.356 | 0.897 | 1.099 |
| s43 | 12.90 | 30.66 | 0.9492 | 24.80 | 0.9698 | 0.0286 | 0.842 | 0.0021 | 0.0066 | 1.353 | 0.893 | 1.093 |
| s44 | 13.60 | 30.67 | 0.9493 | 24.77 | 0.9698 | 0.0288 | 0.841 | 0.0021 | 0.0067 | 1.354 | 0.893 | 1.095 |
| **mean** | **13.46** | 30.63 | 0.9490 | 24.78 | 0.9698 | 0.0287 | 0.841 | 0.0021 | 0.0066 | 1.354 | 0.894 | 1.096 |

`RAPS`, `seam`, `zflick`, `zaniso` are **ratios scored as distance from 1.0**, not
higher-is-better. RAPS 0.841 = 16% of the ground truth's high-frequency amplitude
is missing (blur). seam 1.354 = tile boundaries are still visible; group norm did
**not** fix it (instance-norm predecessor was 1.346, marginally better).

### Noise floor

**3-seed 2σ gate on featHU ≈ 0.84 HU.** Nothing smaller is a result. This has
already killed one headline claim in this project, so it is enforced.

### Predecessor, for the gap

`l1_organ_curriculum` (instance norm), 3 seeds: featHU 14.85 / 14.39 / 14.26,
mean **14.50**. Group norm bought −1.04 HU, above the gate.

---

## 2. The deficiency diffusion is chosen to fix

`scripts/audit_enhancement.py` on the 20 test cases (run against
`l1_organ_curriculum`; the generator emits essentially the same levels under group
norm).

| statistic | value |
|---|---|
| median β, all organs | 0.217 |
| **median β, contrast organs** | **0.181** |
| **median var(gen)/var(real)** | **0.176** |

β = slope of generated organ-median HU regressed on the real one across cases.
β = 1 tracks each case's true level; β = 0 emits a constant. 0.18 means the model
recovers 18% of case-to-case level variation. var ratio 0.176 = **5.7×
under-dispersed**.

Per organ:

| organ | β | var ratio | real sd (HU) | gen sd (HU) | bias (HU) |
|---|---|---|---|---|---|
| aorta | 0.235 | 0.122 | 21.1 | 7.4 | −4.4 |
| portal_vein_and_splenic_vein | 0.199 | 0.156 | 18.7 | 7.4 | −36.1 |
| inferior_vena_cava | 0.160 | 0.108 | 20.6 | 6.8 | −4.7 |
| heart | 0.163 | 0.476 | 17.5 | 12.1 | +1.9 |
| liver | 0.320 | 0.195 | 11.9 | 5.3 | −1.0 |
| pancreas | 0.117 | 0.426 | 10.5 | 6.8 | −0.2 |
| gallbladder | 0.695 | 1.775 | 11.3 | 15.1 | +0.9 |
| colon *(negative control)* | 0.257 | 0.092 | 199.3 | 60.4 | +45.3 |

**Targets for this repo: var ratio → 1.0, 90% interval coverage → 0.90.**

### Per-case error is the level miss

featHU correlated against each organ's true median HU, across the 20 test cases:

| organ | r |
|---|---|
| aorta | **+0.765** |
| portal_vein_and_splenic_vein | +0.765 |
| inferior_vena_cava | +0.696 |
| heart | +0.695 |
| liver | +0.604 |
| pancreas | +0.512 |
| gallbladder | +0.242 |
| colon | +0.127 |

r = +0.765 on the aorta ⇒ **59% of per-case featHU variance *is* the level miss.**
|r| < 0.44 is not significant at p < 0.05 with n = 20.

### And the level is not predictable

`../synthetic_CECT/scripts/probe_level_predictability.py`: **R² predicting aortic CECT HU from the
NCCT's own organ medians = −0.105 held out** — worse than predicting a constant
(held-out MAE 19.4 vs 18.9 HU) against a true spread of 96–218 HU, **sd 24.4 HU**.

That is the direct measurement, not an inference by elimination, and it is why
this repo exists.

---

## 3. What was already ruled out

Do not re-litigate these; each is a measurement in `synthetic_CECT/analysis/`.

| ruled out | evidence |
|---|---|
| overfitting | train 8.32 vs val 8.86 HU — a 6% gap |
| registration error | \|r\| ≤ 0.25; the two worst-synthesised cases are *better*-registered than median |
| voxel-scale inconsistency | all 274 volumes exactly 1.5 mm isotropic |
| model capacity | 9× parameters (3.3 M → 30 M) bought 0.33 HU |
| data volume | learning-curve asymptote ~8.74 HU vs 8.86 now |

---

## 4. Expected results for this repo, stated before running

From `DIFFUSION_PLAN.md` §3 and §7:

| quantity | expectation |
|---|---|
| featHU, single DDIM sample | **~1.41× worse than 13.46**, i.e. ~19 HU. Not a failure. |
| featHU, mean of N samples | ≈ 13.46, confirming collapse to the conditional mean |
| **var(gen)/var(real)** | **the headline**: 0.176 → 1.0 |
| **90% coverage** | **0.90**; deviation = miscalibration |
| RAPS / gradW1 | should *improve* vs 0.841 / 0.0021 — diffusion is not L1 |
