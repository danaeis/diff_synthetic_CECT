# Diffusion for NCCT → CECT: decision plan

Self-contained. Written so this can be picked up in a fresh session, in another
directory, with no prior conversation context.

Companions in this repo: `IMPLEMENTATION.md` (what the code does),
`analysis/BASELINE_REFERENCE.md` (the numbers to beat). The deterministic
roadmap and results live in `../synthetic_CECT/PROJECT_PLAN.md` (and `PLAN_FA.md`,
Persian) — that repo is frozen.

---

## 1. The task, in one screen

Synthesise a contrast-enhanced CT (**CECT**, venous phase) from a non-contrast CT
(**NCCT**) of the same patient.

| | |
|---|---|
| data | 137 registered NCCT/venous pairs (+136 arterial), VinDr |
| split | **97 train / 20 val / 20 test**, case-level, frozen in `splits/split.json` |
| geometry | **1.5 × 1.5 × 1.5 mm isotropic, every volume** (measured, not assumed) |
| model | 2-D U-Net, 13.3 M params, 128² patches, HU window `[-200, 400] → [0,1]` |
| current best | `l1_organ_groupnorm`, **featHU 13.46 HU** at 3 seeds |

**Headline metric `featHU`** (`feature_l1_hu`): mean per-organ |median-HU error|
over 16 organs, computed by an external XGBoost phase model
(`orgFeatXGB_CTPhase/phase_eval.py`). Supporting metrics: `org_mae`, and the
texture set `raps_hf` / `grad_w1` / `seam` / `zflicker` (all **ratios scored as
distance from 1.0**, not higher-is-better — see `metrics.py`).

**Noise floor**: 3-seed 2σ gate on featHU is **≈0.84 HU**. Nothing smaller is a
result. This has already killed one headline claim in this project, so it is
enforced.

---

## 2. What is measured, and why it decides the diffusion design

Five independent measurements, all in `analysis/`:

| ruled out | evidence |
|---|---|
| overfitting | train 8.32 vs val 8.86 HU — a 6% gap |
| registration error | \|r\| ≤ 0.25; the two worst-synthesised cases are *better*-registered than median |
| voxel-scale inconsistency | all 274 volumes exactly 1.5 mm isotropic |
| model capacity | 9× parameters (3.3 M → 30 M) bought 0.33 HU |
| data volume | learning curve saturating; extrapolated asymptote ~8.74 HU vs 8.86 now |

### The finding that governs everything below

`scripts/audit_enhancement.py` and
`../synthetic_CECT/scripts/probe_level_predictability.py`:

| statistic | value | meaning |
|---|---|---|
| β (level tracking, contrast organs) | **0.18** | recovers 18% of case-to-case level variation |
| var(gen)/var(real) | **0.176** | output is **5.7× under-dispersed** |
| featHU vs true aortic HU | **r = +0.765** | 59% of per-case error *is* the level miss |
| **R² predicting aortic CECT HU from NCCT** | **−0.105 (held-out)** | **not predictable at all** |

That last row is the decisive one and it is a *direct* measurement, not an
inference by elimination. Predicting aortic level from the NCCT's own organ
medians is **worse than predicting a constant** (held-out MAE 19.4 vs 18.9 HU),
against a true spread of **96–218 HU, sd 24.4 HU**.

**The residual is aleatoric.** Absolute enhancement depends on injection dose,
bolus timing and cardiac output; none is observable in a non-contrast scan. Write:

```
CECT = f(NCCT, Z)      Z = dose, timing, physiology — unobserved
```

The best any deterministic model can do is `E[CECT | NCCT]`; the irreducible error
is `E[Var(CECT | NCCT)]`. The measurements say the model is **already at that
ceiling**.

---

## 3. The reframing: what diffusion is and is not for here

Diffusion models `p(y|x)` rather than `E[y|x]`. Given an aleatoric residual, that
is conceptually the right tool. But be precise about the consequence, because it
is counter-intuitive and it decides the whole evaluation protocol:

> **A single diffusion sample is expected to be ≈√2 ≈ 1.41× WORSE than the current
> model on featHU.**

For a Gaussian conditional with spread `s`, the mean predictor has
`MAE = 0.798·s`, while one honest draw has `MAE = 1.128·s`. Sampling the true
distribution *costs* you on a point metric — that is not a bug in diffusion, it is
what modelling uncertainty honestly costs when scored by a point metric.

So:

| use of diffusion | verdict |
|---|---|
| beat featHU with one sample | **will not happen** — expect ~1.41× worse |
| average N samples | returns to the conditional mean = where you already are |
| **report mean + calibrated spread** | **the actual contribution** |

### The real target: calibration

You now have something rare — **the ground-truth conditional spread**: aortic level
sd **24.4 HU** across cases, and a measured 5.7× under-dispersion in the current
model. That makes a sharp, falsifiable claim available:

> Does the model's predicted spread match the true conditional spread?

Metrics for that (new, to be added — see §6): predicted-vs-true `var` ratio (target
1.0, currently 0.176), continuous ranked probability score (CRPS), and coverage of
the 90% predictive interval (target 0.90).

**This is the thesis contribution, not a featHU win.** "We show the residual is
irreducible, and we build a model that quantifies it correctly" is stronger and
more defensible than another 0.3 HU.

⚠ **Cheaper alternative that must be beaten to justify diffusion at all.**
Heteroscedastic regression — predict μ and σ per voxel, train with Gaussian NLL —
delivers calibrated uncertainty for **~1% of the compute**, and reuses the current
generator with a 2-channel output head. **Run this first as the baseline.** If
diffusion cannot beat a heteroscedastic U-Net on CRPS and coverage, it is not
worth the cost.

---

## 4. Gate D1 — the VAE fidelity test (~1 hour, run before anything else)

Stable Diffusion operates in a VAE latent space: 8× spatial downsampling, 4 latent
channels, trained on 8-bit RGB natural images. Round-trip your **real** CECTs
through that VAE — no diffusion, no training — and score the reconstruction with
the normal benchmark suite.

```python
# sketch: scripts/vae_fidelity.py
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# per slice: HU -> [-1,1] via the [-200,400] window -> replicate to 3ch
# -> vae.encode(...).latent_dist.mode() -> vae.decode(...) -> back to HU
# reassemble the volume, write a manifest, run phase_eval.py on it
```

**This is a ceiling measurement**: no latent-diffusion method can beat its own
autoencoder. Score the whole suite, not just featHU — the two are likely to
diverge:

| metric | prediction | why |
|---|---|---|
| `featHU`, `org_mae` | may survive | organ **median** HU is a low-frequency statistic and VAEs preserve low frequencies well |
| `raps_hf`, `grad_w1` | likely fails | fine texture at 8× downsampling in a domain the VAE never saw |

**Decision rule**
- VAE featHU **> 13.5 HU** (current best) → every SD/ControlNet route is dead on
  arrival. Go pixel-space.
- VAE featHU **< ~8 HU** but texture metrics ruined → latent diffusion is viable
  *only* if you are optimising HU fidelity and not texture. Weigh against §5.
- Both good → SD fine-tuning becomes worth considering, and §5's recommendation
  changes.

Report this number either way. "We measured the autoencoder ceiling before
committing" is a good methods paragraph.

---

## 5. Recommendation: pixel-space conditional diffusion, from scratch

Assuming D1 fails as expected, and independently of it for these reasons:

**1. The domain gap is large and the win is small.** SD is RGB natural images at
512²; you have single-channel CT over a 600 HU window at 128². Pretrained priors
help most when they transfer texture statistics — yours are CT-specific, and the
one thing you need (HU level) is exactly what the prior cannot know.

**2. You already have most of the model.** A diffusion U-Net is your U-Net plus
timestep conditioning — and **timestep conditioning is FiLM**, which
`models.py` already implements (`class FiLM`, used for phase/level). Adding `t` is
the same code path:

```python
# the existing FiLM already does y = x*(1+γ(c)) + β(c)
# for diffusion, c = [timestep_embedding(t)] (+ phase/level, if kept)
```
The rank-agnostic broadcast and the zero-init/RNG-safety guarantees in
`tests/test_phase_cond.py` carry over unchanged.

**3. Conditioning on the NCCT is concatenation.** `UNetGenerator(in_channels=…)`
already exists (added for 2.5-D). Conditional diffusion feeds
`cat([noisy_target, ncct])` → `in_channels=2` (or `1 + n_input_slices` under
2.5-D). No new plumbing.

**4. Your losses transfer.** Auxiliary losses apply to the predicted `x0` (the
denoised estimate) at each step, so `OrganWeightedLoss` and `OrganHUProfileLoss`
from `losses.py` work directly.

### Architecture

| choice | value | rationale |
|---|---|---|
| space | pixel, 128² | matches the current patch geometry and the frozen split |
| conditioning | channel-concat NCCT + FiLM(t) | reuses existing code |
| parameterisation | **predict `v` or `x0`, not `ε`** | at high noise levels `ε`-prediction is poorly conditioned for a task whose signal is a *level*; `x0` also lets the organ losses attach directly |
| schedule | cosine, 1000 train steps | standard; sample with DDIM at 50–100 steps |
| sampler | DDIM (deterministic) + ancestral (stochastic) | DDIM for the point estimate, ancestral for the N-sample spread |

### Honest cost

Training is roughly comparable to the current model per epoch, but **sampling is
50–100 forward passes per patch**, and volume inference tiles ~49 patches per
slice. Budget: a full test-set inference goes from minutes to hours. With N=20
samples for uncertainty, multiply again. Plan for this before starting.

---

## 6. Implementation steps

Ordered so each step is verifiable before the next.

1. **`scripts/vae_fidelity.py`** — Gate D1 (§4). Reuse `infer_volume.py`'s manifest
   writing so `phase_eval.py` and `benchmark.py` score it like any other model.
2. **Heteroscedastic baseline** — 2-channel output head (μ, log σ²) on the current
   generator + Gaussian NLL in `losses.py`. This is the bar diffusion must clear.
   Cheap, and a complete result on its own.
3. **New calibration metrics in `metrics.py`** — CRPS, 90% interval coverage,
   predicted-vs-true variance ratio. Needed by step 2, reused by step 5.
   *Do this before the diffusion model*: without it there is no way to tell whether
   diffusion helped.
4. **`models_diffusion.py`** — new file; **do not modify `models.py`**, so the
   baseline stays reproducible (the same rule used for the attention work).
   Timestep embedding → existing `FiLM`. Noise schedule + DDIM sampler.
5. **`trainer_diffusion.py`** or a `diffusion` branch in `trainer.py` — the loss is
   an MSE on the parameterisation target, plus optional organ losses on `x0`.
6. **Sampling path in `infer_volume.py`** — `--n_samples N`, emit mean and per-voxel
   std. The tiling/blending code is unchanged; only the per-patch call differs.
7. **Scenario in `run_scenarios.sh`** following the existing format.

### Non-negotiables inherited from this project

- **Frozen split.** Use `splits/split.json`. Do not re-split; every number in
  `analysis/` depends on it.
- **3 seeds before any claim.** featHU 2σ gate ≈0.84 HU. Sampling adds its own
  variance on top of seed variance — budget for both, and fix the sampling seed
  separately so the two are separable.
- **`run_config.json` diff.** Each run should differ from its predecessor only in
  the keys you intended.
- **Never delete `best_model.pth`.** Six earlier runs are permanently
  un-re-benchmarkable because their checkpoints were removed.

---

## 7. Evaluation protocol

Diffusion cannot be scored the same way as a deterministic model. Report:

| quantity | how | target |
|---|---|---|
| featHU, single DDIM sample | existing `benchmark.py` | expect ~1.41× worse than 13.46 — say so **in advance** |
| featHU, mean of N=20 | existing `benchmark.py` | ≈ current model, confirming it collapses to the conditional mean |
| **var(gen)/var(real)** | `scripts/audit_enhancement.py` | **1.0** (currently 0.176) — the headline |
| **CRPS**, per organ | new (step 3) | lower is better; compare to heteroscedastic baseline |
| **90% interval coverage** | new (step 3) | **0.90**; deviation = miscalibration |
| texture: `raps_hf`, `grad_w1` | existing | diffusion should improve these vs L1 |

The `var` ratio is the one to lead with. It is currently **0.176 with a known
target of 1.0**, it is directly measured, and it is exactly the deficiency
diffusion is theoretically supposed to fix.

---

## 8. Stop conditions

State these now, before any results exist:

- **D1 shows the VAE ruins featHU** → do not pursue Stable Diffusion / ControlNet.
- **Heteroscedastic baseline achieves good calibration** → diffusion must beat it
  on CRPS *and* coverage, or it is not worth 50–100× the sampling cost. Report the
  comparison either way.
- **var ratio does not move meaningfully above 0.176** → diffusion failed at the
  one thing it was chosen for. That is a publishable negative result given how
  carefully the ceiling was established.
- **Single-sample featHU is worse *and* calibration is no better** → stop, and
  report the aleatoric ceiling as the finding.

Expected honest outcome given everything measured: **comparable or slightly worse
featHU, better texture, and — if it works — a correctly calibrated uncertainty
estimate that no deterministic model can provide.** Diffusion changes how `f` is
approximated; §2 says the bottleneck is not the approximation of `f`. Go in
expecting an uncertainty contribution, not an accuracy one.

---

## 9. Reference points

Worth reading before implementing; verify details against the papers rather than
this summary.

- **FiLM** — Perez et al., 2018. The conditioning mechanism already in `models.py`;
  timestep conditioning is the same idea.
- **Aleatoric vs epistemic uncertainty** — Kendall & Gal, 2017. The decomposition
  §2 rests on, and the source of the heteroscedastic-NLL baseline in step 2.
- **Palette / SR3** — Saharia et al. Image-to-image diffusion in pixel space with
  channel-concatenation conditioning; the closest match to the recommendation in §5.
- **DDIM** — Song et al. Deterministic fast sampling; the point-estimate sampler.
- **Classifier-free guidance** — Ho & Salimans. Relevant if phase/level
  conditioning is kept, and it gives a knob trading diversity against fidelity —
  i.e. an explicit dial on the var-ratio-vs-featHU trade-off in §7.
- **Latent Diffusion / Stable Diffusion** — Rombach et al. Only relevant if D1 passes.

---

## 10. Current repo state (2026-07-31)

Things a fresh session should know:

- `l1_organ_groupnorm` at 3 seeds is the best model: **featHU 13.46** (−1.04 vs the
  previous baseline, above the 0.84 gate). It did **not** fix `seam` (1.354 vs
  1.346, slightly worse) — `../synthetic_CECT/scripts/erf.py` shows group norm's gradient support is
  still patch-unbounded; only `batch` norm is bounded.
- Level conditioning, 2.5-D input and a conditional discriminator are **implemented
  and tested** but not yet run at scale. Scenarios: `b0_groupnorm_adv`,
  `level_aorta`, `level_aorta_pv`, `level_all8`, `slices5_k2`, `slices11_k5`.
- `splits/levels.json` exists (per-case oracle organ levels, 137 pairs).
- ⚠ **A run failed on 2026-07-31 because of the working directory.** Launched from
  `arc_synthetic_CECT/synthetic_CECT`, the relative `data_dir`
  (`../sample_data_reg/...`) does not resolve. Run from the directory whose parent
  holds `sample_data_reg/`, or make `DATA_DIR` absolute in `config.py`. The
  `l1_organ_huprofile` / `l1_huprofile_only` re-runs did **not** happen.
- ⚠ The output rsync has repeatedly created nested copies
  (`out_synthesis_train/out_synthesis_train/`, and once a clone of the repo inside
  itself). Use an **absolute** destination path.
