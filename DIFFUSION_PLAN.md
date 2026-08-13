# Diffusion for NCCT → CECT: decision plan

Self-contained. Written so this can be picked up in a fresh session, in another
directory, with no prior conversation context.

Companions in this repo: `IMPLEMENTATION.md` (what the code does),
`analysis/BASELINE_REFERENCE.md` (the numbers to beat). The deterministic
roadmap and results live in `../synthetic_CECT/PROJECT_PLAN.md` (and `PLAN_FA.md`,
Persian) — that repo is frozen.

> **Status update.** This document is the plan as written *before* any diffusion
> run existed. Six now do, and all of them plateaued. What that diagnosis found —
> and the changes made in response (offset noise, EMA, SNR loss weighting,
> detail-based checkpoint selection, augmentation) — is recorded in
> `IMPLEMENTATION.md` §2a, §5a and §7a, not here. The plan below is unedited and
> still describes the intended experiment.
>
> Two things in it are now known to need qualification:
>
> - **§7's expectation table is not falsifiable as written.** It predicts a
>   single-sample featHU ~1.41x worse than 13.46 and says "not a failure". The
>   observed val MAE ratio is 1.42. But the *same* number is what a model that
>   simply copies its input would produce, and the measured samples of `diff_v`,
>   `diff_x0` and `diff_v_nocfg` are all closer to the NCCT than the NCCT is to
>   the CECT. Only `raps_hf` and the variance ratio separate those two readings,
>   and **neither has ever been computed for a diffusion run** — `phase_infer/`
>   is empty for all six. That measurement is the gate; nothing in §6's ordering
>   should proceed before it.
> - **The per-voxel metrics cannot rank these models.** MAE/PSNR/SSIM are
>   minimised by the conditional mean, so they score a blurry copy above a
>   correctly-textured sample. Sorting the ten existing runs by val MAE sorts
>   them by how much texture each attempts, in the wrong direction.

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
- **Adversarial branch added 2026-08-12** — a discriminator on the diffusion
  model's one-step x0 estimate, plus the deterministic path. Off by default.
  **See §11 for what it is, the commands, and its stop conditions.**
- ⚠ The adversarial rows recorded in `../synthetic_CECT/analysis/benchmark*` were
  measured with a discriminator that had a BatchNorm real/fake statistics leak, no
  regularisation, and (for `l1_adv` / `pix2pixhd_baseline`) a confounded λ_l1.
  The conditional-D runs there are bimodal — `b0_groupnorm_adv` 26.76,
  `level_aorta` 26.10 and `multiphase_film_adv` 35.01 against a 13.46 baseline,
  while `level_aorta_pv` 14.81 and `level_all8` 14.43 are fine, n=1 each. Do not
  cite those as evidence that adversarial training hurts. Every NON-adversarial
  number, including all of `analysis/BASELINE_REFERENCE.md`, is unaffected — the
  defects are all gated behind `use_adversarial` / `use_feature_matching`.
- ⚠ **A run failed on 2026-07-31 because of the working directory.** Launched from
  `arc_synthetic_CECT/synthetic_CECT`, the relative `data_dir`
  (`../sample_data_reg/...`) does not resolve. Run from the directory whose parent
  holds `sample_data_reg/`, or make `DATA_DIR` absolute in `config.py`. The
  `l1_organ_huprofile` / `l1_huprofile_only` re-runs did **not** happen.
- ⚠ The output rsync has repeatedly created nested copies
  (`out_synthesis_train/out_synthesis_train/`, and once a clone of the repo inside
  itself). Use an **absolute** destination path.

---

## 11. Adversarial branch — what it is and how to run it

Added 2026-08-12. A discriminator loss on the **one-step x0 estimate** of the
diffusion model. Code: `models_disc.py`, `trainer_adv.py`,
`trainer_diffusion.DiffusionTrainer._adversarial_term`, `losses.AdversarialLoss`.
Off by default; a run without `--use_adversarial` is bit-identical to before.

### What it does, in one paragraph

A diffusion training step produces no image, only a regression target at a random
noise level, so there is nothing for a critic to judge. Running the sampler inside
the step would cost 25–1000 U-Net forwards per iteration. Instead the critic is
given the quantity the organ losses already attach to: the closed-form
`x0_hat = predict_x0(out, x_t, t)`, which is free because `out` is already
computed. Per step: one denoiser forward, `x0_hat.detach()` trains D, then D is
frozen and its verdict on the attached `x0_hat` is added to the diffusion MSE.

Four things make that workable and none are optional:

1. **D is conditioned on `t`**, through its own timestep embedding. `x0_hat`'s
   sharpness is a monotone function of `t`, so an unconditional critic's cheapest
   solution is "blurry ⇒ fake" — a timestep regressor — and the only way the
   generator can beat that is by hallucinating detail at high `t`.
2. **Gated to `t < adv_max_t`** (default T/2). Above it `x0_hat` is a genuine
   conditional mean, and "make the mean look like a draw" is a request to be
   overconfident.
3. **Straight-through clamp** on `x0_hat` before D sees it: value bounded to
   [-1,1], gradient passes through. A plain `clamp` would zero the gradient on
   exactly the saturated voxels that are out of range — the mistake already made
   once with the organ losses (see `AUX_MAX_T`).
4. **λ_adv warms up** over `adv_warmup_epochs`. At epoch 1 an `x0_hat` at any
   moderate `t` is visibly not a CT; D wins immediately and its gradient is noise.

### Known cost, stated in advance

`x0_hat` is the posterior **mean** E[x0|x_t], not a sample. Sharpening a mean
spends calibration — and calibration is this repo's headline (§7). Expect
`raps_hf` / `grad_w1` to improve and per-voxel `featHU` / MAE to get *worse*; that
is the term working, not failing. **The number that decides it is `var_ratio`.**
If var_ratio falls relative to the non-adversarial twin, the critic is pulling the
sampler back toward the conditional mean and the term is not worth its cost.

### Commands

Run from the directory whose parent holds `sample_data_reg/` (§10).

```bash
# 0. Correctness gate. ~30 s, no data, no GPU. Run it before burning GPU time.
python tests/test_adversarial.py            # expect: ALL PASS

# 1. The comparison twin MUST exist first — an adversarial run alone says nothing.
./run_scenarios.sh diff_v_organ

# 2. Adversarial, one seed. Start with lambda 0.5, not the 2.0 default.
./run_scenarios.sh diff_v_organ_adv_lam05

# 3. The rest of the lambda sweep, only if step 2 moved something.
./run_scenarios.sh diff_v_organ_adv_lam01 diff_v_organ_adv

# 4. Deterministic-baseline counterpart, if you want the GAN-vs-diffusion contrast
#    on identical code (this is also the clean re-run of synthetic_CECT's
#    b0_groupnorm_adv, whose recorded featHU of 26.76 was measured with a
#    defective discriminator).
./run_scenarios.sh diff_l1_organ_groupnorm diff_l1_organ_groupnorm_adv

# 5. Three seeds, ONLY for a configuration that already moved var_ratio.
SEEDS="42 43 44" ./run_scenarios.sh diff_v_organ_adv_lam05
```

Equivalent direct invocation, if you want to change one knob without editing the
scenario list:

```bash
python train.py \
  --output_dir ../out_synthesis_train/literature_baseline_diff_v_organ_adv_lam05 \
  --seed 42 \
  --use_diffusion --parameterisation v --generator_norm group \
  --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile \
  --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lambda_adv 0.5
```

### While it trains — the two curves that matter

`history.json` / `curves.png`, top-right panel. `train_adv` (dashed) is the raw
generator-side GAN loss, `train_disc` (dashed) is D's. **Read them together:**

| pattern | reading |
|---|---|
| both hovering, neither → 0 | healthy; the term is doing work |
| `train_disc` → 0 within a few epochs | D has won. Lower `lr_disc`, or turn on `--lambda_r1 1.0` |
| `train_adv` → 0, `train_disc` high | G has won; the critic is not constraining anything |
| `train_adv` spikes then plateaus flat | collapse — stop and lower `--lambda_adv` |

Sanity check the scale once, at any epoch: `train_adv × lambda_adv` should be a
**minority** of `train_gen_total`. If it is the majority, λ is too high — that is
the single most likely way this term hurts.

### Evaluating it

```bash
RUN=../out_synthesis_train/literature_baseline_diff_v_organ_adv_lam05

# a. budget check first — needs run_config.json only, not a checkpoint
python infer_volume.py --scenario_dir $RUN --dry_run

# b. sample the test split (8 to explore; 20 for the final coverage number)
python infer_volume.py --scenario_dir $RUN --split test --n_samples 8

# c. pixel + texture metrics, paired against the non-adversarial twin
python benchmark.py \
  --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
  --organ_map orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json \
  --runs_dir ../out_synthesis_train \
  --baseline diff_v_organ \
  --out analysis/benchmark_adv

# d. THE headline: variance ratio and per-organ beta
python scripts/audit_enhancement.py \
  --manifest $RUN/phase_infer/manifest.csv \
  --out analysis/enhancement_diff_v_organ_adv.json

# e. calibration — the cost side of the trade
python scripts/calibration_eval.py --mode ensemble --dir $RUN/phase_infer
```

### Stop conditions for this branch specifically

- **`var_ratio` drops vs `diff_v_organ`** → the critic is collapsing the posterior.
  Report it and stop; do not chase it with more λ.
- **`raps_hf` does not move toward 1.0 at any λ** → the term is buying nothing that
  the diffusion objective was not already buying. Drop it.
- **featHU degrades by more than the 0.84 HU gate *and* var_ratio is flat** → strictly
  worse on both axes. Stop.
- **`train_disc` → 0 in under 5 epochs at every λ** → the critic is overfitting the
  ~137-pair training set. Try `--lambda_r1 1.0 --adv_mode hinge` once; if that does
  not fix it, the dataset is too small for an adversarial term and that is the
  finding.

### Knobs

| flag | default | note |
|---|---|---|
| `--use_adversarial` | off | the switch |
| `--lambda_adv` | 2.0 | **sweep down.** Scaled for the GAN baseline, not for a diffusion MSE |
| `--adv_warmup_epochs` | 10 | linear ramp of λ_adv |
| `--adv_max_t` | T/2 | timestep gate; below it `x0_hat` is near-deterministic |
| `--adv_mode` | lsgan | `hinge` degrades more gracefully when D is winning |
| `--use_cond_disc` | off | D sees `cat([NCCT, image])` — pix2pix's D(x,y). Worth having on |
| `--lr_disc` | 1e-4 | halve it if D wins too fast |
| `--lambda_r1` | 0 | lazy R1 on real samples; try 1.0 if `train_disc` collapses |
| `--disc_norm` | group | `batch` reproduces the reference D's real/fake statistics leak |
| `--adv_clip_mode` | straight_through | `hard` / `none` exist to be ablated against |
| `--use_feature_matching` | off | pix2pixHD L1 on D's intermediate features |
