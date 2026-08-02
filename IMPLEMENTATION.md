# Implementation reference — diffusion fork

What the code **actually does** right now, not the design intent. A snapshot —
re-verify against the code before trusting a claim here.

This repo is a fork of `../synthetic_CECT` for the work in `DIFFUSION_PLAN.md`:
model `p(CECT | NCCT)` instead of `E[CECT | NCCT]`, and measure whether the
predicted spread is the true conditional spread. The deterministic repo is frozen
and still holds every result recorded in `analysis/BASELINE_REFERENCE.md`.

**Nothing here has been run on real data yet.** Everything below is verified only
by `tests/`, which run on random tensors on CPU.

---

## 0. Layout

```
train.py trainer.py trainer_diffusion.py     training loops
dataset.py models.py models_hetero.py models_diffusion.py
losses.py                                    core
metrics.py benchmark.py infer_volume.py      evaluation
config.py                                    all defaults + train_config
run_scenarios.sh                             scenario driver
scripts/                                     gates + analysis CLIs
tests/                                       run directly: python tests/<f>.py
vendor/                                      offline diffusers + VAE weights
splits/split.json                            frozen split, for external models
orgFeatXGB_CTPhase/                          REQUIRED dependency
analysis/BASELINE_REFERENCE.md               the numbers to beat
```

Outputs go to `../out_synthesis_train/<run>/`, patch caches to
`../out_synthesis_train/patch_cache/`. Files in `scripts/` and `tests/` carry a
3-line `sys.path` shim so they run from the repo root as `python scripts/foo.py`.

### What the fork removed, and where it went

`AdversarialLoss`, `FeatureMatchingLoss`, `PerceptualLoss`, `DinoPerceptualLoss`,
`PhaseSaliencyLoss`, `DinoSaliencyLoss`, `CyclicConsistencyLoss`,
`SegmentationConsistencyLoss`, `PatchGANDiscriminator`, `dino_backbone.py`, the
discriminator branch of `trainer.py`, 14 one-off `scripts/`, the historical
`analysis/` JSONs, and `PROJECT_PLAN.md` / `PLAN_FA.md`. All still in
`../synthetic_CECT`. Diffusion uses none of them, and VGG/DINO need a network
download the training host cannot do.

Consequence: `lambda_l1` is no longer parametric on other losses — the three
terms it used to back off for are gone, so L1 always starts at `LAMBDA_L1` and
`LAMBDA_L1_REDUCED` no longer exists. The **L1 decay curriculum is kept**; it is
what gives a zero entry in `ORGAN_WEIGHTS` any force.

`models.py` is otherwise **frozen**. `models_hetero.py` and `models_diffusion.py`
import from it and compose.

---

## 1. Three model families, one pipeline

Selected by config flag; `train.py` picks the trainer, `infer_volume.load_model`
picks the reconstruction path. `--use_diffusion` and `--use_hetero` are mutually
exclusive and `train.py` raises if both are set.

| family | flag | model | trainer | loss | inference output |
|---|---|---|---|---|---|
| deterministic | *(none)* | `UNetGenerator` | `Trainer` | L1 (+organ) | `_syn.nii.gz` |
| heteroscedastic | `--use_hetero` | `HeteroGenerator` | `Trainer` | Gaussian NLL | `_syn` + `_std` |
| diffusion | `--use_diffusion` | `DiffusionUNet` | `DiffusionTrainer` | MSE on `v`/`x0` | mean, `_syn1`, `_std`, organ medians |

The dataset, the patch geometry, the frozen split, the HU window, the tiling and
the blending are identical for all three.

---

## 2. `models_diffusion.py`

- `NoiseSchedule` — cosine, 1000 steps. Buffers `alphas_cumprod`, `sqrt_ab`,
  `sqrt_1mab`; methods `q_sample`, `to_target`, `predict_x0`, `predict_eps`,
  `eps_from_x0`. Round-trips exactly for both parameterisations at every `t`.
- `DiffusionUNet` — the `UNetGenerator` topology, built from the same
  `_EncBlock` / `_DecBlock` / `_norm` / `FiLM`. Three departures:
  1. `in_channels = 1 + cond_channels` — the noisy target concatenated with the
     NCCT (Palette/SR3-style conditioning).
  2. **Plain conv output head, no Sigmoid** — `v` and `eps` are unbounded and
     symmetric about zero.
  3. **FiLM in the encoder as well as the decoder** — unlike phase conditioning,
     which keeps the encoder phase-agnostic on purpose. The noise level changes
     what the input *is*.

  Every FiLM site is zero-init, so at step 0 the model is a plain U-Net on
  `cat([x_t, ncct])` and is exactly `t`-independent.
- Domain: **`[-1,1]` internally**. `dataset.py` is untouched and still emits
  `[0,1]`; `to_model` / `from_model` convert at the boundary.
- `TimestepEmbedding` — sinusoidal → 2-layer MLP → `cond_dim`, i.e. the same
  `(B, cond_dim)` vector `FiLM` already takes. Timestep, phase and level
  conditioning are literally one code path.
- CFG — `cfg_drop_prob=0.1` replaces the NCCT with a **learned** null embedding
  (not zeros: zeros are a valid mid-window HU value, so the unconditional branch
  would be trained on a lie). At sampling,
  `out_uncond + w*(out_cond - out_uncond)`.
- `DDIMSampler(eta, guidance)` — `eta=0` deterministic, `eta=1` ancestral.
  `sample()` takes `x_T` as an argument and never draws it (see §4).

`build_diffusion(cfg)` constructs model + schedule from a run config in one
place, so a checkpoint cannot be rebuilt under a different schedule.

Parameterisation is `--parameterisation {v,x0}`, default `v`. `eps` is
deliberately unavailable: at high noise it is poorly conditioned for a task whose
signal is a *level*, and its x0 error is amplified by `1/sqrt(alpha_bar)`.

---

## 3. `models_hetero.py`

`HeteroGenerator(UNetGenerator)` with a 2-channel head: `mu` keeps the parent's
Sigmoid, `log_var` is linear and bias-init to −6 (σ ≈ 30 HU, the right order for
the ~13 HU error being modelled). `out_conv` becomes `Identity` so both heads read
the last decoder feature map.

`losses.GaussianNLLLoss` clamps `log_var` to `[-14, 4]`; the lower bound is
load-bearing — NLL is unbounded below and would otherwise diverge on voxels the
model fits exactly.

**It cannot represent case-level spread.** Per-voxel σ is independent, so the
spread it induces on an organ median over ~10⁵ voxels is ~σ/√n ≈ 0. Its per-organ
variance ratio and coverage will be ~0 by construction. That is the finding, not a
bug — `scripts/calibration_eval.py` reports both levels so the gap is visible.

---

## 4. `infer_volume.py` — one tiling loop, pluggable predictors

`PatchPredictor` protocol: `n_out`, `begin_volume(padded_shape)`,
`__call__(batch, coords) -> (B, n_out, ...)`. Three implementations:
`DeterministicPredictor`, `HeteroPredictor` (emits `exp(log_var/2)`),
`DiffusionPredictor`.

Return is `(n_out, D, H, W)` in HU. Channel 0 gets the `hu_min` offset; channels
1+ are **spreads**, scaled by the window width but never offset.

### Volume-wide noise — the load-bearing part

`DiffusionPredictor.begin_volume` draws **one** `(D,H,W)` noise field per
(case, sample), seeded by `sha256(case_id)`, `sample_seed` and `k`. Each tile
crops its `x_T` from it. Overlapping tiles therefore share noise exactly where
they overlap and, with deterministic DDIM, the whole volume is one coherent draw.

Independent per-tile noise would give every tile its own contrast level, average
the case-level offset away across ~49 tiles per slice, worsen seams and flicker in
z — and the variance ratio would stay near 0.176 for a plumbing reason rather than
a scientific one. `tests/test_diffusion.py::test_volume_noise_coherence` pins it.

Per-case emission runs Welford over samples rather than holding N volumes:
`_syn` (mean), `_syn1` (sample 0), `_std`, `organ_medians.json`, and `_s{k}` only
with `--save_samples`. Two manifests — `manifest.csv` and `manifest_single.csv` —
so `benchmark.py` scores the mean and the single sample as two separate models.

`--dry_run` prints the tile count per volume without running the model. **Run it
first**: the sampling budget is `n_tiles × ddim_steps × n_samples × n_cases`,
doubled when `--guidance != 1`.

---

## 5. `trainer_diffusion.py`

Subclasses `Trainer` for checkpointing, history, curves and early stopping. It
does not call `Trainer.__init__` — that would build a `UNetGenerator` and a
`CompositeLoss` and shift every later RNG draw.

- Step: sample `t ~ U[0,T)`, sample noise, `q_sample`, MSE on the target.
  `OrganWeightedLoss` / `OrganHUProfileLoss` optionally on the predicted `x0`
  (converted back to `[0,1]`, the domain their λs were sized for), applied only
  for `t < aux_max_t` (default 700 of 1000).
- **Validation is deterministic**: each val patch gets a fixed `t` from an even
  grid and a fixed noise tensor, seeded once and identical every epoch. Without
  it the val curve is dominated by which timesteps were drawn and checkpoint
  selection is selecting on noise.
- Selection and early stopping on that val loss — *not* `val_org_ssim`, which for
  a diffusion model needs a sampling pass and is only computed periodically.
- A DDIM sample on ~16 fixed val patches runs every `diffusion_sample_every`
  epochs for a `val_org_ssim` curve and the sample grid. For looking at, never
  for selection. `_save_samples` draws from that sample; a plain forward pass
  would show one denoising step from noise, i.e. a grey blur.
- `history['gamma_*']` records mean |γ| per FiLM site. If they stay at 0, the
  network is denoising without knowing the noise level.

---

## 6. Calibration metrics — `metrics.py`

`crps_ensemble`, `interval_coverage`, `variance_ratio`, `pit_values`,
`calibration_metrics`, `min_samples_for_level`, `achieved_level`. Pure NumPy, no
distributional assumption (a diffusion ensemble has no closed form).

**Coverage uses order statistics, not `np.quantile`.** Interpolated empirical
quantiles are badly biased downward at small n — measured on perfectly calibrated
Gaussian draws: 0.726 at n=8, 0.813 at n=20, still only 0.891 at n=200. A correct
model would have read as over-confident, and at N=8 the bias (−0.17) exceeds any
miscalibration this project is likely to measure. The order-statistic form
(n samples → n+1 equally likely gaps) is unbiased at every n.

**A 90% interval needs N ≥ 19 samples to be expressible at all** — `[min, max]`
holds n−1 of the n+1 gaps, so the ceiling is (n−1)/(n+1). Below that
`interval_coverage` returns NaN rather than a low score. A 50% interval needs only
3, so it is the coverage number an N=8 exploratory run can produce; CRPS and the
variance ratio have no such floor.

→ **explore with `--n_samples 8`, run the final calibration pass with
`--n_samples 20`.**

`scripts/calibration_eval.py` scores at two levels — per-organ median HU per case
(the headline; the same quantity featHU reads) and per-voxel over the organ mask —
from either a diffusion ensemble (`--mode ensemble`) or a heteroscedastic (μ, σ)
field (`--mode gaussian`, which Monte-Carlos the organ-median distribution from
the organ's own voxels rather than using an asymptotic).

---

## 7. Gate D1 — `scripts/vae_fidelity.py`

Round-trips the **real** CECTs through `sd-vae-ft-mse` and scores the
reconstruction with the normal suite. A ceiling measurement: no latent-diffusion
method beats its own autoencoder. featHU > 13.5 HU kills every SD/ControlNet
route. Score the whole suite — featHU may survive where `raps_hf` does not.

Offline: `--vae_path vendor/sd-vae-ft-mse` (a local directory), `HF_HUB_OFFLINE=1`
set before diffusers is imported. See `vendor/README.md`. This is the only thing
in the repo that needs `diffusers`.

---

## 8. Non-negotiables inherited from the deterministic project

- **Frozen split.** `data_seed=42`, `val_split=test_split=0.15` → 97/20/20.
  `dataset.find_pairs_and_split` derives it from `rng(42).permutation`;
  `splits/split.json` is a dump for external models, not the source of truth.
- **3 seeds before any claim.** featHU 2σ gate ≈ 0.84 HU. `--sample_seed` is
  separate from `--seed` so sampling variance and seed variance stay separable.
- **`run_config.json` diff.** Each run should differ from its predecessor only in
  the keys intended.
- **Never delete `best_model.pth`.** Six earlier runs in the deterministic repo
  are permanently un-re-benchmarkable because their checkpoints were removed.
- **Absolute paths.** A run was lost on 2026-07-31 to a relative `data_dir`, and
  the output rsync has repeatedly created nested copies.
- `splits/levels.json` **does not exist** here. Level conditioning is off
  (`COND_ORGANS = []`); the plumbing survives in `models.py` / `dataset.py` and is
  inert. Regenerate it with the deterministic repo's `dump_levels.py` if it is
  ever turned on.

---

## 9. Tests

Run directly; there is no CI and pytest is not installed on the training host.

| file | what it pins |
|---|---|
| `test_diffusion.py` | schedule algebra at every t, unbounded output head, FiLM zero-init, CFG null path, DDIM determinism, **volume-noise coherence**, dims=3 |
| `test_calibration.py` | CRPS vs the Gaussian closed form, unbiased coverage at every n, the N≥19 limit, variance ratio at 1.0 and 0.176, PIT shape |
| `smoke_test_diffusion.py` | end-to-end train + sample + checkpoint round-trip for all three families; **deterministic validation**; the tiling loop through every predictor |
| `smoke_test.py` | every loss-flag combination incl. hetero, 2-D and 3-D |
| `smoke_test_infer.py` | tiling coverage, blend windows, the `run()` driver |
| inherited | `test_metrics`, `test_patch_cache`, `test_phase_cond`, `test_level_cond`, `test_max_train_cases`, `smoke_test_organ_focus`, `smoke_test_organ_weights` |
