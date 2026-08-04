# Implementation reference — diffusion fork

What the code **actually does** right now, not the design intent. A snapshot —
re-verify against the code before trusting a claim here.

This repo is a fork of `../synthetic_CECT` for the work in `DIFFUSION_PLAN.md`:
model `p(CECT | NCCT)` instead of `E[CECT | NCCT]`, and measure whether the
predicted spread is the true conditional spread. The deterministic repo is frozen
and still holds every result recorded in `analysis/BASELINE_REFERENCE.md`.

### Status

Ten runs exist in `../out_synthesis_train/` — six `diff_*` and four
`multiphase_*` (the latter produced by `../synthetic_CECT`, not this repo). All
of them plateaued, and diagnosing that produced the changes marked **(new)**
below.

**The gate has not been run.** `phase_infer/` is empty for all six `diff_*` runs,
so `raps_hf`, `grad_w1`, `seam`, `zflicker`, `featHU` and `var(gen)/var(real)` —
every number this fork exists to move — have never been computed for a diffusion
model. Until they are, "the diffusion plateaued" is not a statement that can be
evaluated: a single-sample val MAE of 1.42x the deterministic baseline is
*simultaneously* the behaviour §7 predicted as correct and the signature of a
model emitting a copy of its input. Run Step 0 of `run_scenarios.sh`'s scenario
block before training anything new.

**Read the per-voxel metrics with care.** MAE / PSNR / SSIM cannot see
blur-versus-texture and are minimised by the conditional mean, so they rank a
blurry copy of the NCCT *above* a correctly textured sample. Sorting the ten runs
by val MAE sorts them almost exactly by how much texture each model attempts, in
the wrong direction. `raps_hf` and `grad_w1` are the axis that discriminates;
`benchmark.py` has reported them all along.

Everything below is verified by `tests/`, which run on random tensors on CPU. The
numbered facts are checked against the code; the measurements quoted from
`../out_synthesis_train/*/history.json` are labelled where they appear.

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
  Plus `snr` and `snr_loss_weight` — see §2a.
- `offset_noise(like, c)` — `randn_like` plus a per-item constant of scale `c`.
  See §2a; this is the single most load-bearing change in the fork.
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

### `DiffusionUNet` in full

Measured by instantiating it (`dims=2, base_channels=64, cond_channels=1,
cond_dim=64, norm=group`): **13,793,474 parameters.** The deterministic
`UNetGenerator` at the same settings is 13,325,057 — the diffusion model is the
same network plus 434,752 of FiLM heads, 33,088 of timestep MLP, one extra input
channel and one `null_cond` scalar.

**Topology** — 5 resolutions, 4 MaxPool downsamples, at `patch_size=128`:

| stage | output shape | params |
|---|---|---|
| input | `(1 + cond_channels, 128, 128)` | — |
| `enc1` | `(64, 128, 128)` | 38,400 |
| `enc2` | `(128, 64, 64)` | 221,952 |
| `enc3` | `(256, 32, 32)` | 886,272 |
| `enc4` | `(512, 16, 16)` | 3,542,016 |
| `bottleneck` | `(512, 8, 8)` | 4,721,664 |
| `up4` → `dec4` | `(256, 16, 16)` | 524,544 + 2,360,832 |
| `up3` → `dec3` | `(128, 32, 32)` | 131,200 + 590,592 |
| `up2` → `dec2` | `(64, 64, 64)` | 32,832 + 147,840 |
| `up1` → `dec1` | `(64, 128, 128)` | 16,448 + 110,976 |
| `out_conv` | `(1, 128, 128)` | 65 |

Channel multipliers are hardcoded `(1, 2, 4, 8)` on `base_channels`. The decoder
is **asymmetric**: each `ConvTranspose` halves channels *before* concatenating the
skip, so `dec_k` is half the width of `enc_k`. **34% of all parameters sit in the
8x8 bottleneck.**

**Blocks.** `_EncBlock` / `_DecBlock` (`models.py:115-144`) are plain double-conv
stacks — `Conv3x3 → norm → LeakyReLU(0.2)` twice, with `Dropout2d` between the
two convs in the decoder. **There are no residual connections inside a block**
and no 1x1 shortcut; the only skips are the U-Net long skips (`torch.cat`).
Activation is `LeakyReLU(0.2)` throughout, not the SiLU of the DDPM lineage
(SiLU appears only inside the timestep MLP).

**Normalisation.** `GroupNorm(8, C)` at every site when `--generator_norm group`
is passed. Note `config.GEN_NORM` is still `'instance'`, and `build_diffusion`
reads `cfg['generator_norm']`, which is always present — so running
`--use_diffusion` *without* `--generator_norm group` silently gives InstanceNorm.
Every diffusion scenario in `run_scenarios.sh` passes it explicitly.

**Dropout.** `nn.Dropout2d` (whole feature maps, not element-wise) at rate 0.2:
twice in the bottleneck, once per decoder block, **none in the encoder**.

**Timestep injection — FiLM on the block output.** `TimestepEmbedding` is a
sinusoidal embedding of the raw integer `t` (32 frequencies → 64) through
`Linear(64,256) → SiLU → Linear(256,64)`. That 64-vector feeds nine `FiLM` sites,
one per block, each `Linear(64,h) → ReLU → Linear(h,2C)` with the last layer
**zero-initialised**, so at step 0 the model is exactly a plain U-Net on
`cat([x_t, cond])` and exactly `t`-independent.

`FiLM` applies `y = x*(1+gamma) + beta` per channel, spatially uniform. It is
applied to each block's **output**, i.e. **one injection per block** — a DDPM
ResBlock injects `t` *between* its two convs. Phase and level conditioning sum
into the same 64-vector, so timestep/phase/level share one code path.

`history['gamma_*']` logs mean `|gamma|` per site, which is how the failure in
§5 became visible: `gamma_bottleneck` is 0.0000 in three runs.

**Conditioning on the NCCT — channel concat only.** `in_channels = 1 +
cond_channels`, `x = cat([x_t, cond], dim=1)` at the input, never re-injected at
depth. No cross-attention, no separate condition encoder.

**Attention: there is none.** A repo-wide grep for `attention|attn|
MultiheadAttention` returns one hit, a `--help` string. No self-attention at
16x16 or 8x8. Combined with the zero bottleneck FiLM, the network has no
`t`-aware globally-mixing path at all.

**Output head.** `Conv1x1 → 1 channel, no activation` — deliberate, since `v` and
`eps` are unbounded and symmetric about zero and the parent's `Sigmoid` would
clip them.

**Domain.** `[-1,1]` internally; `dataset.py` emits `[0,1]` and `to_model` /
`from_model` convert at the boundary.

**Schedule.** Cosine (Nichol & Dhariwal), `s=0.008`, 1000 steps, verified
`alphas_cumprod[0] = 0.999959` and `[-1] = 1.0e-08` — terminal SNR is
effectively zero, so the sampler's first step is not leaking signal.

### Gaps versus a standard diffusion U-Net

| feature | status |
|---|---|
| self-attention at 16x16 / 8x8 | **absent** |
| residual blocks | **absent** — plain double-conv, long skips only |
| EMA of weights | **added** (§2a); was absent |
| SNR loss weighting | **added** (§2a); was absent |
| offset noise | **added** (§2a); was absent |
| activation | LeakyReLU(0.2), not SiLU |
| Adam beta1 | **0.9** (§2a); was 0.5 |
| t injection | one FiLM per block *output*, not per-conv inside a ResBlock |
| condition injection | channel concat at the input only |
| dropout | `Dropout2d` 0.2, decoder + bottleneck, none in encoder |

The first two are the outstanding items: they are what would let the 8x8
bottleneck stop reading `gamma = 0.0000`, and they change checkpoint topology, so
they belong in their own scenario.

---

## 2a. The level channel — offset noise, SNR weighting, EMA

Three changes aimed at one problem: **the quantity this fork exists to sample is
a per-case, spatially flat HU offset (aortic spread 96-218 HU, sd 24.4), and
almost nothing in the original pipeline gave that one degree of freedom any
weight.**

### `diffusion_offset_noise` (default 0.1)

Two independent failures, both structural:

1. The mean of a 128x128 unit-noise patch has sd `1/128 = 0.0078`, i.e. **2.3 HU**
   on this window, against a required 24.4 HU. At `eta=0` the DDIM sample is a
   deterministic function of `x_T`, so 2.3 HU was the sampler's entire budget for
   the level — roughly 10x short.
2. `F.mse_loss` is unweighted over spatial frequency, so the DC direction is 1 of
   16384 and contributed ~0.006% of the gradient.

`offset_noise` adds a per-item constant of scale `c`, which fixes both at once.
**It is applied at training AND at sampling** (`trainer_diffusion._train_step`,
`_fixed_val`, and `infer_volume.DiffusionPredictor._build_noise`) — one side only
would be a train/test mismatch in precisely the channel being measured.
`DiffusionPredictor` reads it out of `run_config.json`, never from a CLI default.

At inference the offset is **one constant for the whole padded volume**, not one
per tile: the same coherence argument that makes the noise field volume-wide (§4).

### `min_snr_gamma` (default 5.0) — and where it departs from the paper

Measured on this schedule, the weight each plain MSE places on the **x0 error**:

| t | 0 | 250 | 500 | 700 | 900 | 999 |
|---|---|---|---|---|---|---|
| plain v-MSE | 2.4e+04 | 6.49 | 1.97 | 1.25 | 1.02 | 1.00 |
| plain x0-MSE | 1 | 1 | 1 | 1 | 1 | 1 |

So `--parameterisation v`, the default, over-weights `t=0` by **24,000x** relative
to `t=999`, and `diff_v` vs `diff_x0` was never a comparison between two
objectives that differ only in parameterisation.

`NoiseSchedule.snr_loss_weight` divides out that implicit factor so both
parameterisations optimise the same thing. **The target is
`min(SNR, gamma) / SNR`, not the published `min(SNR, gamma)`** of Hang et al.
2023. The published target is `[5, 5, 0.97, 0.25, 0.024, 1e-8]` — it clamps the
low-t blowup but drives high-t weight to essentially zero, which is right for
image quality and backwards here, because the level is decided at high t. The
target used instead is flat 1.0 wherever `SNR <= gamma` and decays only above it.

### `ema_decay` (default 0.999)

There was no EMA anywhere in the repo. `trainer.WeightEMA`, wired on the
diffusion path only. Every val pass and every sample grid runs under the averaged
weights (`_validate` swaps them in), and the checkpoint stores:

| key | weights | read by |
|---|---|---|
| `G_state` | **averaged** | `infer_volume.load_model` — unchanged, EMA-unaware |
| `G_raw_state` | live | `load_checkpoint`, so a resume is exact |

Checkpoints written before EMA have only `G_state` and load unchanged.

### `diffusion_betas` (default `(0.9, 0.999)`)

Adam beta1 was 0.5, inherited from the GAN baseline where a short momentum window
keeps the generator responsive to a moving discriminator. There is no
discriminator here and the objective is stationary. The project-wide `betas` key
is deliberately **not** consulted on this path — it is always present, so falling
back to it would silently keep 0.5.

### `predict_x0(..., clip=False)` in the aux-loss path

`predict_x0` clamps to `[-1,1]` by default, which is correct when the estimate is
fed back into the next DDIM step and wrong in a loss: `clamp` has zero gradient
outside the range, so every saturated voxel contributed nothing to the organ and
HU-profile terms. At moderate-to-high t a large fraction of `x0_hat` is saturated,
so the aux gradient was masked far more aggressively than the documented
`t < aux_max_t` gate — and those are the only terms in the diffusion objective
that know about anatomy.

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

`--dry_run` prints the tile count per volume and the total forward-pass budget
without loading the model — it needs `run_config.json` and the data, but **not a
checkpoint**, so it is usable while the run is still training. **Run it
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
- **Selection is on the detail metrics, not the denoising loss.**
  `diffusion_selection` (config / `--diffusion_selection`, default `detail`):

  | mode | score, minimised |
  |---|---|
  | `detail` | `\|raps_hf - 1\| + val_org_grad_w1`, from the DDIM sample |
  | `val_loss` | the denoising MSE. Legacy; reproduces the pre-change runs |

  `val_loss` is a per-voxel loss and every per-voxel loss is minimised by the
  conditional mean, so selecting on it picks the epoch whose samples are
  *blurriest* — the exact failure this fork exists to fix. `selection_metric` is
  not consulted on this path at all.

  The detail score exists only on sampling epochs; a stale row scores `+inf` and
  is not a candidate. Before the first sample the score falls back to
  `1e6 + val_loss`, so `best_model.pth` always exists (a run whose
  `diffusion_sample_every` exceeds its epoch count would otherwise finish with no
  checkpoint at all, and `infer_volume.py` requires the file). It logs a warning
  when that fallback is in use.
- **Early stopping reads `val_loss`, not the selection score**
  (`Trainer._early_stop_score`, overridden here). Patience has to count against a
  signal that exists every epoch; counting it against a score that is `+inf` on
  ~90% of epochs would stop every run after `early_stop_patience` epochs.
- A DDIM sample on ~16 fixed val patches runs every `diffusion_sample_every`
  epochs for a `val_org_ssim` curve and the sample grid. For looking at, never
  for selection. `_save_samples` draws from that sample; a plain forward pass
  would show one denoising step from noise, i.e. a grey blur.
- `history['gamma_*']` records mean |γ| per FiLM site. If they stay at 0, the
  network is denoising without knowing the noise level. **This has already
  happened**: `gamma_bottleneck` is 0.0000 at the last epoch of `diff_v`,
  `diff_x0` and `diff_v_nocfg`, and `gamma_enc4`/`gamma_dec4` are ~1e-4. Only the
  two highest-resolution encoder/decoder levels learned any `t` dependence, so
  the one patch-global path in the network — the 8×8 bottleneck, the only place a
  case-level HU offset could be set — is still at its zero-init.

---

## 5a. Validation metrics — the detail axis

`trainer._detail_metric_set` scores a stack of paired 2-D centre slices with
`metrics.raps_hf_ratio` and `metrics.grad_hist_distance`, recording
`val_raps_hf`, `val_grad_w1` and `val_org_grad_w1` in `history.json`. Both
trainers call it: the deterministic one on the forward pass, the diffusion one on
the DDIM sample inside `_sampled_metrics` (a plain forward pass there would score
as pure blur regardless of the model). Capped at `_DETAIL_MAX_SLICES = 64`.

These exist because **MAE / PSNR / SSIM / NCC cannot see blur-vs-texture and rank
a blurry copy of the input above a correctly-textured sample.** Read `raps_hf` as
a distance from 1.0 (<1 blur, >1 hallucinated noise) and `grad_w1` as a distance
from 0. Neither is capped by residual registration error, because neither depends
on where the edges are.

`curves.png` is now 3×4. Row 3 is the detail axis plus `val_loss`, which drives
selection and was previously absent from the figure entirely. Two plotting rules:

- **Carried-forward values are never drawn as measurements.** `val_sampled` is
  1.0 on epochs whose val row was freshly computed and 0.0 when
  `_last_sampled` was reused; markers go on the measured epochs only, with a
  faint line for the shape. The 10-epoch flat steps in every pre-change diffusion
  `curves.png` are this artefact, not model behaviour.
- **Exact `0.0` is masked to NaN.** It is the "not measured" sentinel throughout
  (`_validate` seeds the val keys at 0.0, `_detail_metric_set` maps NaN to 0.0,
  `n_organ_items` exists to disambiguate a real 0). Plotting it rescaled the PSNR
  panel from a 23.5–28 dB range down to 0–28 and flattened the curve.

Histories written before these keys existed still plot: `val_sampled` defaults to
all-fresh and the row-3 panels say so rather than drawing an empty axis.

`losses.CompositeLoss` now logs `d['l1']` as the **raw** MAE in both hetero and
non-hetero mode. It used to log the λ-scaled value in one and the raw value in
the other, so `history['train_l1']` changed units by ~100× depending on a flag
while `_plot_history` drew both on one axis. The scaled contribution is
`train_l1 * lambda_l1`; `train_gen_total` is unaffected.

`KEEP_N_CHECKPOINTS` is 10, up from 3: `best_model.pth` is chosen on one metric,
and when that metric turns out to have been the wrong one the alternative epochs
were already deleted and the run had to be repeated to recover them.

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

## 7a. `dataset.py` — augmentation and the cache key

**`augment` (default `'flip'`, train split only).** There was no augmentation of
any kind: the cache is a fixed set of patches and `__getitem__` a list lookup, so
200 epochs saw 200 copies of the same 20,000 images from 97 cases — 10.6% of the
188k available. The measured cost is a ~2x train/val gap (`memorize97`: train MAE
0.0086 vs val 0.0157; `capacity_overfit`: 0.0058 vs 0.0192). **These models
overfit; they do not underfit**, so capacity and schedule length are not the
lever.

Modes: `none` | `flip_ap` | `flip` | `flip_rot90`. Source, target and mask get
the identical transform, in-plane axes only (always the last two, which holds for
2-D, 2.5-D stacks and 3-D alike). `tests/test_augment.py` pins the pairing, the
layouts, and that the cached arrays are not written through.

**There is no intensity mode, deliberately.** The prediction target IS an
absolute HU level. Jittering the target destroys the label; jittering only the
source teaches the model the level is unrelated to the input. A *paired* offset
is the only safe intensity transform and is left unimplemented rather than left
as a footgun.

Augmentation is applied at `__getitem__`, so it does **not** enter the cache key
and costs no re-preload.

**Cache key.** Now includes `(size, mtime_ns)` per file, not just the path
strings. Paths alone hash the *name* of the data: regenerating
`B2_deeds__aligned` in place — re-running deeds, fixing an alignment bug,
re-exporting — leaves every path identical, so the digest would not move and
every later scenario would silently train on the old registration. Given that the
whole question is downstream of registration quality, that failure would have
been invisible and total.

Also fixed: `dataset.py` did `from random import random` (the function) and
`_worker_init` called `random.seed(s)`, which would raise. Latent only because
`NUM_WORKERS = 0`. Augmentation decisions are drawn from **torch's** RNG, which
the DataLoader re-seeds per worker correctly.

**`organ_focus_frac` is still 0.0** and is now reachable as `--organ_focus_frac`.
It is the largest single lever identified and it has not been exercised: of 16
sampled training patches, roughly four contain an abdominal organ; the rest are
shoulder, arm, chest wall, fat, spine and air, where the correct output is a copy
of the input.

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
  the output rsync has repeatedly created nested copies. `train.py --data_dir`
  now overrides `config.DATA_DIR` (and is recorded in `run_config.json`);
  `train.py` warns when the path is relative and **raises** when it does not
  exist, rather than training on a silently smaller split.
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
| `test_diffusion_core.py` | offset noise raises the DC spread to ~c and is one constant per volume at inference; both parameterisations get the same effective x0 weighting; high-t keeps full weight; EMA swap restores exactly and ships the average |
| `test_augment.py` | source/target/mask stay aligned, values are permuted not altered, val/test never augmented, 2.5-D and non-square layouts, the cache is not mutated |
| `test_calibration.py` | CRPS vs the Gaussian closed form, unbiased coverage at every n, the N≥19 limit, variance ratio at 1.0 and 0.176, PIT shape |
| `smoke_test_diffusion.py` | end-to-end train + sample + checkpoint round-trip for all three families; **deterministic validation**; the tiling loop through every predictor |
| `smoke_test.py` | every loss-flag combination incl. hetero, 2-D and 3-D |
| `smoke_test_infer.py` | tiling coverage, blend windows, the `run()` driver |
| `test_benchmark_discovery.py` | multi-phase runs are found (one row per phase, never pooled), tiling resolves for both layouts, disjoint-case models are not silently reported as identical, an empty table exits with a message instead of a bare `StopIteration` |
| inherited | `test_metrics`, `test_patch_cache`, `test_phase_cond`, `test_level_cond`, `test_max_train_cases`, `smoke_test_organ_focus`, `smoke_test_organ_weights` |
