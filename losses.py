"""
Loss functions for NCCT→CECT synthesis — 2-D and 3-D compatible.

All losses accept inputs of shape (B, C, H, W) [2-D] or (B, C, D, H, W) [3-D].
3-D handling strategy per loss:
  L1 / MSE / OrganWeightedLoss / OrganHUProfileLoss / GaussianNLLLoss
    → dimension-agnostic PyTorch ops, work unchanged.

  SSIMLoss
    → 2-D: standard Gaussian-window SSIM.
    → 3-D: applied per 2-D slice (depth loop), averaged.

  GradientLoss (Sobel)
    → 2-D: x- and y-gradients.
    → 3-D: x-, y-, and z-gradients.

  FrequencyLoss (FFT)
    → 2-D: torch.fft.fft2 amplitude.
    → 3-D: torch.fft.fftn over H×W per slice (same spatial frequency target).

WHAT THIS FORK DELETED, AND WHY (see DIFFUSION_PLAN.md)
------------------------------------------------------
`PerceptualLoss`, `DinoPerceptualLoss`, `PhaseSaliencyLoss`, `DinoSaliencyLoss`,
`CyclicConsistencyLoss` and `SegmentationConsistencyLoss` are gone: the VGG/DINO
backbones need a network download the training host cannot do, and the rest went
with them. The originals are in `../synthetic_CECT/losses.py`, which is frozen
and still reproduces every number in `analysis/BASELINE_REFERENCE.md`.

WHAT CAME BACK
--------------
`AdversarialLoss` and `FeatureMatchingLoss`. The original fork note said
"diffusion does not use a discriminator", which is true of DDPM as published and
false of every diffusion model that has to produce sharp output in a handful of
sampling steps. Here the critic is applied to the model's ONE-STEP x0 ESTIMATE
(see `trainer_diffusion.DiffusionTrainer._train_step` and `models_disc.py`), so
it attaches to the diffusion objective the same way the organ losses already do
-- through `predict_x0` -- and needs no sampling loop in the training step.

READ THIS BEFORE TURNING IT ON. x0_hat is the posterior MEAN E[x0|x_t], not a
sample from p(x0|x_t). Pushing a conditional mean to be indistinguishable from a
sample is, strictly, asking the model to be overconfident: it buys sharpness by
spending calibration. That is a real cost in this thesis, which measures
per-voxel CRPS/coverage and the sample-to-target variance ratio. The term is
therefore (a) gated to low t, where the posterior is nearly a point mass and the
mean/sample distinction almost vanishes, (b) small by default (lambda_adv=2
against an MSE of order 0.01-1), and (c) something to report var-ratio and
coverage for BEFORE and AFTER, not to adopt on a PSNR table.

WHAT IS NEW HERE
----------------
`GaussianNLLLoss` — the heteroscedastic baseline of DIFFUSION_PLAN.md §6 step 2,
the bar diffusion has to clear on per-voxel calibration.

`OrganWeightedLoss` and `OrganHUProfileLoss` are unchanged and are reused by the
diffusion trainer, applied to the predicted x0 (§5.4).
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0.  Adversarial loss
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """LSGAN / BCE / hinge PatchGAN adversarial loss (dimension-agnostic).

    'lsgan' is the default and matches the GAN baseline in ../synthetic_CECT, so
    the adversarial scenarios here are comparable to the recorded ones.

    'hinge' is offered because it is the pairing that spectral normalisation was
    introduced with (Miyato et al. 2018) and is markedly better behaved when D is
    winning -- which it always is early in a diffusion run, where x0_hat at
    moderate t is visibly not a CT. Under LSGAN a saturated D still returns a
    large gradient that is pure noise to G; the hinge clamps it to zero.
    """

    def __init__(self, mode: str = 'lsgan'):
        super().__init__()
        assert mode in ('bce', 'lsgan', 'hinge')
        self.mode = mode

    def disc_loss(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == 'lsgan':
            return 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                          F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
        if self.mode == 'hinge':
            return 0.5 * (F.relu(1.0 - pred_real).mean() +
                          F.relu(1.0 + pred_fake).mean())
        # One-sided label smoothing on the real branch only (0.9), as in the
        # baseline: smoothing the FAKE label too is the variant that is known to
        # reinforce a wrong D rather than regularise it.
        real = F.binary_cross_entropy_with_logits(pred_real, torch.full_like(pred_real, 0.9))
        fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (real + fake)

    def gen_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == 'lsgan':
            return F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        if self.mode == 'hinge':
            return -pred_fake.mean()
        return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))


# ---------------------------------------------------------------------------
# 0b. Feature-matching loss (discriminator intermediate layers)
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """L1 on discriminator intermediate features (pix2pixHD / Hau21).

    The real features MUST come from the same D state as the fake ones. The
    reference implementation reused the features produced inside the D step,
    i.e. from D *before* its update, and compared them against fake features
    from D *after* it -- and under disc_update_freq > 1 it silently contributed
    exactly zero on the steps where D did not run, which scales the term's
    effective weight by 1/disc_update_freq without saying so. The trainer here
    recomputes both under the post-update D, in one place.
    """

    def forward(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if not real_features:
            # Device- and dtype-correct, unlike a bare torch.tensor(0.0), which
            # raises the moment it is added to a CUDA total.
            return fake_features[0].new_zeros(()) if fake_features else torch.zeros(())
        loss = sum(F.l1_loss(fk, re.detach())
                   for re, fk in zip(real_features, fake_features))
        return loss / len(real_features)


# ---------------------------------------------------------------------------
# 1.  SSIM loss (2-D; 3-D = per-slice average)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    1 − SSIM loss with Gaussian window.
    3-D inputs are processed slice-by-slice (average over depth).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ws  = window_size
        self.pad = window_size // 2
        self.register_buffer('win', self._make_win(window_size, sigma))

    @staticmethod
    def _make_win(sz, sigma) -> torch.Tensor:
        c = torch.arange(sz, dtype=torch.float32) - sz // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        w = g.unsqueeze(0) * g.unsqueeze(1)
        return w.unsqueeze(0).unsqueeze(0)           # (1, 1, sz, sz)

    def _ssim2d(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute 1-SSIM on 2-D tensors (B, C, H, W)."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        C  = p.size(1)
        w  = self.win.expand(C, 1, -1, -1)
        pad = self.pad

        mu_p  = F.conv2d(p, w, padding=pad, groups=C)
        mu_t  = F.conv2d(t, w, padding=pad, groups=C)
        mu_pp = mu_p ** 2; mu_tt = mu_t ** 2; mu_pt = mu_p * mu_t

        sig_pp = F.conv2d(p * p, w, padding=pad, groups=C) - mu_pp
        sig_tt = F.conv2d(t * t, w, padding=pad, groups=C) - mu_tt
        sig_pt = F.conv2d(p * t, w, padding=pad, groups=C) - mu_pt

        ssim_map = ((2*mu_pt + C1)*(2*sig_pt + C2)) / \
                   ((mu_pp + mu_tt + C1)*(sig_pp + sig_tt + C2))
        return 1.0 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            # 3-D: iterate over depth slices
            B, C, D, H, W = pred.shape
            losses = [self._ssim2d(pred[:, :, d], target[:, :, d]) for d in range(D)]
            return torch.stack(losses).mean()
        return self._ssim2d(pred, target)


# ---------------------------------------------------------------------------
# 2.  Gradient / Sobel loss (extended to 3-D)
# ---------------------------------------------------------------------------

class GradientLoss(nn.Module):
    """
    L1 loss on Sobel gradient magnitude.
    2-D: x- and y-directions.
    3-D: x-, y-, and z-directions (3-D Sobel approximation).
    """

    def __init__(self):
        super().__init__()
        # 2-D Sobel
        kx2 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky2 = kx2.t()
        self.register_buffer('kx2', kx2.view(1, 1, 3, 3))
        self.register_buffer('ky2', ky2.view(1, 1, 3, 3))

        # 3-D Sobel (x-direction kernel, others derived by permutation)
        kx3 = torch.zeros(3, 3, 3, dtype=torch.float32)
        kx3[:, :, 0] = -1; kx3[:, :, 2] = 1
        kx3[1, 1, 0] *= 2; kx3[1, 1, 2] *= 2
        ky3 = kx3.permute(0, 2, 1)
        kz3 = kx3.permute(2, 1, 0)
        self.register_buffer('kx3', kx3.view(1, 1, 3, 3, 3))
        self.register_buffer('ky3', ky3.view(1, 1, 3, 3, 3))
        self.register_buffer('kz3', kz3.view(1, 1, 3, 3, 3))

    def _grad2d(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        kx = self.kx2.expand(C, 1, -1, -1)
        ky = self.ky2.expand(C, 1, -1, -1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def _grad3d(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        kx = self.kx3.expand(C, 1, -1, -1, -1)
        ky = self.ky3.expand(C, 1, -1, -1, -1)
        kz = self.kz3.expand(C, 1, -1, -1, -1)
        gx = F.conv3d(x, kx, padding=1, groups=C)
        gy = F.conv3d(x, ky, padding=1, groups=C)
        gz = F.conv3d(x, kz, padding=1, groups=C)
        return torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            return F.l1_loss(self._grad3d(pred), self._grad3d(target))
        return F.l1_loss(self._grad2d(pred), self._grad2d(target))


# ---------------------------------------------------------------------------
# 3.  Frequency (FFT) loss
# ---------------------------------------------------------------------------

class FrequencyLoss(nn.Module):
    """L1 on the FFT AMPLITUDE spectrum, in one of three weightings.

    2-D: fft2 over H×W.  3-D: fft2 over H×W per slice, averaged over D.

    WHERE THIS LOSS ACTUALLY PUTS ITS GRADIENT (measured, not assumed)
    -----------------------------------------------------------------
    It is tempting to argue that an L1 on raw |FFT| must be dominated by the
    low-frequency bins, because their amplitudes are ~45x larger per bin, and
    therefore that the term is just another blur-inducing reconstruction loss.
    THAT ARGUMENT IS WRONG, and it is worth recording why so nobody re-derives it.

    L1 is not scale-sensitive the way L2 is: d|L1|/d(bin) is +-1/N for every bin
    regardless of that bin's magnitude. What decides the balance is where the
    prediction's absolute amplitude ERRORS land. Measured on a synthetic CT-like
    slice (smooth body + fine grain), as the share of total L1 loss per radial
    band:

        failure mode            DC+vlow    low     mid    high
        blurred (5x5 box)          4.9%   10.9%   30.0%   54.2%
        global +20 HU offset     100.0%    0.0%    0.0%    0.0%
        organ contrast x1.2       26.6%    4.7%   23.7%   45.0%

    So against BLUR — the failure this term is meant to catch — a majority of the
    gradient is already in the high band. The term works. What is true, and much
    milder, is that the DC+very-low band is only 0.8% of the bins while carrying
    4.9% of the loss, i.e. it is over-weighted ~6x PER BIN relative to uniform.
    That is a tuning imbalance, not a defect, and `banded` exists to correct it.

    Note the middle row: for a pure level error the loss is entirely a DC term.
    That is a feature here, not a bug — case-level HU offset is this project's
    primary endpoint (DIFFUSION_PLAN.md section 2).

    AMPLITUDE ONLY, NOT THE COMPLEX SPECTRUM. Discarding phase makes the term
    translation-invariant, so it reads edge STATISTICS rather than edge POSITION
    and is not capped by the residual registration error that limits every
    per-voxel metric in this project (the same argument metrics.py makes for
    `grad_hist_distance`). The published Focal Frequency Loss uses the complex
    difference; `focal` here deliberately does not, because reintroducing phase
    would reintroduce that ceiling.

    MODES
      'raw'    (default) uniform weighting. What every existing config means;
               unchanged so recorded runs stay reproducible.
      'banded' weight the per-bin difference by radial frequency r, normalised
               to mean 1 so lambda_frequency keeps its scale. Removes the ~6x
               per-bin DC over-weighting above and pushes the gradient further
               into the band `metrics.raps_hf_ratio` reports.
      'focal'  Focal Frequency Loss (Jiang et al., ICCV 2021) adapted to
               amplitude: weight each bin by its own DETACHED normalised error to
               the power alpha, so bins the model is already fitting stop
               competing for gradient with the ones it is not. Adaptive, and
               therefore the arm to reach for only if 'banded' shows the axis
               matters at all.

    CONFOUND, STATE IT IN THE WRITE-UP: 'banded' optimises almost exactly what
    `metrics.raps_hf_ratio` reports. For runs using it, raps_hf and grad_w1 are
    DIAGNOSTICS, not evidence; the decision metrics stay featHU, org_mae and the
    calibration numbers.
    """

    def __init__(self, mode: str = 'raw', focal_alpha: float = 1.0):
        super().__init__()
        if mode not in ('raw', 'banded', 'focal'):
            raise ValueError(f"unknown frequency_mode {mode!r} — "
                             f"expected 'raw', 'banded' or 'focal'")
        self.mode = mode
        self.focal_alpha = float(focal_alpha)
        # Radial-frequency weights are per (H, W); cached because the patch size
        # is fixed for a run and rebuilding them every step is pure overhead.
        self._wcache = {}

    def _amp(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            return torch.abs(torch.fft.fft2(x2d, norm='ortho'))
        return torch.abs(torch.fft.fft2(x, norm='ortho'))

    def _radial_w(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Radial frequency per bin, normalised to mean 1.

        Mean-1 normalisation is what keeps lambda_frequency comparable between
        modes: without it 'banded' would silently be a ~2x smaller term than
        'raw' at the same lambda, and the mode ablation would be confounded with
        a weight change — the exact mistake config.py's LAMBDA_L1 note warns
        about.
        """
        key = (h, w, device, dtype)
        if key not in self._wcache:
            fy = torch.fft.fftfreq(h, device=device, dtype=dtype)[:, None]
            fx = torch.fft.fftfreq(w, device=device, dtype=dtype)[None, :]
            r = torch.sqrt(fy ** 2 + fx ** 2)
            self._wcache[key] = r / r.mean().clamp_min(1e-12)
        return self._wcache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ap, at = self._amp(pred), self._amp(target)
        diff = (ap - at).abs()

        if self.mode == 'raw':
            return diff.mean()

        if self.mode == 'banded':
            w = self._radial_w(diff.shape[-2], diff.shape[-1],
                               diff.device, diff.dtype)
            return (diff * w).mean()

        # focal: per-bin weight from the bin's own error, DETACHED.
        # Detaching is load-bearing — a live weight would let the model lower the
        # loss by shrinking the weight instead of the error, which is a
        # degenerate optimum, and it is what the original paper does too.
        w = diff.detach()
        w = w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        if self.focal_alpha != 1.0:
            w = w ** self.focal_alpha
        return (diff * w).mean()


# ---------------------------------------------------------------------------
# 4.  Organ-weighted loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class OrganWeightedLoss(nn.Module):
    """Mask-weighted L1, in one of two modes.

    Per-organ (`organ_weights` given): a {label_id: weight} lookup table, so each
    TotalSegmentator label carries its own weight and a weight of 0 excludes that
    anatomy from the gradient entirely. Requires a MULTI-LABEL mask (raw label
    ids) — a binarised mask collapses every organ onto the label-1 weight.

    Uniform (`organ_weights` None): the legacy behaviour — every masked voxel
    gets `organ_weight`× the background, from a binarised mask.

    L1 rather than MSE: MSE penalises large errors quadratically and so regresses
    to the conditional mean harder than L1 does, which is precisely the blur this
    term exists to counteract.
    """

    def __init__(
        self,
        organ_weight:      float = 10.0,
        organ_weights:     Optional[Dict[int, float]] = None,
        default_weight:    float = 1.0,
        background_weight: float = 1.0,
        max_label:         int = 256,
    ):
        super().__init__()
        self.uniform_weight = organ_weight
        self.per_organ = bool(organ_weights)
        if self.per_organ:
            lut = torch.full((max_label,), float(default_weight))
            lut[0] = float(background_weight)
            for lid, w in organ_weights.items():
                if not 0 <= int(lid) < max_label:
                    raise ValueError(f"organ label id {lid} outside [0,{max_label})")
                lut[int(lid)] = float(w)
            if float(lut.sum()) == 0.0:
                raise ValueError(
                    "all organ weights are zero — the organ loss would be "
                    "identically 0 and contribute no gradient."
                )
            self.register_buffer('lut', lut)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return F.l1_loss(pred, target)
        if self.per_organ:
            w = self.lut[mask.round().long().clamp(0, self.lut.numel() - 1)]
        else:
            w = 1.0 + (self.uniform_weight - 1.0) * mask.clamp(0, 1)
        # Normalise by w.sum(), not .mean(): with zero-weighted regions present a
        # plain mean shrinks the loss as more area is excluded, which would tie
        # the effective lambda_organ to whichever weight scheme is in use and make
        # scenarios incomparable.
        return (w * (pred - target).abs()).sum() / w.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# 5.  Organ HU-profile loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class OrganHUProfileLoss(nn.Module):
    """Penalise each organ's MEAN intensity deviation, not its per-voxel error.

    Motivation, from the measured ablation: the XGBoost phase classifier reads
    per-organ *median HU* — nothing else. Contrast phase is defined by the
    absolute enhancement level of each organ, so that is what a phase-faithful
    generator has to get right. Per-voxel losses optimise it only indirectly, and
    the scenario that most improved per-organ HU error (organ_curriculum,
    -1.58 HU vs baseline, t=-4.22) did so as a side effect rather than by
    targeting it.

    This complements OrganWeightedLoss rather than replacing it: that one
    sharpens texture *within* an organ, this one fixes the organ's overall level.
    A patch can score 0 here while looking nothing like the target, so it must
    never be the only spatial term.

    Weight-0 organs (bowel) are skipped entirely, as in OrganWeightedLoss —
    their content is not inferable from NCCT, so their mean HU is not a
    meaningful target either.
    """

    def __init__(
        self,
        organ_weights:  Optional[Dict[int, float]] = None,
        default_weight: float = 1.0,
        min_voxels:     int = 16,
        max_label:      int = 256,
    ):
        super().__init__()
        self.min_voxels = min_voxels
        self.default_weight = float(default_weight)
        lut = torch.full((max_label,), float(default_weight))
        lut[0] = 0.0                      # background has no meaningful "level"
        for lid, w in (organ_weights or {}).items():
            lut[int(lid)] = float(w)
        self.register_buffer('lut', lut)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return pred.new_zeros(())

        lbl = mask.round().long().clamp(0, self.lut.numel() - 1)
        total = pred.new_zeros(())
        wsum = pred.new_zeros(())
        # Loop over the labels actually present — typically ~10-40 per patch, far
        # fewer than the 117 possible, and each iteration is two masked means.
        for lid in torch.unique(lbl):
            if lid.item() == 0:
                continue
            w = self.lut[lid]
            if w == 0:
                continue
            sel = (lbl == lid)
            n = sel.sum()
            if n < self.min_voxels:       # too few voxels for a stable mean
                continue
            mu_p = pred[sel].mean()
            mu_t = target[sel].mean()
            total = total + w * (mu_p - mu_t).abs()
            wsum = wsum + w
        return total / wsum.clamp_min(1e-8)


# ---------------------------------------------------------------------------
# 6.  Gaussian NLL — the heteroscedastic baseline (DIFFUSION_PLAN.md §6 step 2)
# ---------------------------------------------------------------------------

class GaussianNLLLoss(nn.Module):
    """Per-voxel heteroscedastic negative log-likelihood.

    The model emits (mu, log_var) per voxel and is trained on

        0.5 * [ log_var + (target - mu)^2 / exp(log_var) ]

    which is the Gaussian NLL up to an additive constant. Unlike L1/MSE this
    makes the model state how uncertain it is, and Kendall & Gal (2017) is the
    reference: log_var, not var, because the exponential keeps the variance
    positive without a clamp and keeps the gradient well-scaled near zero.

    WHAT THIS CAN AND CANNOT REPRESENT — read before interpreting its numbers.
    Every voxel gets its OWN independent sigma. The spread this induces on an
    organ's MEDIAN HU, taken over ~1e5 voxels, is therefore ~sigma/sqrt(n), i.e.
    essentially zero. So this loss will score a near-zero organ-level variance
    ratio and near-zero 90% interval coverage BY CONSTRUCTION, no matter how well
    it is trained.

    That is not a bug to be tuned away — it is the point of including it.
    DIFFUSION_PLAN.md §2 measures the residual as case-level (dose, bolus timing,
    cardiac output), and a per-voxel independent noise model cannot express a
    case-level degree of freedom at all. This term is a genuine bar on the
    PER-VOXEL calibration rows and a demonstration of the gap on the per-organ
    ones. Both are reported; see scripts/calibration_eval.py.

    `log_var_min/max` bound the variance to a sane HU range in [0,1] units. The
    lower bound is the load-bearing one: without it, NLL is unbounded below —
    the model drives log_var to -inf on voxels it happens to fit exactly and the
    loss diverges. -14 corresponds to sigma ~ 9e-4, i.e. ~0.55 HU on the 600 HU
    window, comfortably below the ~13 HU error being modelled.
    """

    def __init__(self, log_var_min: float = -14.0, log_var_max: float = 4.0):
        super().__init__()
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        lv = log_var.clamp(self.log_var_min, self.log_var_max)
        nll = 0.5 * (lv + (target - mu) ** 2 * torch.exp(-lv))
        if weight is None:
            return nll.mean()
        # Same normalisation rule as OrganWeightedLoss: divide by the weight sum,
        # not the element count, so a zero-weighted region shrinking the numerator
        # does not silently shrink the effective lambda too.
        return (weight * nll).sum() / weight.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# CompositeLoss — all losses combined
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Configurable composite loss.

    Default: L1 alone. Every extra term is off; enable via config flags.

    Works identically for 2-D (B,1,H,W) and 3-D (B,1,D,H,W) inputs — each
    sub-loss handles the dimension check internally.

    Key config flags  (default → value):
      use_ssim             False
      use_gradient         False
      use_frequency        False
      use_organ            False    needs 'mask' in batch (auto-loaded by dataset.py
                                     when use_organ is True)
      use_hu_profile       False    per-organ MEAN-HU term; needs 'mask'
      use_hetero           False    heteroscedastic head: `pred` becomes (mu, log_var)
                                     and L1 is replaced by Gaussian NLL

    Key λ values (defaults): lambda_l1=100 (decayed by the curriculum when
    use_l1_decay is on), lambda_organ=20, lambda_hu_profile=50, lambda_fm=10.
    lambda_adv is NOT one of them — it is passed in per call as `adv_weight`,
    because its warmup schedule is owned by the trainer (see __init__).

      use_adversarial      False    PatchGAN adversarial term; needs
                                     `adv_fake_logits` from the trainer
      use_feature_matching False    L1 on D's intermediate features; needs
                                     `real_features` / `fake_features`

    WHAT CHANGED IN THIS FORK. `use_perceptual`, `use_saliency`, `use_cycle` and
    `use_seg_consistency` are gone along with their terms — see this module's
    docstring. lambda_l1 is NOT made parametric on which losses are active, even
    now that the adversarial term is back: the reference implementation dropped
    lambda_l1 100 -> 25 the moment any realism loss was switched on, which
    silently confounded "added adversarial" with "quartered L1" in every
    adversarial-vs-L1 comparison it produced. Use `use_l1_decay` if the L1 weight
    should move; then it moves on an epoch schedule that is logged, auditable and
    identical across scenarios.
    """

    def __init__(self, config: Dict):
        super().__init__()
        c = config

        # L1 decay curriculum: hold lambda_l1 until l1_decay_start_epoch, then
        # ramp linearly down to lambda_l1_floor by l1_decay_end_epoch and hold.
        # The floor is deliberately non-zero — see config.py's USE_L1_DECAY note.
        self.use_l1_decay    = c.get('use_l1_decay', False)
        self.l1_decay_start  = c.get('l1_decay_start_epoch', 10)
        self.l1_decay_end    = c.get('l1_decay_end_epoch', 30)
        self.lambda_l1_floor = c.get('lambda_l1_floor', 25.0)
        self.lambda_l1       = c.get('lambda_l1', 100.0)
        self._epoch          = 0

        if self.use_l1_decay:
            if self.lambda_l1_floor > self.lambda_l1:
                log.warning(f"lambda_l1_floor ({self.lambda_l1_floor}) > lambda_l1 "
                            f"({self.lambda_l1}) — L1 will ramp UP, not decay.")
            elif self.lambda_l1_floor == self.lambda_l1:
                log.warning(
                    f"use_l1_decay is ON but lambda_l1 == lambda_l1_floor "
                    f"({self.lambda_l1}) — the curriculum is a NO-OP and L1 will "
                    f"stay constant for the whole run. Lower lambda_l1_floor or "
                    f"raise lambda_l1.")

        # Heteroscedastic head. It REPLACES the L1 fidelity term rather than
        # adding to it: L1 and Gaussian NLL are two different likelihoods for the
        # same quantity, and keeping both would let the model satisfy the L1 term
        # while leaving sigma free to drift, which is exactly the calibration this
        # branch exists to measure.
        self.use_hetero   = c.get('use_hetero', False)
        self.lambda_nll   = c.get('lambda_nll', 1.0)
        if self.use_hetero:
            self.nll = GaussianNLLLoss(
                log_var_min = c.get('log_var_min', -14.0),
                log_var_max = c.get('log_var_max', 4.0),
            )

        # Adversarial / feature matching — the GENERATOR half only. D's half of
        # the objective lives in `trainer_adv.AdversarialMixin`, which builds its
        # own `AdversarialLoss` from the same `adv_mode` key; the class is
        # stateless, so the two halves cannot disagree.
        #
        # lambda_adv and its warmup are NOT owned here, unlike lambda_l1: the
        # weight is passed in per call (`adv_weight`). The diffusion trainer does
        # not use CompositeLoss at all, so a warmup counter here would be a
        # SECOND schedule that only one of the two paths advances — and the one
        # the trainer logs into history['lambda_adv'] would not be the one the
        # loss applied. `AdversarialMixin._adv_w` is the single owner.
        self.use_adv    = c.get('use_adversarial', False)
        self.use_fm     = c.get('use_feature_matching', False)
        self.lambda_fm  = c.get('lambda_fm', 10.0)
        if self.use_adv or self.use_fm:
            self.adv_loss = AdversarialLoss(mode=c.get('adv_mode', 'lsgan'))
        if self.use_fm:
            self.fm = FeatureMatchingLoss()

        self.use_ssim   = c.get('use_ssim', False)
        self.lambda_ssim= c.get('lambda_ssim', 10.0)
        if self.use_ssim:
            self.ssim = SSIMLoss()

        self.use_grad   = c.get('use_gradient', False)
        self.lambda_grad= c.get('lambda_gradient', 5.0)
        if self.use_grad:
            self.gradient = GradientLoss()

        self.use_freq   = c.get('use_frequency', False)
        self.lambda_freq= c.get('lambda_frequency', 1.0)
        if self.use_freq:
            self.frequency = FrequencyLoss(
                mode        = c.get('frequency_mode', 'raw'),
                focal_alpha = c.get('frequency_focal_alpha', 1.0),
            )

        self.use_organ  = c.get('use_organ', False)
        self.lambda_organ = c.get('lambda_organ', 5.0)
        if self.use_organ:
            self.organ = OrganWeightedLoss(
                organ_weight      = c.get('organ_weight', 10.0),
                organ_weights     = c.get('organ_weights'),
                default_weight    = c.get('organ_weight_default', 1.0),
                background_weight = c.get('organ_weight_background', 1.0),
            )

        self.use_hu_profile  = c.get('use_hu_profile', False)
        self.lambda_hu_profile = c.get('lambda_hu_profile', 10.0)
        if self.use_hu_profile:
            self.hu_profile = OrganHUProfileLoss(
                organ_weights  = c.get('organ_weights'),
                default_weight = c.get('organ_weight_default', 1.0),
            )
            if not c.get('organ_weights'):
                log.warning("use_hu_profile is on but organ_weights is None — every "
                            "organ gets the default weight. The term still works, but "
                            "the phase-critical vessels get no priority.")

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _l1_w(self) -> float:
        """Current lambda_l1 under the decay curriculum (constant if disabled).

        Zero under the heteroscedastic head, which REPLACES the L1 term with the
        Gaussian NLL. The trainer logs this into `history['lambda_l1']`, so
        returning the nominal 100 there would record a curriculum that never ran
        and make a hetero run look like an L1 run in the curves.
        """
        if self.use_hetero:
            return 0.0
        if not self.use_l1_decay or self._epoch <= self.l1_decay_start:
            return self.lambda_l1
        span = max(1, self.l1_decay_end - self.l1_decay_start)
        f = min(1.0, (self._epoch - self.l1_decay_start) / span)
        return self.lambda_l1 + f * (self.lambda_l1_floor - self.lambda_l1)

    def forward(
        self,
        pred:             torch.Tensor,
        target:           torch.Tensor,
        mask:             Optional[torch.Tensor] = None,
        log_var:          Optional[torch.Tensor] = None,
        adv_fake_logits:  Optional[torch.Tensor] = None,
        adv_weight:       Optional[float] = None,
        real_features:    Optional[List[torch.Tensor]] = None,
        fake_features:    Optional[List[torch.Tensor]] = None,
    ):
        """Returns (total_loss, loss_dict).

        `log_var` is required exactly when use_hetero is True; passing it to a
        non-heteroscedastic loss raises rather than being silently ignored, the
        same rule the generator applies to its conditioning inputs.

        `adv_fake_logits` / `*_features` come from the trainer's discriminator
        and are required exactly when the matching flag is on, for the same
        reason.
        """
        d: Dict[str, float] = {}
        total = pred.new_zeros(1).squeeze()

        # CONTRIBUTION CHANNELS. Every term also records `<name>_contrib` = the
        # lambda-SCALED value it actually added to `total`.
        #
        # This exists because the plain `<name>` entries are NOT in consistent
        # units and never have been: `l1` and `adversarial` are raw, while
        # `organ`, `ssim`, `gradient`, `frequency`, `hu_profile`, `nll` and
        # `feature_matching` are already lambda-scaled. Those units are frozen
        # here on purpose — changing them would silently redefine every curve in
        # every recorded history.json — but it makes the one question that
        # matters for tuning ("what fraction of the gradient is this term?")
        # unanswerable from the logs. `_contrib / train_gen_total` answers it.
        def _add(name: str, term, lam: float):
            """Accumulate `term` (already lambda-scaled) and record both views."""
            nonlocal total
            total = total + term
            v = float(term.detach())
            d[name] = v
            d[f'{name}_contrib'] = v
            return v

        if self.use_hetero:
            if log_var is None:
                raise ValueError("use_hetero=True requires `log_var` — the model's "
                                 "second output channel")
            nll = self.nll(pred, log_var, target) * self.lambda_nll
            _add('nll', nll, self.lambda_nll)
            d['l1'] = float(F.l1_loss(pred, target))   # logged, not optimised
            d['l1_contrib'] = 0.0
            d['lambda_l1'] = 0.0
        else:
            if log_var is not None:
                raise ValueError("`log_var` was given but use_hetero is False — it "
                                 "would be silently ignored.")
            _lam_l1 = self._l1_w()
            l1 = F.l1_loss(pred, target) * _lam_l1
            # RAW MAE, not the lambda-scaled contribution. The hetero branch above
            # logs the raw value and this branch used to log the scaled one, so
            # `history['train_l1']` changed units by ~100x depending on a flag and
            # `_plot_history` drew both on one axis. The scaled contribution is
            # still recoverable as train_l1 * lambda_l1, and `train_gen_total`
            # carries the optimised sum either way.
            d['l1'] = float(l1.detach()) / _lam_l1 if _lam_l1 else 0.0
            total = total + l1
            d['l1_contrib'] = float(l1.detach())
            d['lambda_l1'] = _lam_l1    # logged so the curriculum is auditable
            d['nll'] = 0.0
            d['nll_contrib'] = 0.0

        if self.use_adv:
            if adv_fake_logits is None:
                raise ValueError("use_adversarial=True requires `adv_fake_logits` "
                                 "— the discriminator's verdict on `pred`")
            if adv_weight is None:
                raise ValueError("use_adversarial=True requires `adv_weight` — "
                                 "the warmed-up lambda_adv, owned by the trainer")
            _lam_adv = float(adv_weight)
            raw = self.adv_loss.gen_loss(adv_fake_logits)
            # RAW value, not the lambda-scaled contribution — same rule as L1
            # above, so `train_adv` does not change units when the warmup ends,
            # and is still recorded while the warmup weight is still 0.
            d['adversarial'] = float(raw.detach())
            d['lambda_adv'] = _lam_adv
            total = total + raw * _lam_adv
            # The one term whose plain channel is raw AND whose lambda moves
            # (warmup), so the two views genuinely differ epoch to epoch.
            d['adversarial_contrib'] = float(raw.detach()) * _lam_adv
        else:
            if adv_fake_logits is not None:
                raise ValueError("`adv_fake_logits` was given but use_adversarial "
                                 "is False — it would be silently ignored.")
            d['adversarial'] = 0.0
            d['adversarial_contrib'] = 0.0
            d['lambda_adv'] = 0.0

        if self.use_fm:
            if fake_features is None:
                raise ValueError("use_feature_matching=True requires "
                                 "`fake_features` from the discriminator")
            _add('feature_matching',
                 self.fm(real_features or [], fake_features) * self.lambda_fm,
                 self.lambda_fm)
        else:
            d['feature_matching'] = d['feature_matching_contrib'] = 0.0

        if self.use_ssim:
            _add('ssim', self.ssim(pred, target) * self.lambda_ssim, self.lambda_ssim)
        else:
            d['ssim'] = d['ssim_contrib'] = 0.0

        if self.use_grad:
            _add('gradient', self.gradient(pred, target) * self.lambda_grad,
                 self.lambda_grad)
        else:
            d['gradient'] = d['gradient_contrib'] = 0.0

        if self.use_freq:
            _add('frequency', self.frequency(pred, target) * self.lambda_freq,
                 self.lambda_freq)
        else:
            d['frequency'] = d['frequency_contrib'] = 0.0

        if self.use_organ:
            _add('organ', self.organ(pred, target, mask) * self.lambda_organ,
                 self.lambda_organ)
        else:
            d['organ'] = d['organ_contrib'] = 0.0

        if self.use_hu_profile:
            _add('hu_profile',
                 self.hu_profile(pred, target, mask) * self.lambda_hu_profile,
                 self.lambda_hu_profile)
        else:
            d['hu_profile'] = d['hu_profile_contrib'] = 0.0

        d['total'] = total.item()
        return total, d
