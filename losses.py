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
`AdversarialLoss`, `FeatureMatchingLoss`, `PerceptualLoss`, `DinoPerceptualLoss`,
`PhaseSaliencyLoss`, `DinoSaliencyLoss`, `CyclicConsistencyLoss` and
`SegmentationConsistencyLoss` are gone. Diffusion does not use a discriminator,
and the VGG/DINO backbones need a network download the training host cannot do.
The originals are in `../synthetic_CECT/losses.py`, which is frozen and still
reproduces every number in `analysis/BASELINE_REFERENCE.md`.

WHAT IS NEW HERE
----------------
`GaussianNLLLoss` — the heteroscedastic baseline of DIFFUSION_PLAN.md §6 step 2,
the bar diffusion has to clear on per-voxel calibration.

`OrganWeightedLoss` and `OrganHUProfileLoss` are unchanged and are reused by the
diffusion trainer, applied to the predicted x0 (§5.4).
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


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
    """
    L1 loss on FFT amplitude spectrum.
    2-D: fft2 over H×W.
    3-D: fftn over H×W per slice (spatial frequency target), averaged over D.
    """

    def _amp(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            return torch.abs(torch.fft.fft2(x2d, norm='ortho'))
        return torch.abs(torch.fft.fft2(x, norm='ortho'))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._amp(pred), self._amp(target))


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
    use_l1_decay is on), lambda_organ=20, lambda_hu_profile=50.

    WHAT CHANGED IN THIS FORK. `use_adversarial`, `use_perceptual`,
    `use_feature_matching`, `use_saliency`, `use_cycle` and `use_seg_consistency`
    are gone along with their terms — see this module's docstring. lambda_l1 is
    consequently no longer parametric on which of them is active: nothing left in
    this file trades pixel fidelity for realism, so L1 always starts at
    `lambda_l1` and `lambda_l1_reduced` is unused. The L1 DECAY CURRICULUM is
    kept, because that is what gives a zero entry in ORGAN_WEIGHTS any force.
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
            self.frequency = FrequencyLoss()

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
    ):
        """Returns (total_loss, loss_dict).

        `log_var` is required exactly when use_hetero is True; passing it to a
        non-heteroscedastic loss raises rather than being silently ignored, the
        same rule the generator applies to its conditioning inputs.
        """
        d: Dict[str, float] = {}
        total = pred.new_zeros(1).squeeze()

        if self.use_hetero:
            if log_var is None:
                raise ValueError("use_hetero=True requires `log_var` — the model's "
                                 "second output channel")
            nll = self.nll(pred, log_var, target) * self.lambda_nll
            d['nll'] = nll.item();  total = total + nll
            d['l1'] = float(F.l1_loss(pred, target))   # logged, not optimised
            d['lambda_l1'] = 0.0
        else:
            if log_var is not None:
                raise ValueError("`log_var` was given but use_hetero is False — it "
                                 "would be silently ignored.")
            _lam_l1 = self._l1_w()
            l1 = F.l1_loss(pred, target) * _lam_l1
            d['l1'] = l1.item();  total = total + l1
            d['lambda_l1'] = _lam_l1    # logged so the curriculum is auditable
            d['nll'] = 0.0

        if self.use_ssim:
            ssim = self.ssim(pred, target) * self.lambda_ssim
            d['ssim'] = ssim.item();  total = total + ssim
        else:
            d['ssim'] = 0.0

        if self.use_grad:
            grad = self.gradient(pred, target) * self.lambda_grad
            d['gradient'] = grad.item();  total = total + grad
        else:
            d['gradient'] = 0.0

        if self.use_freq:
            freq = self.frequency(pred, target) * self.lambda_freq
            d['frequency'] = freq.item();  total = total + freq
        else:
            d['frequency'] = 0.0

        if self.use_organ:
            org = self.organ(pred, target, mask) * self.lambda_organ
            d['organ'] = org.item();  total = total + org
        else:
            d['organ'] = 0.0

        if self.use_hu_profile:
            hup = self.hu_profile(pred, target, mask) * self.lambda_hu_profile
            d['hu_profile'] = hup.item();  total = total + hup
        else:
            d['hu_profile'] = 0.0

        d['total'] = total.item()
        return total, d
