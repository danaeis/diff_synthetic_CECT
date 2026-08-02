"""
Heteroscedastic generator — the calibration baseline of DIFFUSION_PLAN.md §6 step 2.

WHY THIS EXISTS
---------------
Diffusion is being built to fix one measured deficiency: the deterministic model
is 5.7x under-dispersed (`var(gen)/var(real) = 0.176`, target 1.0). Diffusion is
also 50-100x the sampling cost. Before paying that, the cheap alternative has to
be measured: predict mu and sigma per voxel, train with Gaussian NLL, and read the
uncertainty straight off the second output channel. That is one extra conv layer
and one loss term, and it reuses the current generator wholesale.

If diffusion cannot beat this on CRPS and coverage, it is not worth the cost.
Report the comparison either way — see DIFFUSION_PLAN.md §8.

WHAT IT CANNOT DO, STATED BEFORE THE RESULTS
--------------------------------------------
The predicted sigma is INDEPENDENT PER VOXEL. The spread that induces on an
organ's median HU, over ~1e5 voxels, is ~sigma/sqrt(n) — i.e. ~0. So on the
headline per-organ metrics this model will score a variance ratio near 0 and 90%
coverage near 0 however well it is trained, because a per-voxel independent noise
model has no case-level degree of freedom at all, and DIFFUSION_PLAN.md §2
measured the residual as exactly that: dose, bolus timing, cardiac output, one
draw per scan.

This is not a straw man to be quietly dropped. It is a real bar on the PER-VOXEL
CRPS/coverage rows, and the gap between its per-voxel and per-organ numbers is the
cleanest available statement of what per-voxel uncertainty can and cannot
represent. Both levels are reported by `scripts/calibration_eval.py`.

DESIGN
------
A subclass, not a flag on `UNetGenerator`, for the same reason phase conditioning
is built strictly inside its `if`: constructing an extra head unconditionally
would draw from the global RNG and shift the init of a plain run under a fixed
seed. A run with `use_hetero=False` therefore builds the untouched baseline class
and stays bitwise comparable to everything in `analysis/BASELINE_REFERENCE.md`.

Reference: Kendall & Gal, 2017 — "What Uncertainties Do We Need in Bayesian Deep
Learning for Computer Vision?", the aleatoric/epistemic decomposition §2 rests on.
"""

import logging

import torch
import torch.nn as nn

from models import UNetGenerator, _conv

log = logging.getLogger(__name__)


class HeteroGenerator(UNetGenerator):
    """`UNetGenerator` with a second output channel carrying log-variance.

    forward(...) -> (B, 2, ...) where channel 0 is mu and channel 1 is log sigma^2.
    Callers split it; `trainer.Trainer._split_out` is the single place that does.

    Signature is identical to `UNetGenerator`, so the trainer can swap the class
    without special-casing any argument.
    """

    def __init__(self, *args, log_var_init: float = -6.0, **kwargs):
        super().__init__(*args, **kwargs)
        dims = self.dims
        Conv = _conv(dims)
        ch = self.dec1.block[0].out_channels        # base_channels, whatever it was

        # mu keeps the parent's Sigmoid: targets are normalised to [0,1] by
        # dataset.py and the parent's docstring explains why Tanh is wrong here.
        self.mu_head = self.out_conv

        # log-variance head is LINEAR — a variance is not bounded to [0,1] and a
        # Sigmoid here would cap sigma at 1.0 while also saturating its gradient
        # exactly where the model is most confident.
        self.logvar_head = Conv(ch, 1, 1)
        nn.init.zeros_(self.logvar_head.weight)
        # Bias-init the variance LOW, not at zero. log_var=0 means sigma=1 in
        # [0,1] units — a 600 HU standard deviation, i.e. the model claiming it
        # knows nothing. The NLL's (target-mu)^2/exp(log_var) term is then ~0 and
        # the fidelity gradient vanishes for the first epochs while log_var walks
        # down on its own. -6 is sigma ~= 0.05, about 30 HU: the right order for
        # the ~13 HU error being modelled, and small enough that the squared-error
        # term dominates from step 1.
        nn.init.constant_(self.logvar_head.bias, float(log_var_init))

        # out_conv is REPLACED, not kept alongside: leaving it in place would put
        # its parameters in the optimiser with no gradient path, which shows up
        # later as a confusing "unused parameter" and inflates the param count.
        self.out_conv = nn.Identity()

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"HeteroGenerator | 2-channel head (mu, log_var) | "
                 f"log_var_init={log_var_init} | {n:.2f}M params")

    def forward(self, x, phase=None, level=None):
        # The parent's forward ends in `self.out_conv(d)`, now an Identity, so
        # what comes back is the last decoder feature map rather than an image.
        # That is the whole point: both heads read the same features.
        d = super().forward(x, phase, level)
        return torch.cat([self.mu_head(d), self.logvar_head(d)], dim=1)
