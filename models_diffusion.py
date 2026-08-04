"""
Pixel-space conditional diffusion for NCCT → CECT (DIFFUSION_PLAN.md §4/§5).

WHY PIXEL SPACE, NOT STABLE DIFFUSION
-------------------------------------
Two reasons, in §5's order. (1) The domain gap is large and the win is small: SD
is RGB natural images at 512^2, this is single-channel CT over a 600 HU window at
128^2, and the one thing this task needs — the absolute HU LEVEL — is exactly what
a natural-image prior cannot know. (2) Most of the model already exists. A
diffusion U-Net is `UNetGenerator` plus timestep conditioning, and timestep
conditioning IS FiLM, which `models.py` already implements and
`tests/test_phase_cond.py` already pins. `scripts/vae_fidelity.py` measures the
latent-space ceiling anyway, so the choice is recorded rather than assumed.

WHAT THIS FILE ADDS ON TOP OF models.py
---------------------------------------
`models.py` is FROZEN. Everything here imports from it and composes; nothing
edits it, so the deterministic baseline stays reproducible.

  NoiseSchedule    cosine schedule, 1000 train steps, plus the algebra to convert
                   between (x0, eps, v) at any t.
  DiffusionUNet    the same U-Net topology, conditioned on the NCCT by
                   channel-concatenation and on t by FiLM.
  DDIMSampler      eta=0 deterministic DDIM (the point estimate), eta=1 ancestral
                   (the spread). One class, both of §5's samplers.

THREE DELIBERATE DEPARTURES FROM UNetGenerator, each load-bearing
-----------------------------------------------------------------
1. The output head is a plain conv with NO activation. The parent's Sigmoid is
   correct for a [0,1] target and wrong for `v` or `eps`, which are unbounded and
   symmetric about zero; a Sigmoid there would clip half the target range and
   saturate its gradient exactly where the noise is largest.

2. FiLM is applied in the ENCODER as well as the decoder and bottleneck. The
   phase-conditioning design deliberately keeps the encoder phase-agnostic so one
   encoder learns anatomy from every phase's pairs — that argument does not
   transfer to t. The noise level changes what the input IS, so an encoder that
   cannot see t has to infer it from the statistics of its own input, which is
   precisely the thing that makes eps-prediction badly conditioned.

3. Everything runs in [-1, 1], not [0, 1]. `dataset.py` is untouched and still
   emits [0,1]; `to_model` / `from_model` convert at the boundary. Diffusion adds
   zero-mean noise, so a target centred on 0.5 would put the signal and the noise
   on different scales at every t and make the schedule's variances wrong.

PARAMETERISATION
----------------
`--parameterisation {v,x0}`, default v. §5 rules out eps: at high noise levels
eps-prediction is poorly conditioned for a task whose signal is a LEVEL, and the
error it makes on x0 is amplified by 1/sqrt(alpha_bar). Both supported options
give a well-scaled target at every t, and both recover x0 in closed form so the
organ losses attach directly to it (§5.4).

References: Ho et al. (DDPM), Song et al. (DDIM), Salimans & Ho (v-parameterisation),
Nichol & Dhariwal (cosine schedule), Saharia et al. (Palette/SR3 — the
channel-concatenation conditioning this copies), Ho & Salimans (classifier-free
guidance), Perez et al. (FiLM).
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models import FiLM, _DecBlock, _EncBlock, _conv, _convT, _drop, _norm, _pool

log = logging.getLogger(__name__)

PARAMETERISATIONS = ('v', 'x0')


# ---------------------------------------------------------------------------
# Domain conversion — dataset.py stays in [0,1], diffusion works in [-1,1]
# ---------------------------------------------------------------------------

def to_model(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] (dataset/HU-window domain) → [-1,1] (diffusion domain)."""
    return x01 * 2.0 - 1.0


def from_model(x11: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1]. Used before every organ loss and before writing HU."""
    return (x11 + 1.0) * 0.5


# ---------------------------------------------------------------------------
# Offset noise
# ---------------------------------------------------------------------------

def offset_noise(like: torch.Tensor, c: float,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """i.i.d. Gaussian noise plus a per-item constant of scale `c`.

    WHY THIS EXISTS, in the terms of this project.

    The quantity the whole fork is trying to sample is a per-case, spatially
    flat HU offset — the real aortic level spans 96-218 HU across cases, sd 24.4
    HU (analysis/BASELINE_REFERENCE.md §2). In a 128x128 patch that is exactly
    one degree of freedom: the patch mean. Two things then go wrong at once with
    plain `randn_like`:

    1. **The prior has almost no DC.** The mean of 128x128 unit white noise has
       sd 1/128 = 0.0078, and this module's domain is [-1,1] over a 600 HU
       window, i.e. 300 HU per unit — so 2.3 HU. At eta=0 the DDIM sample is a
       deterministic function of x_T, so 2.3 HU is the entire budget the sampler
       has for a quantity whose true spread is 24.4 HU. Roughly 10x short.
    2. **The loss barely sees it.** `F.mse_loss` is unweighted over spatial
       frequencies, so the DC direction is 1 of 16384 and contributes ~0.006% of
       the gradient. The model is never paid to get it right.

    Adding a per-item constant (Guttenberg's offset noise, the standard fix for
    "the model cannot produce very dark or very bright images") gives the DC
    channel a prior with sd `c` instead of 1/sqrt(HW), and raises its share of
    the MSE by the same factor.

    THIS MUST BE APPLIED AT BOTH TRAINING AND SAMPLING. The noise is no longer
    exactly N(0, I) — total variance is 1 + c^2 — which is a deliberate deviation
    from the schedule's assumption. Using it on one side only is a train/test
    mismatch, not a half-measure. `infer_volume.DiffusionPredictor._build_noise`
    is the sampling half, and it draws ONE constant for the whole padded volume
    rather than one per tile, so every tile of a case agrees on the level — which
    is the same coherence argument that makes the noise field volume-wide in the
    first place.

    `c=0` returns plain `randn_like` exactly, so the flag is a true no-op.
    """
    if generator is not None:
        n = torch.randn(like.shape, generator=generator, dtype=like.dtype,
                        device=like.device)
    else:
        n = torch.randn_like(like)
    if not c:
        return n
    # One scalar per (batch item, channel), broadcast over every spatial axis.
    shape = (like.shape[0], like.shape[1]) + (1,) * (like.ndim - 2)
    if generator is not None:
        off = torch.randn(shape, generator=generator, dtype=like.dtype,
                          device=like.device)
    else:
        off = torch.randn(shape, dtype=like.dtype, device=like.device)
    return n + c * off


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule(nn.Module):
    """Cosine alpha_bar schedule (Nichol & Dhariwal) with the (x0, eps, v) algebra.

    Cosine rather than the original linear schedule: a linear beta schedule
    destroys the signal too early at this resolution, spending most of its
    timesteps on inputs that are already indistinguishable from noise. That costs
    little on 256^2 natural images and a lot here, where the quantity being
        recovered is a low-frequency LEVEL that survives — and therefore has to be
    learned at — high noise.

    Buffers, not plain tensors, so `.to(device)` and the checkpoint move them.

    Convention throughout: `t` is a LongTensor (B,) in [0, n_steps), and
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps.
    """

    def __init__(self, n_steps: int = 1000, s: float = 0.008,
                 max_beta: float = 0.999):
        super().__init__()
        self.n_steps = int(n_steps)
        t = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64) / n_steps
        f = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-8)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=max_beta)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', ab.float())
        self.register_buffer('sqrt_ab', ab.sqrt().float())
        self.register_buffer('sqrt_1mab', (1.0 - ab).sqrt().float())

    def _g(self, buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Gather buf[t] and broadcast to (B, 1, 1, ...) for `ndim`-D tensors.

        Rank-agnostic for the same reason FiLM is: a literal view(B,1,1,1) works
        in 2-D and breaks silently the first time this runs with dims=3.
        """
        return buf.to(t.device)[t].reshape((-1,) + (1,) * (ndim - 1))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """The forward process: x_t from a clean x0 and a noise draw."""
        return (self._g(self.sqrt_ab, t, x0.ndim) * x0
                + self._g(self.sqrt_1mab, t, x0.ndim) * noise)

    def to_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
                  param: str) -> torch.Tensor:
        """What the network is trained to output at this t."""
        if param == 'x0':
            return x0
        if param == 'v':
            # v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0  (Salimans & Ho).
            # It interpolates between predicting eps at t=0 and -x0 at t=T, which
            # is why its target has unit-ish scale at EVERY t — the property that
            # makes a single MSE weight correct across the whole schedule.
            return (self._g(self.sqrt_ab, t, x0.ndim) * noise
                    - self._g(self.sqrt_1mab, t, x0.ndim) * x0)
        raise ValueError(f"unknown parameterisation {param!r}; "
                         f"expected one of {PARAMETERISATIONS}")

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar_t / (1 - alpha_bar_t), the signal-to-noise ratio at t."""
        ab = self.alphas_cumprod.to(t.device)[t]
        return ab / (1.0 - ab).clamp_min(1e-12)

    # Implicit weight that each parameterisation's plain MSE already places on
    # the x0 ERROR — the common currency for comparing objectives across t.
    #   ||d_eps||^2 = SNR      * ||d_x0||^2
    #   ||d_x0 ||^2 = 1        * ||d_x0||^2
    #   ||d_v  ||^2 = (SNR+1)  * ||d_x0||^2      (v = (sqrt(ab)*x_t - x0)/sqrt(1-ab))
    _IMPLICIT_X0_WEIGHT = {
        'x0': lambda snr: torch.ones_like(snr),
        'v':  lambda snr: snr + 1.0,
    }

    def snr_loss_weight(self, t: torch.Tensor, gamma: float,
                        param: str) -> torch.Tensor:
        """Per-timestep loss weight that flattens the objective in x0 space.

        Diffusion training is a multi-task problem across t and the tasks
        conflict. Measured on this schedule, the x0-error weight each plain MSE
        implies is:

            t              0        250      500     700     900     999
            plain v-MSE    2.4e+04  6.49     1.97    1.25    1.02    1.00
            plain x0-MSE   1        1        1       1       1       1

        So `--parameterisation v` — the default — over-weights t=0 by a factor of
        24,000 relative to t=999. That is the low-t domination min-SNR-gamma
        exists to fix, and it means `diff_v` and `diff_x0` were never optimising
        comparable objectives.

        WHERE THIS DEVIATES FROM Hang et al. 2023 (arXiv 2303.09556), and why.
        The published target is `min(SNR, gamma)` on the x0 error, which on this
        schedule is [5, 5, 0.97, 0.25, 0.024, 1e-8] — it clamps the low-t blowup
        but drives the weight at high t to essentially ZERO. For image-quality
        work that is fine; the high-t steps carry little visible detail. For THIS
        project it is backwards: the quantity being recovered is a case-level HU
        offset, which is decided at high t (v ~ -x0 there), and the published
        weighting would suppress exactly the timesteps that carry it.

        So the target used here is `min(SNR, gamma) / SNR`, which is flat at 1.0
        for every t with SNR <= gamma and decays only where SNR exceeds it:

            t              0        250      500     700     900     999
            target         2.1e-04  0.911    1       1       1       1

        Low-t domination is removed, high t keeps full weight. `gamma` is the SNR
        above which a timestep starts being discounted; 5.0 is the paper's value
        and is kept because it is the point where this schedule's SNR curve turns
        over, not out of deference.

        The returned multiplier divides out whatever the parameterisation already
        implies, so both parameterisations optimise the SAME objective and their
        losses become comparable. Returns a (B,) tensor; the caller broadcasts.
        """
        try:
            implicit = self._IMPLICIT_X0_WEIGHT[param](self.snr(t))
        except KeyError:
            raise ValueError(f"unknown parameterisation {param!r}; "
                             f"expected one of {PARAMETERISATIONS}") from None
        snr = self.snr(t)
        target = snr.clamp(max=gamma) / snr.clamp_min(1e-12)
        return target / implicit

    def predict_x0(self, out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
                   param: str, clip: bool = True) -> torch.Tensor:
        """Recover the denoised estimate x0 from the network's output.

        This is what the organ losses attach to (§5.4) and what the sampler
        iterates on. `clip` bounds it to the valid [-1,1] image range: the
        estimate at high t is extremely low-confidence and, left unbounded, an
        out-of-range x0 gets fed straight back into the next step and compounds.
        Standard practice, and it matters more here than usual because the
        quantity being estimated is a level.
        """
        if param == 'x0':
            x0 = out
        elif param == 'v':
            x0 = (self._g(self.sqrt_ab, t, x_t.ndim) * x_t
                  - self._g(self.sqrt_1mab, t, x_t.ndim) * out)
        else:
            raise ValueError(f"unknown parameterisation {param!r}")
        return x0.clamp(-1.0, 1.0) if clip else x0

    def predict_eps(self, out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
                    param: str) -> torch.Tensor:
        """Recover eps — needed by the DDIM update, which is written in eps."""
        if param == 'v':
            return (self._g(self.sqrt_1mab, t, x_t.ndim) * x_t
                    + self._g(self.sqrt_ab, t, x_t.ndim) * out)
        x0 = out
        return ((x_t - self._g(self.sqrt_ab, t, x_t.ndim) * x0)
                / self._g(self.sqrt_1mab, t, x_t.ndim).clamp_min(1e-8))

    def eps_from_x0(self, x0: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """eps implied by a (possibly clipped) x0 — keeps the sampler consistent
        with the clipping applied in predict_x0, which is otherwise silently
        discarded by an eps-space update."""
        return ((x_t - self._g(self.sqrt_ab, t, x_t.ndim) * x0)
                / self._g(self.sqrt_1mab, t, x_t.ndim).clamp_min(1e-8))


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal position embedding of t, then a 2-layer MLP to `cond_dim`.

    The MLP is what makes this usable by the existing `FiLM`, which expects a
    (B, cond_dim) conditioning vector and nothing else — so timestep, phase and
    level conditioning are literally the same code path downstream, as §5.2 says.
    """

    def __init__(self, cond_dim: int = 64, n_freq: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(n_freq, cond_dim * 4), nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.n_freq // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device,
                                              dtype=torch.float32) / half)
        a = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class DiffusionUNet(nn.Module):
    """Conditional diffusion U-Net: same topology as `UNetGenerator`, plus t.

    Args mirror `UNetGenerator` where they mean the same thing.

        cond_channels   channels of the NCCT condition. 1 for plain 2-D, or
                        `n_input_slices` under 2.5-D. The network's first conv
                        therefore takes 1 + cond_channels (§5.3).
        parameterisation 'v' (default) or 'x0'.
        cfg_drop_prob   probability of replacing the NCCT condition with a
                        LEARNED null embedding during training, enabling
                        classifier-free guidance at sampling time. §9 flags
                        guidance as an explicit dial on the var-ratio-vs-featHU
                        trade-off, which is exactly §7's tension; 0.1 is Ho &
                        Salimans' value and costs nothing at train time.
        use_phase_cond / n_levels — carried through so phase and level
                        conditioning remain available. They are OFF in every
                        planned run (COND_ORGANS=[] and splits/levels.json does
                        not exist in this fork), but the modules are still built
                        strictly inside their `if`, so leaving them off draws
                        exactly the RNG sequence a model without them would.

    forward(x_t, cond, t, phase=None, level=None, drop_cond=None) -> model output
    in the chosen parameterisation. All tensors are in the [-1,1] domain; use
    `to_model` / `from_model` at the boundary.
    """

    def __init__(
        self,
        dims:            int = 2,
        base_channels:   int = 64,
        dropout:         float = 0.2,
        pool_stride           = None,
        norm:            str = 'group',
        cond_channels:   int = 1,
        cond_dim:        int = 64,
        parameterisation: str = 'v',
        cfg_drop_prob:   float = 0.1,
        use_phase_cond:  bool = False,
        n_phases:        int = 2,
        n_levels:        int = 0,
    ):
        super().__init__()
        if parameterisation not in PARAMETERISATIONS:
            raise ValueError(f"parameterisation must be one of {PARAMETERISATIONS}, "
                             f"got {parameterisation!r}")
        self.dims = dims
        self.norm = norm
        self.cond_channels = cond_channels
        self.parameterisation = parameterisation
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.use_phase_cond = use_phase_cond
        self.n_phases = n_phases
        self.n_levels = n_levels
        in_channels = 1 + cond_channels

        ch = base_channels
        bn = ch * 8
        Conv = _conv(dims)
        ConvT = _convT(dims)
        Drop = _drop(dims)
        stride = 2 if dims == 2 else (pool_stride if pool_stride is not None else (1, 2, 2))

        # ── encoder / bottleneck / decoder: identical to UNetGenerator ───────
        self.enc1 = _EncBlock(dims, in_channels, ch, dropout=0.0, norm=norm)
        self.enc2 = _EncBlock(dims, ch,   ch * 2, dropout=0.0, norm=norm)
        self.enc3 = _EncBlock(dims, ch*2, ch * 4, dropout=0.0, norm=norm)
        self.enc4 = _EncBlock(dims, ch*4, ch * 8, dropout=0.0, norm=norm)
        self.pool = _pool(dims)(kernel_size=stride, stride=stride)

        self.bottleneck = nn.Sequential(
            Conv(bn, bn, 3, padding=1), _norm(dims, bn, norm), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
            Conv(bn, bn, 3, padding=1), _norm(dims, bn, norm), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
        )

        self.up4 = ConvT(bn,   ch*4, kernel_size=stride, stride=stride)
        self.dec4 = _DecBlock(dims, ch*4 + ch*8, ch*4, dropout, norm=norm)
        self.up3 = ConvT(ch*4, ch*2, kernel_size=stride, stride=stride)
        self.dec3 = _DecBlock(dims, ch*2 + ch*4, ch*2, dropout, norm=norm)
        self.up2 = ConvT(ch*2, ch,   kernel_size=stride, stride=stride)
        self.dec2 = _DecBlock(dims, ch + ch*2, ch, dropout, norm=norm)
        self.up1 = ConvT(ch,   ch,   kernel_size=stride, stride=stride)
        self.dec1 = _DecBlock(dims, ch + ch, ch, dropout, norm=norm)

        # NO activation — see this module's docstring, departure (1).
        self.out_conv = Conv(ch, 1, 1)

        # ── conditioning ────────────────────────────────────────────────────
        self.t_emb = TimestepEmbedding(cond_dim)
        if use_phase_cond:
            self.phase_emb = nn.Embedding(n_phases, cond_dim)
        if n_levels:
            self.level_proj = nn.Linear(n_levels, cond_dim)
            nn.init.zeros_(self.level_proj.weight)
            nn.init.zeros_(self.level_proj.bias)

        # FiLM everywhere, including the encoder — departure (2). Each site is
        # zero-init (see models.FiLM), so at step 0 every one of them is an exact
        # identity and the network starts as a plain U-Net on cat([x_t, ncct]).
        for name, c in [('enc1', ch), ('enc2', ch*2), ('enc3', ch*4), ('enc4', ch*8),
                        ('bottleneck', bn),
                        ('dec4', ch*4), ('dec3', ch*2), ('dec2', ch), ('dec1', ch)]:
            setattr(self, f'film_{name}', FiLM(cond_dim, c))

        # Learned null condition for classifier-free guidance. A LEARNED constant
        # rather than zeros: zeros are a valid NCCT patch value (mid-window HU
        # 100), so "conditioning dropped" and "this region is 100 HU" would be the
        # same input and the unconditional branch would be trained on a lie.
        self.null_cond = nn.Parameter(torch.zeros(1, cond_channels,
                                                  *([1] * dims)))

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"DiffusionUNet | dims={dims} | in_ch={in_channels} "
                 f"(1 noisy + {cond_channels} cond) | base_ch={base_channels} | "
                 f"norm={norm} | param={parameterisation} | "
                 f"cfg_drop={cfg_drop_prob} | {n:.2f}M params")

    # -----------------------------------------------------------------------
    def film_stats(self) -> dict:
        """Mean |gamma| per FiLM site, probed over the timestep range.

        Same role as `UNetGenerator.film_stats`: a site whose gamma stays at ~0
        learned to ignore its conditioning. For a diffusion model that would mean
        the network is denoising without knowing the noise level, which is a
        failure worth catching in history.json rather than in the samples.
        """
        with torch.no_grad():
            dev = self.out_conv.weight.device
            t = torch.linspace(0, 999, 8, device=dev).long()
            c = self.t_emb(t)
            out = {}
            for name in ('enc1', 'enc2', 'enc3', 'enc4', 'bottleneck',
                         'dec4', 'dec3', 'dec2', 'dec1'):
                gamma, _ = getattr(self, f'film_{name}').to_scale_shift(c).chunk(2, dim=1)
                out[f'gamma_{name}'] = float(gamma.abs().mean())
        return out

    def cond_vec(self, t: torch.Tensor,
                 phase: Optional[torch.Tensor] = None,
                 level: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Conditioning vector: timestep, plus phase/level if they are enabled.

        Summed into one vector, exactly as `UNetGenerator.cond_vec` does, so every
        downstream FiLM stays agnostic about which sources are active.
        """
        c = self.t_emb(t)
        if self.use_phase_cond:
            if phase is None:
                raise ValueError("use_phase_cond=True requires a `phase` tensor (B,)")
            c = c + self.phase_emb(phase.reshape(-1).long())
        if self.n_levels:
            if level is None:
                raise ValueError(f"n_levels={self.n_levels} requires a `level` "
                                 f"tensor (B, {self.n_levels})")
            c = c + self.level_proj(level.float())
        return c

    def _apply_cfg_dropout(self, cond: torch.Tensor,
                           drop: Optional[torch.Tensor]) -> torch.Tensor:
        """Replace dropped samples' NCCT with the learned null embedding.

        `drop` is a (B,) bool mask. When None and the module is in training mode,
        it is drawn from `cfg_drop_prob`; when None in eval mode nothing is
        dropped, so evaluation is never silently unconditional.
        """
        if drop is None:
            if not self.training or self.cfg_drop_prob <= 0:
                return cond
            drop = (torch.rand(cond.shape[0], device=cond.device)
                    < self.cfg_drop_prob)
        if not drop.any():
            return cond
        m = drop.reshape((-1,) + (1,) * (cond.ndim - 1)).to(cond.dtype)
        return cond * (1 - m) + self.null_cond.expand_as(cond) * m

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                phase: Optional[torch.Tensor] = None,
                level: Optional[torch.Tensor] = None,
                drop_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.cond_vec(t, phase, level)
        cond = self._apply_cfg_dropout(cond, drop_cond)
        x = torch.cat([x_t, cond], dim=1)

        e1 = self.film_enc1(self.enc1(x), c)
        e2 = self.film_enc2(self.enc2(self.pool(e1)), c)
        e3 = self.film_enc3(self.enc3(self.pool(e2)), c)
        e4 = self.film_enc4(self.enc4(self.pool(e3)), c)

        b = self.film_bottleneck(self.bottleneck(self.pool(e4)), c)

        d = self.up4(b);  d = self.film_dec4(self.dec4(torch.cat([d, e4], dim=1)), c)
        d = self.up3(d);  d = self.film_dec3(self.dec3(torch.cat([d, e3], dim=1)), c)
        d = self.up2(d);  d = self.film_dec2(self.dec2(torch.cat([d, e2], dim=1)), c)
        d = self.up1(d);  d = self.film_dec1(self.dec1(torch.cat([d, e1], dim=1)), c)
        return self.out_conv(d)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class DDIMSampler:
    """DDIM / ancestral sampling over a strided subsequence of the schedule.

    `eta` is §5's two samplers in one parameter:
      eta = 0   deterministic DDIM. The output is a fixed function of x_T, which
                is what makes the volume-wide noise field in infer_volume.py work
                — one field per sample means one coherent volume per sample.
      eta = 1   ancestral (DDPM-equivalent at full step count). More diverse per
                step, but the extra per-step noise is drawn independently PER
                TILE, so at eta=1 tiles decorrelate as sampling proceeds and the
                case-level coherence the noise field was chosen to provide is
                partly given back. Use eta=0 unless measuring what the extra
                stochasticity buys.

    `guidance` w applies classifier-free guidance:
        out = out_uncond + w * (out_cond - out_uncond)
    w = 1 is plain conditional sampling and costs one forward pass; w != 1 costs
    two. Raising w sharpens toward the conditional mean — i.e. it trades the
    variance ratio §7 leads with against featHU, which is exactly the dial §9
    describes. Sweep it; do not tune it.
    """

    def __init__(self, model: DiffusionUNet, schedule: NoiseSchedule,
                 n_steps: int = 25, eta: float = 0.0, guidance: float = 1.0,
                 clip_x0: bool = True):
        self.model = model
        self.sch = schedule
        self.n_steps = int(n_steps)
        self.eta = float(eta)
        self.guidance = float(guidance)
        self.clip_x0 = clip_x0

    def timesteps(self, device) -> torch.Tensor:
        """Descending subsequence of [0, T), always ending at 0.

        Uniform striding rather than a quadratic one: the quantity being recovered
        is a low-frequency level that is decided EARLY in the reverse process (at
        high t), so a schedule that crowds its steps near t=0 spends them refining
        texture that is already settled.
        """
        T = self.sch.n_steps
        ts = torch.linspace(T - 1, 0, self.n_steps, device=device).round().long()
        return torch.unique_consecutive(ts)

    def _model_out(self, x_t, cond, t_b, phase, level):
        if self.guidance == 1.0:
            return self.model(x_t, cond, t_b, phase, level)
        B = x_t.shape[0]
        none_ = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        all_ = ~none_
        o_c = self.model(x_t, cond, t_b, phase, level, drop_cond=none_)
        o_u = self.model(x_t, cond, t_b, phase, level, drop_cond=all_)
        return o_u + self.guidance * (o_c - o_u)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_T: torch.Tensor,
               phase: Optional[torch.Tensor] = None,
               level: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None,
               return_x0: bool = False):
        """Reverse the diffusion from a GIVEN x_T.

        x_T is an argument, never drawn here: infer_volume.py crops it from one
        volume-wide noise field so that overlapping tiles share noise and the
        whole volume is one coherent sample. Drawing it internally would silently
        destroy that (DIFFUSION_PLAN.md §6 step 6).

        Returns the sample in the [-1,1] domain, or (sample, x0_trajectory_final)
        if `return_x0`.
        """
        x = x_T
        ts = self.timesteps(x.device)
        B = x.shape[0]
        sch = self.sch
        for i, t in enumerate(ts):
            t_b = torch.full((B,), int(t), device=x.device, dtype=torch.long)
            out = self._model_out(x, cond, t_b, phase, level)
            x0 = sch.predict_x0(out, x, t_b, self.model.parameterisation,
                                clip=self.clip_x0)
            # eps re-derived FROM the clipped x0, not taken from the model: the
            # two disagree whenever the clip bit, and using the raw eps would
            # quietly discard the clip that was just applied.
            eps = sch.eps_from_x0(x0, x, t_b)

            ab_t = sch._g(sch.alphas_cumprod, t_b, x.ndim)
            if i + 1 < len(ts):
                t_prev = torch.full_like(t_b, int(ts[i + 1]))
                ab_prev = sch._g(sch.alphas_cumprod, t_prev, x.ndim)
            else:
                ab_prev = torch.ones_like(ab_t)

            # DDIM update (Song et al. eq. 12). sigma=0 at eta=0 makes it
            # deterministic; eta=1 recovers the ancestral variance.
            sigma = (self.eta * torch.sqrt((1 - ab_prev) / (1 - ab_t))
                     * torch.sqrt(1 - ab_t / ab_prev)) if self.eta > 0 \
                else torch.zeros_like(ab_t)
            dir_xt = torch.sqrt((1 - ab_prev - sigma ** 2).clamp_min(0.0)) * eps
            x = torch.sqrt(ab_prev) * x0 + dir_xt
            if self.eta > 0 and i + 1 < len(ts):
                noise = torch.empty_like(x).normal_(generator=generator)
                x = x + sigma * noise
        return (x, x0) if return_x0 else x


# ---------------------------------------------------------------------------
# Construction from a run config
# ---------------------------------------------------------------------------

def build_diffusion(cfg: dict) -> Tuple[DiffusionUNet, NoiseSchedule]:
    """Build (model, schedule) from a `run_config.json`-shaped dict.

    One place, used by both the trainer and inference, so a checkpoint can never
    be rebuilt with a schedule different from the one it was trained under.
    """
    _phases = cfg.get('target_phases') or [cfg.get('target_phase', 'venous')]
    model = DiffusionUNet(
        dims             = cfg.get('dims', 2),
        base_channels    = cfg['generator_base_channels'],
        dropout          = cfg.get('generator_dropout', 0.2),
        norm             = cfg.get('generator_norm', 'group'),
        cond_channels    = cfg.get('in_channels', 1),
        cond_dim         = cfg.get('phase_cond_dim', 64),
        parameterisation = cfg.get('parameterisation', 'v'),
        cfg_drop_prob    = cfg.get('cfg_drop_prob', 0.1),
        use_phase_cond   = cfg.get('use_phase_cond', False),
        n_phases         = max(2, len(_phases)),
        n_levels         = len(cfg.get('cond_organs') or []),
    )
    schedule = NoiseSchedule(n_steps=cfg.get('diffusion_steps', 1000))
    return model, schedule
