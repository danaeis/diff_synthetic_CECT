"""
PatchGAN discriminator for the adversarial branch of this fork.

This is NOT a copy of `../synthetic_CECT/models.py`. It is the same 70x70
PatchGAN topology (identical block/stride/receptive-field layout, so the two are
comparable), with three deliberate changes, each of which exists because the
critic here judges the ONE-STEP x0 ESTIMATE OF A DIFFUSION MODEL rather than the
output of a deterministic generator:

1. TIMESTEP CONDITIONING (`use_t_cond`). The single most important difference.
   The fake shown to D is x0_hat = E[x0 | x_t], whose sharpness is a monotone
   function of t: near t=0 it is almost the target, near t=T it is a blurred
   conditional mean. An unconditional D therefore has a trivial, useless
   solution -- "blurry => fake" -- which is just a timestep regressor. The only
   way G can lower that loss is to hallucinate high-frequency texture at high t,
   i.e. exactly the failure this term is supposed to prevent. Conditioning D on
   t makes the question "is this a plausible x0 estimate AT THIS NOISE LEVEL",
   which is the question worth asking. Same argument as the D of Denoising
   Diffusion GANs (Xiao et al., ICLR 2022), which is conditioned on t.

   D owns its OWN `TimestepEmbedding` rather than reusing the generator's. The
   generator's embedding is a moving target during training; a critic whose
   conditioning input drifts under it is measuring two things at once.

2. GROUPNORM, NOT BATCHNORM (`norm='group'`). The reference D uses BatchNorm and
   is called on the real batch and the fake batch in SEPARATE forward passes, so
   each pass normalises by its own batch statistics. Real and fake then differ in
   their normalisation constants before a single weight is applied, and D can
   separate them from the statistics alone without ever looking at the content.
   With batch_size 8-16 patches this is not a subtlety. GroupNorm is per-sample
   and removes the leak entirely. 'batch' is still selectable to reproduce the
   reference behaviour.

3. SPECTRAL NORM (`spectral=True`). The dataset here is a few hundred volumes;
   a PatchGAN with 2.7M parameters memorises it and its gradient to G becomes
   noise. Spectral normalisation bounds D's Lipschitz constant and is the
   cheapest stabiliser that does not need a second backward pass (unlike R1,
   which is also available -- see `trainer_adv.AdversarialMixin`).

Everything else -- 2-D/3-D switching, the anisotropic 3-D stride, the optional
conditional (pix2pix) input pairing, intermediate-feature extraction for the
feature-matching loss -- matches the reference.

    forward(x, t=None, cond=None, return_features=False) -> (logits, features|None)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models import _conv
from models_diffusion import TimestepEmbedding

log = logging.getLogger(__name__)


def _disc_norm(kind: str, dims: int, nf: int) -> Optional[nn.Module]:
    """Normalisation layer for a discriminator block.

    'group' is the default for the real/fake batch-statistics reason in this
    module's docstring. 'none' is legitimate when spectral norm is on -- the two
    do overlapping jobs -- and is the cheapest configuration.
    """
    if kind == 'group':
        return nn.GroupNorm(min(32, nf), nf)
    if kind == 'batch':
        return getattr(nn, f'BatchNorm{dims}d')(nf)
    if kind == 'instance':
        return getattr(nn, f'InstanceNorm{dims}d')(nf, affine=True)
    if kind == 'none':
        return None
    raise ValueError(f"unknown discriminator norm {kind!r}")


class PatchGANDiscriminator(nn.Module):
    """2-D or 3-D 70x70 PatchGAN critic, optionally conditioned on t and phase.

    Args:
        dims:         2 or 3.
        ndf:          Base feature maps.
        n_layers:     Number of strided blocks (4 => the 70x70 receptive field).
        stride_3d:    Stride for the strided blocks when dims=3. (1,2,2) keeps
                      depth, which is what patch_depth 8-16 can afford.
        in_channels:  1 = unconditional (a pure texture critic). Set to
                      1 + n_input_slices and feed cat([source, image]) for
                      pix2pix's D(x, y), which judges whether the PAIR is
                      consistent rather than whether the image alone looks real.
        cond_dim:     Width of the conditioning space. >0 enables a projection
                      (projection-discriminator style) onto the deepest feature
                      map. Both the timestep embedding and the external
                      phase/level vector live in this space and are SUMMED, so
                      every caller stays agnostic about which sources are active
                      -- the same contract as `UNetGenerator.cond_vec`.
        use_t_cond:   Build an internal `TimestepEmbedding(cond_dim)`. Required
                      on the diffusion path; meaningless on the deterministic
                      one. Requires cond_dim > 0.
        norm:         'group' (default) | 'batch' | 'instance' | 'none'.
        spectral:     Wrap every conv in spectral normalisation.

    Input shapes:
        dims=2: (B, in_channels, H, W)
        dims=3: (B, in_channels, D, H, W)
    """

    def __init__(self, dims: int = 2, ndf: int = 64, n_layers: int = 4,
                 stride_3d=None, in_channels: int = 1, cond_dim: int = 0,
                 use_t_cond: bool = False, norm: str = 'group',
                 spectral: bool = True):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.use_t_cond = use_t_cond
        if use_t_cond and not cond_dim:
            raise ValueError("use_t_cond=True requires cond_dim > 0 -- the "
                             "timestep embedding has nowhere to project to")

        Conv = _conv(dims)
        sn = (nn.utils.parametrizations.spectral_norm if spectral else (lambda m: m))

        # Strided-block stride: 2 for 2-D, anisotropic tuple for 3-D. s1 is the
        # unit stride of the last two layers, which is what makes the receptive
        # field 70x70 rather than unbounded.
        if dims == 2:
            s, s1 = 2, 1
        else:
            s = stride_3d if stride_3d is not None else (1, 2, 2)
            s1 = tuple(1 for _ in s) if isinstance(s, tuple) else 1

        self.blocks = nn.ModuleList()

        # Block 0 -- never normalised (pix2pix): the first layer has to be able
        # to see absolute intensity, and in a CT task absolute intensity IS the
        # signal. A norm here would throw away the HU level D is meant to judge.
        self.blocks.append(nn.Sequential(
            sn(Conv(in_channels, ndf, 4, stride=s, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        nf = ndf
        for _ in range(1, n_layers - 1):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers = [sn(Conv(nf_prev, nf, 4, stride=s, padding=1))]
            nrm = _disc_norm(norm, dims, nf)
            if nrm is not None:
                layers.append(nrm)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.blocks.append(nn.Sequential(*layers))

        nf_prev, nf = nf, min(nf * 2, 512)
        layers = [sn(Conv(nf_prev, nf, 4, stride=s1, padding=1))]
        nrm = _disc_norm(norm, dims, nf)
        if nrm is not None:
            layers.append(nrm)
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.blocks.append(nn.Sequential(*layers))

        self.output_conv = sn(Conv(nf, 1, 4, stride=s1, padding=1))

        # Built strictly inside their `if`, the same RNG-draw reason as the
        # generator's FiLM: constructing an unused module would shift every
        # later init and silently change a run that has the flag off.
        if use_t_cond:
            self.t_emb = TimestepEmbedding(cond_dim=cond_dim)
        if cond_dim:
            self.cond_proj = nn.Linear(cond_dim, nf)
            # Zero-init so D STARTS as its unconditional self and grows the
            # conditioning as it earns it. The gradient w.r.t. the weight is
            # non-zero from step 1 (it is dL/dproj * cond), so this is a warm
            # start, not a dead branch.
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"PatchGANDiscriminator | dims={dims} | ndf={ndf} | stride={s} | "
                 f"in_ch={in_channels} | norm={norm} | spectral={spectral} | "
                 f"t_cond={use_t_cond} | cond_dim={cond_dim} | {n:.2f}M params")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        features: List[torch.Tensor] = []
        h = x
        for block in self.blocks:
            h = block(h)
            features.append(h)

        if self.cond_dim:
            c = None
            if self.use_t_cond:
                if t is None:
                    raise ValueError("discriminator was built with use_t_cond=True "
                                     "but no `t` was passed")
                c = self.t_emb(t)
            if cond is not None:
                c = cond if c is None else c + cond
            if c is None:
                raise ValueError("discriminator was built with cond_dim>0 but "
                                 "neither `t` nor `cond` was passed")
            # Projection conditioning: a per-channel bias derived from the
            # conditioning vector, added to the deepest feature map.
            # Rank-agnostic broadcast, same reason as FiLM.
            proj = self.cond_proj(c.to(h.dtype))
            h = h + proj.view(h.shape[0], -1, *([1] * (h.ndim - 2)))
        elif t is not None or cond is not None:
            raise ValueError("`t`/`cond` were given but the discriminator is "
                             "unconditional (cond_dim=0)")

        logits = self.output_conv(h)
        return (logits, features) if return_features else (logits, None)
