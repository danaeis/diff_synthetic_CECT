"""
The discriminator half of the training step, shared by both trainers.

`Trainer` (deterministic) and `DiffusionTrainer` differ completely in what they
produce as the "fake" image -- one forwards the generator, the other reads the
one-step x0 estimate off the denoiser -- but everything downstream of that is
identical: build D, pair it with the source, update it, freeze it, ask it for a
verdict on the fake, checkpoint it. That is what lives here.

DIFFERENCES FROM ../synthetic_CECT/trainer.py, WHICH THIS REPLACES
------------------------------------------------------------------
1. ONE GENERATOR FORWARD PER STEP, NOT TWO. The reference runs the generator
   once under `no_grad` to feed D and a second time with grad for the G step,
   with a comment claiming this "halves G forward-passes/step". It does the
   opposite: `no_grad` avoids building a graph, not the forward itself, so the
   step pays two full generator forwards where the standard pix2pix step pays
   one and detaches. On a diffusion U-Net -- the expensive object in the room,
   ~30x the discriminator -- that is close to a 2x slowdown for nothing. Here the
   fake is produced once and `.detach()` feeds D.

2. D IS FROZEN DURING THE G STEP. The reference leaves `requires_grad` on
   throughout, so G's backward also populates D's `.grad` with scaler_G-scaled
   garbage. It is not a correctness bug there only because `opt_D.zero_grad()`
   happens to run first next step -- i.e. it is one reordering away from
   training D on the wrong objective, and meanwhile it pays a full backward
   through D's parameters every step. `_frozen_d()` is the CycleGAN
   `set_requires_grad` idiom, made explicit.

3. FEATURE MATCHING IS CONSISTENT. Both real and fake features are taken from
   the SAME (post-update) D inside the G step. See `FeatureMatchingLoss`.

4. OPTIONAL R1. `lambda_r1 > 0` adds a zero-centred gradient penalty on real
   samples every `r1_every` D steps (lazy regularisation, Karras et al. 2019).
   Off by default; it is the knob to reach for when D's loss collapses toward
   zero in the first few epochs, which is the expected failure mode on a
   few-hundred-volume dataset.

5. DISC_UPDATE_FREQ IS HONEST. Under the reference, a `disc_update_freq > 1` run
   also switched the feature-matching term on and off with it. Here D's update
   cadence changes only D's update cadence.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from losses import AdversarialLoss, FeatureMatchingLoss
from models_disc import PatchGANDiscriminator

log = logging.getLogger(__name__)


class AdversarialMixin:
    """Discriminator ownership. Mixed into `Trainer`, inherited by every subclass.

    A trainer that never calls `_init_adversarial` still works: `self.D` is None
    and every method here is a no-op, which is what keeps the non-adversarial
    scenarios on exactly the code path (and RNG sequence) they had before.
    """

    D: Optional[nn.Module] = None
    opt_D: Optional[optim.Optimizer] = None
    use_adv: bool = False
    use_fm: bool = False

    # -----------------------------------------------------------------------
    def _init_adversarial(self, config: Dict, *, dims: int,
                          source_channels: int = 1,
                          cond_dim: int = 0,
                          use_t_cond: bool = False) -> None:
        """Build D, its optimiser and its AMP scaler. Call AFTER building G.

        Order matters: D is constructed strictly inside the `use_adversarial`
        branch and strictly after the generator, so a run with the flag off draws
        exactly the RNG sequence -- and therefore holds exactly the generator
        weights -- it drew before this file existed.

        `source_channels` is the channel count of the conditioning image (1, or
        `n_input_slices` under 2.5-D). `cond_dim` > 0 enables projection
        conditioning; `use_t_cond` adds D's own timestep embedding and is the
        diffusion path's requirement, not an option (see models_disc.py).
        """
        self.use_adv = bool(config.get('use_adversarial', False))
        self.use_fm  = bool(config.get('use_feature_matching', False))
        if not (self.use_adv or self.use_fm):
            self.D, self.opt_D = None, None
            return

        cond_d = bool(config.get('use_cond_disc', False))
        d_in = (1 + int(source_channels)) if cond_d else 1
        self.D = PatchGANDiscriminator(
            dims        = dims,
            ndf         = int(config.get('disc_ndf', 64)),
            in_channels = d_in,
            cond_dim    = cond_dim,
            use_t_cond  = use_t_cond,
            norm        = config.get('disc_norm', 'group'),
            spectral    = bool(config.get('disc_spectral', True)),
        ).to(self.device)

        # beta1 0.5 for D regardless of what G uses. The diffusion generator runs
        # at 0.9 because its objective is stationary; D's objective is not -- it
        # chases a generator that moves every step -- so it keeps the short
        # momentum window that GAN training was tuned with.
        self.opt_D = optim.Adam(
            self.D.parameters(),
            lr    = float(config.get('lr_disc', 1e-4)),
            betas = tuple(config.get('disc_betas') or (0.5, 0.999)),
        )
        self.scaler_D = GradScaler('cuda', enabled=self.use_amp)

        # D's half of the GAN objective. `CompositeLoss` builds its own instance
        # for the G half; both read `adv_mode` from the same config key and the
        # class is stateless, so the two halves cannot disagree. The diffusion
        # trainer has no CompositeLoss at all and uses this one for both halves.
        self.criterion_adv = AdversarialLoss(mode=config.get('adv_mode', 'lsgan'))
        self.fm_loss = FeatureMatchingLoss()

        self.lambda_adv  = float(config.get('lambda_adv', 2.0))
        self.lambda_fm   = float(config.get('lambda_fm', 10.0))
        self.adv_warmup  = int(config.get('adv_warmup_epochs', 5))
        self.disc_freq   = int(config.get('disc_update_freq', 1))
        self.lambda_r1   = float(config.get('lambda_r1', 0.0))
        self.r1_every    = max(1, int(config.get('r1_every', 16)))
        self._d_steps    = 0

        log.info(f"Adversarial | cond_disc={cond_d} (D in_ch={d_in}) | "
                 f"mode={config.get('adv_mode', 'lsgan')} | "
                 f"lambda_adv={config.get('lambda_adv', 2.0)} | "
                 f"warmup={config.get('adv_warmup_epochs', 5)} ep | "
                 f"lr_disc={config.get('lr_disc', 1e-4)} | "
                 f"D steps/G step=1/{self.disc_freq} | "
                 f"r1={self.lambda_r1} (every {self.r1_every})")

    # -----------------------------------------------------------------------
    def _adv_w(self) -> float:
        """Current lambda_adv under the linear warmup, 0 when the flag is off.

        Reads `self.current_epoch`, which the shared train loop sets, rather than
        a `set_epoch` hook: the diffusion trainer's `criterion` is a stub with no
        state, so a second epoch counter there would be one more thing that can
        silently fall out of step with the loop.

        The warmup is not cosmetic on the diffusion path. At epoch 1 the one-step
        x0 estimate at any t above ~200 is visibly not a CT, D reaches ~0 loss
        within an epoch, and an un-warmed adversarial gradient at that point is
        pure noise injected into a model that has not yet learned to denoise.
        """
        if not self.use_adv:
            return 0.0
        return self.lambda_adv * min(
            1.0, getattr(self, 'current_epoch', 0) / max(1, self.adv_warmup))

    # -----------------------------------------------------------------------
    def _d_in(self, img: torch.Tensor, source: Optional[torch.Tensor]) -> torch.Tensor:
        """Discriminator input: cat([source, img]) when D is conditional.

        Both tensors must already be in the SAME domain. On the diffusion path
        that is the [-1,1] model domain for both, not [0,1] for one of them --
        a mismatch there hands D a free feature that has nothing to do with
        realism.
        """
        if getattr(self.D, 'in_channels', 1) > 1:
            if source is None:
                raise ValueError("conditional discriminator (in_channels>1) but "
                                 "no `source` was passed")
            return torch.cat([source, img], dim=1)
        return img

    @contextmanager
    def _frozen_d(self):
        """Turn off grad for D's parameters for the duration of the G step.

        Without this the generator's backward walks D's weights as well: wasted
        compute, and `.grad` left holding values scaled by the WRONG GradScaler
        (scaler_G, not scaler_D) until the next `opt_D.zero_grad()` clears them.
        """
        for p in self.D.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p in self.D.parameters():
                p.requires_grad_(True)

    # -----------------------------------------------------------------------
    def _disc_step(self, real: torch.Tensor, fake: torch.Tensor,
                   source: Optional[torch.Tensor] = None,
                   t: Optional[torch.Tensor] = None,
                   cond: Optional[torch.Tensor] = None) -> float:
        """One discriminator update. `fake` is expected to be already detached.

        `t` and `cond` are passed IDENTICALLY to the real and the fake branch.
        Giving D the timestep for one and not the other, or two different phase
        vectors, would let it separate real from fake on the conditioning rather
        than on the image, which is the whole failure mode conditioning exists to
        avoid.
        """
        self.opt_D.zero_grad(set_to_none=True)
        d_real = self._d_in(real, source)
        d_fake = self._d_in(fake, source)

        with autocast('cuda', enabled=self.use_amp):
            logits_real, _ = self.D(d_real, t=t, cond=cond)
            logits_fake, _ = self.D(d_fake, t=t, cond=cond)
            loss_D = self.criterion_adv.disc_loss(logits_real, logits_fake)

        if self.lambda_r1 > 0 and (self._d_steps % self.r1_every == 0):
            # LAZY R1: the penalty is applied every r1_every steps and scaled by
            # r1_every, so its time-average weight is lambda_r1 at 1/r1_every of
            # the cost. Computed with autocast OFF: a squared gradient norm in
            # fp16 underflows to zero for exactly the small gradients the
            # penalty is meant to measure, so it would silently do nothing.
            real_in = d_real.detach().float().requires_grad_(True)
            with autocast('cuda', enabled=False):
                logits_r1, _ = self.D(real_in, t=t, cond=cond)
                g = torch.autograd.grad(logits_r1.sum(), real_in,
                                        create_graph=True)[0]
                r1 = g.pow(2).flatten(1).sum(1).mean()
            loss_D = loss_D + 0.5 * self.lambda_r1 * self.r1_every * r1

        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.unscale_(self.opt_D)
        nn.utils.clip_grad_norm_(self.D.parameters(), 10.0)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        self._d_steps += 1
        return float(loss_D.detach())

    # -----------------------------------------------------------------------
    def _disc_verdict(self, fake: torch.Tensor, real: Optional[torch.Tensor] = None,
                      source: Optional[torch.Tensor] = None,
                      t: Optional[torch.Tensor] = None,
                      cond: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]],
                                 Optional[List[torch.Tensor]]]:
        """D's verdict on `fake`, for the GENERATOR step. Returns
        (logits, real_features, fake_features).

        Called inside `_frozen_d()`; the returned logits are still attached to
        the generator's graph, which is the point. Real features (feature
        matching only) are recomputed here rather than reused from `_disc_step`
        so both sides come from the same D state -- see FeatureMatchingLoss.
        """
        want_feats = self.use_fm
        real_feats = None
        if want_feats:
            if real is None:
                raise ValueError("use_feature_matching=True needs `real`")
            with torch.no_grad():
                _, real_feats = self.D(self._d_in(real, source), t=t, cond=cond,
                                       return_features=True)
        logits, fake_feats = self.D(self._d_in(fake, source), t=t, cond=cond,
                                    return_features=want_feats)
        return logits, real_feats, fake_feats

    # -----------------------------------------------------------------------
    def _adv_checkpoint_state(self, state: Dict) -> None:
        """Add D's half to a checkpoint dict, in place. No-op without a D."""
        if self.D is None:
            return
        state['D_state'] = self.D.state_dict()
        state['opt_D']   = self.opt_D.state_dict()
        state['d_steps'] = self._d_steps
        if self.use_amp:
            state['scaler_D'] = self.scaler_D.state_dict()

    def _adv_load_state(self, state: Dict) -> None:
        """Restore D's half from a checkpoint dict. No-op without a D.

        A resume that silently reinitialised D would hand the generator a fresh,
        untrained critic mid-run -- a large, unlogged change in the objective
        that looks exactly like a training instability. So a checkpoint without
        `D_state` is loud about it rather than quiet.
        """
        if self.D is None:
            return
        if 'D_state' not in state:
            log.warning("use_adversarial is ON but the checkpoint has no D_state "
                        "— resuming with a RANDOMLY INITIALISED discriminator. "
                        "Expect a transient in the adversarial loss.")
            return
        self.D.load_state_dict(state['D_state'])
        self.opt_D.load_state_dict(state['opt_D'])
        self._d_steps = state.get('d_steps', 0)
        if self.use_amp and 'scaler_D' in state:
            self.scaler_D.load_state_dict(state['scaler_D'])
