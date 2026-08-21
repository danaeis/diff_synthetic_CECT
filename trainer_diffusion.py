"""
Training loop for the conditional diffusion model (DIFFUSION_PLAN.md §6 step 5).

Subclasses `Trainer` so checkpointing, history, curves, per-organ metrics, early
stopping and the sample grid are shared verbatim. Only three things differ, and
each one is a decision rather than plumbing:

1. THE STEP. Draw t ~ U[0,T), draw noise, form x_t, regress the parameterisation
   target with MSE. Optionally add `OrganWeightedLoss` / `OrganHUProfileLoss` on
   the PREDICTED x0 — §5.4's point, and the reason `v`/`x0` were chosen over
   `eps`: x0 is recoverable in closed form at every t, so the losses that fixed
   the baseline's organ levels attach unchanged.

2. VALIDATION IS DETERMINISTIC. The obvious implementation — sample t and noise
   fresh each epoch — produces a val loss whose epoch-to-epoch variation is
   dominated by which timesteps happened to be drawn, not by whether the model
   improved. Selecting a checkpoint on that is selecting on noise. Instead each
   val patch is assigned a FIXED timestep from an even grid and a FIXED noise
   tensor, seeded once at construction, identical at every epoch and across
   resumes. This makes val_loss a genuine ordering of epochs.

3. SAMPLE QUALITY IS MEASURED SEPARATELY, AND RARELY. The denoising loss is a
   weak proxy for how good the samples are, but a full DDIM pass every epoch is
   real money on a single-GPU budget. Every `diffusion_sample_every` epochs a
   short DDIM sample runs on a few val patches to log `val_org_ssim` and draw the
   grid. Selection stays on the cheap deterministic loss; the sampled metric is
   there to be looked at, and to catch a model whose loss falls while its samples
   fall apart.

4. OPTIONAL ADVERSARIAL TERM ON THE PREDICTED x0 (`use_adversarial`).
   A discriminator needs an image, and a diffusion training step does not
   produce one — it produces a regression target at one random noise level.
   Running the sampler inside the step to get a real image is 25-1000 U-Net
   forwards per iteration and is not affordable here. The image this uses
   instead is the SAME one the organ losses already attach to: the closed-form
   one-step estimate x0_hat = predict_x0(out, x_t, t), which costs nothing
   because `out` has already been computed.

   Four things make that workable, and all four are load-bearing:

   a. D IS CONDITIONED ON t. x0_hat's sharpness is a monotone function of t, so
      an unconditional critic solves the task by regressing the noise level and
      the only way G can beat it is to invent texture at high t. See
      models_disc.py.
   b. THE TERM IS GATED TO t < `adv_max_t`. Same argument as `aux_max_t`, only
      stronger: at high t the posterior over x0 is wide, x0_hat is a genuine
      conditional MEAN, and "make the mean look like a sample" is a request to
      be overconfident.
   c. THE CLIP IS STRAIGHT-THROUGH, not `clamp`. Real x0 lives in [-1,1] and
      unclipped x0_hat does not, which would hand D a free discriminating
      feature that is about range rather than realism. A hard clamp fixes the
      range and kills the gradient on exactly the saturated voxels that need it
      — the mistake this repo already made once with the organ losses. So the
      forward value is clamped and the gradient passes through.
   d. λ_adv WARMS UP OVER `adv_warmup_epochs`. At epoch 1 an x0_hat at any
      moderate t is visibly not a CT; D wins immediately and its gradient is
      noise.

   THIS IS NOT FREE, AND THE COST IS THE ONE THIS THESIS MEASURES. Sharpening a
   posterior mean spends calibration. Report `var_ratio`, CRPS and coverage from
   `benchmark.py` for the adversarial run against its non-adversarial twin
   before claiming the term helped; a PSNR/SSIM table cannot see what it costs,
   and per-voxel metrics will in fact get WORSE if the term is working.

Everything is in the [-1,1] domain internally (`models_diffusion.to_model`); the
dataset and every metric stay in [0,1].
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from losses import OrganHUProfileLoss, OrganWeightedLoss
from models_diffusion import (DDIMSampler, build_diffusion, from_model,
                              offset_noise, to_model)
from trainer import (Trainer, CONTRIB_TERMS, _center_2d, _center_flat,
                     _detail_metric_set, _metric_set, _METRICS, _MIN_MASK_VOXELS)

log = logging.getLogger(__name__)

# Added to a provisional (val_loss-based) selection score so that ANY real
# detail score beats it. The detail score is |raps_hf - 1| + gradW1, both O(1),
# so 1e6 is unreachable from below.
_PROVISIONAL_SELECTION_OFFSET = 1e6


class DiffusionTrainer(Trainer):
    """Trainer for `DiffusionUNet`. See the module docstring for what differs."""

    def __init__(self, config: Dict):
        # Deliberately NOT calling Trainer.__init__: it builds a UNetGenerator and
        # a CompositeLoss, neither of which applies here, and building them just
        # to throw them away would draw from the global RNG and change this
        # model's init under a fixed seed.
        self.cfg = config
        self.device = config['device']
        from pathlib import Path
        self.out = Path(config['output_dir'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples = self.out / 'samples'
        self.samples.mkdir(exist_ok=True)

        self.use_hetero = False          # the parent's _split_out contract
        self.param = config.get('parameterisation', 'v')
        # The shared train loop calls criterion.set_epoch() and logs
        # criterion._l1_w() into history['lambda_l1']. Diffusion has no L1
        # curriculum, so this stub keeps the inherited loop intact and records a
        # constant 0 rather than a number that would look like a live schedule.
        self.criterion = _NoCurriculum()

        self.G, self.schedule = build_diffusion(config)
        self.G = self.G.to(self.device)
        self.schedule = self.schedule.to(self.device)

        # beta1 0.9, not the project-wide 0.5. `BETAS = (0.5, 0.999)` is inherited
        # from the GAN baseline, where a short momentum window keeps the generator
        # responsive to a moving discriminator. There is no discriminator here:
        # the diffusion objective is stationary and its per-step gradient is
        # dominated by which timestep was drawn, so a longer momentum window is
        # what averages that away. 0.9 is the DDPM/ADM standard.
        # The project-wide `betas` key is NOT consulted here — it is always
        # present, so falling back to it would silently keep 0.5. Set
        # `diffusion_betas` (or --diffusion_beta1) to reproduce the old runs.
        betas = tuple(config.get('diffusion_betas') or (0.9, 0.999))
        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr           = config.get('learning_rate', 2e-4),
            betas        = betas,
            weight_decay = config.get('weight_decay', 1e-5),
        )
        if config.get('use_cosine_schedule', True):
            self.sched_G = sched.CosineAnnealingWarmRestarts(
                self.opt_G,
                T_0    = config.get('cosine_t0', 15),
                T_mult = config.get('cosine_tmult', 2),
                eta_min= config.get('cosine_eta_min', 5e-7),
            )
        else:
            self.sched_G = None

        # ── auxiliary losses on the predicted x0 ────────────────────────────
        # Built strictly inside their `if`, same RNG-draw reason as everywhere
        # else in this codebase.
        self.use_organ = config.get('use_organ', False)
        self.lambda_organ = config.get('lambda_organ', 20.0)
        if self.use_organ:
            self.organ_loss = OrganWeightedLoss(
                organ_weight      = config.get('organ_weight', 10.0),
                organ_weights     = config.get('organ_weights'),
                default_weight    = config.get('organ_weight_default', 1.0),
                background_weight = config.get('organ_weight_background', 1.0),
            ).to(self.device)
        self.use_hu_profile = config.get('use_hu_profile', False)
        self.lambda_hu_profile = config.get('lambda_hu_profile', 50.0)
        if self.use_hu_profile:
            self.hu_profile_loss = OrganHUProfileLoss(
                organ_weights  = config.get('organ_weights'),
                default_weight = config.get('organ_weight_default', 1.0),
            ).to(self.device)

        # x0-space auxiliary losses are only meaningful where x0 is actually
        # recoverable. At very high t the estimate is dominated by the clip and
        # its gradient is noise; weighting it in anyway is a known way to make a
        # diffusion model blurry. Applied below `aux_max_t` only.
        self.aux_max_t = int(config.get('aux_max_t',
                                        int(0.7 * config.get('diffusion_steps', 1000))))

        # Whether the gated aux terms share the MSE's per-sample accounting.
        #
        # THE PROBLEM. `mse` averages over the WHOLE batch. The aux terms average
        # over the sub-batch that passed `t < aux_max_t` and are then added at
        # full lambda. So a term whose gate admits 70% of the batch contributes
        # as if it had been computed on every item — its effective weight is
        # inflated by 1/0.7 ~ 1.43x — and because |sel| is a fresh binomial draw
        # each step, that inflation is also NOISY, varying batch to batch.
        # Nothing about the direction of the gradient is wrong; the scale is.
        #
        # Default False so every lambda in config.py keeps the meaning it had
        # when it was tuned. Switching it on is a ~1.43x change to lambda_organ
        # and lambda_hu_profile, which is a real intervention and belongs in its
        # own ablation, not folded silently into another run.
        self.aux_gate_normalise = bool(config.get('aux_gate_normalise', False))

        # Adversarial term on the predicted x0 — see the module docstring, §4.
        # The discriminator itself is built further down, after `use_amp` exists,
        # because its GradScaler needs it.
        # `or`, not a .get default: ADV_MAX_T is present-and-None in config.py,
        # meaning "derive it", which a plain default would never see.
        self.adv_max_t = int(config.get('adv_max_t')
                             or 0.5 * config.get('diffusion_steps', 1000))
        self.adv_clip_mode = config.get('adv_clip_mode', 'straight_through')
        if self.adv_clip_mode not in ('straight_through', 'hard', 'none'):
            raise ValueError(f"unknown adv_clip_mode {self.adv_clip_mode!r}")

        # Scale of the per-item DC component added to the training noise. See
        # models_diffusion.offset_noise for the 2.3-HU-vs-24.4-HU argument. Must
        # match what infer_volume.DiffusionPredictor uses at sampling time, which
        # is why it is recorded in run_config.json and read back from there.
        self.offset_noise = float(config.get('diffusion_offset_noise', 0.0))

        # 0 disables. See NoiseSchedule.min_snr_weight.
        self.min_snr_gamma = float(config.get('min_snr_gamma', 0.0))

        # 0 disables. Every val pass, every sample grid and the checkpoint's
        # G_state use the averaged weights; the live weights are kept in
        # G_raw_state so a resume is still exact. See trainer.WeightEMA.
        from trainer import WeightEMA
        ema_decay = float(config.get('ema_decay', 0.0))
        self.ema = WeightEMA(self.G, ema_decay) if ema_decay > 0 else None

        self.use_amp = config.get('use_mixed_precision', True) and self.device == 'cuda'
        self.scaler_G = GradScaler('cuda', enabled=self.use_amp)

        # ── discriminator (optional) ────────────────────────────────────────
        # Built strictly inside its flag and after the generator, the same
        # RNG-draw reason as everything else here. `use_t_cond=True` is not
        # optional on this path — see the module docstring, §4a. `cond_dim` is
        # the width of the generator's own conditioning space, which is what
        # `_cond_for_d` produces for the phase/level half of D's conditioning.
        self._init_adversarial(
            config,
            dims            = config.get('dims', 2),
            source_channels = config.get('in_channels', 1),
            cond_dim        = config.get('phase_cond_dim', 64),
            use_t_cond      = True,
        )
        if self.D is not None and self.adv_max_t >= config.get('diffusion_steps', 1000):
            log.warning(f"adv_max_t={self.adv_max_t} covers the whole schedule — "
                        f"the critic will be asked to judge x0 estimates at t~T, "
                        f"where they are conditional means, not samples. See "
                        f"trainer_diffusion's module docstring, §4b.")

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        from trainer import EarlyStopping
        self.early_stop = EarlyStopping(config.get('early_stop_patience', 30))
        self._warned_selection = False
        # 'detail' | 'val_loss'. See _selection_score for why the default moved
        # off the denoising MSE. `selection_metric` is not consulted on this
        # path at all — it names deterministic-only metrics.
        self.selection_mode = config.get('diffusion_selection', 'detail')
        self._have_detail_score = False

        self.history: Dict[str, List] = {k: [] for k in [
            'epoch', 'lr_gen', 'lambda_l1',
            'train_gen_total', 'train_l1', 'train_nll',
            'train_ssim', 'train_grad', 'train_freq', 'train_organ', 'train_hu_profile',
            'train_adv', 'train_fm', 'train_disc', 'lambda_adv',
            'val_loss',
            'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',
            'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc',
            'val_raps_hf', 'val_grad_w1', 'val_org_grad_w1', 'val_sampled',
        ] + [
            # Same contract as the base trainer — see trainer.CONTRIB_TERMS. On
            # this path `train_l1_contrib` is the diffusion MSE (no lambda), so
            # it is the denominator the organ / hu_profile / adv shares are read
            # against. That ratio is what sizes lambda_organ, and it is the
            # number that would have shown at a glance that lambda_adv=0.5
            # against an MSE of order 1e-2 still makes the critic dominant.
            f'train_{k}_contrib' for k in CONTRIB_TERMS
        ] + [f'gamma_{s}' for s in ('enc1', 'enc2', 'enc3', 'enc4', 'bottleneck',
                                    'dec4', 'dec3', 'dec2', 'dec1')]}
        self._hist_keys = list(self.history)

        self.per_organ = config.get('report_per_organ_metrics', False)
        self.organ_id_to_name = self._load_organ_id_map(config.get('organ_label_map_json'))
        self.per_organ_history: List[Dict] = []

        # Deterministic validation — see the module docstring, point 2.
        self._val_seed = int(config.get('diffusion_val_seed', 1234))
        self._val_fixed: Optional[Dict[int, torch.Tensor]] = None

        self.sample_every = int(config.get('diffusion_sample_every', 10))
        self.sample_steps = int(config.get('diffusion_sample_steps', 25))
        self._last_sampled: Dict = {}

        n_ts = config.get('diffusion_steps', 1000)
        log.info(f"DiffusionTrainer | param={self.param} | T={n_ts} | "
                 f"aux_max_t={self.aux_max_t}"
                 f"{' (gate-normalised)' if self.aux_gate_normalise else ''} | "
                 f"min_snr_gamma={self.min_snr_gamma or 'off'} | "
                 f"ema={ema_decay or 'off'} | "
                 f"sample every {self.sample_every} ep at {self.sample_steps} DDIM steps")
        log.info(f"Active losses: MSE({self.param})"
                 + (" + organ" if self.use_organ else "")
                 + (" + hu_profile" if self.use_hu_profile else "")
                 + (f" + adv(x0_hat, t<{self.adv_max_t}, "
                    f"clip={self.adv_clip_mode})" if self.use_adv else "")
                 + (" + feature_matching" if self.use_fm else ""))

    # -----------------------------------------------------------------------
    def _split_out(self, out):
        """Parent-contract shim: diffusion has no variance channel."""
        return out, None

    # -----------------------------------------------------------------------
    def _diffusion_loss(self, source, target, mask, phase, level, t, noise):
        """Shared by the train step and the deterministic val pass.

        Returns (total, dict, x0_hat) where x0_hat is the one-step estimate in
        the MODEL domain [-1,1], or None when nothing asked for it. The train
        step needs it for the adversarial term and must not recompute it — a
        second `predict_x0` would be free of arithmetic error but would silently
        diverge the moment the parameterisation changed in one place only.

        `t` and `noise` are passed in rather than drawn here precisely so
        validation can hold them fixed.
        """
        x0 = to_model(target)
        cond = to_model(source)
        x_t = self.schedule.q_sample(x0, t, noise)

        out = self.G(x_t, cond, t, phase, level)
        tgt = self.schedule.to_target(x0, noise, t, self.param)
        if self.min_snr_gamma:
            # Flattens the objective in x0 space, so both parameterisations
            # optimise the same thing and the low-t steps stop dominating. On
            # this schedule plain v-MSE weights the x0 error at t=0 24,000x more
            # than at t=999 — and t=999 is where the case-level HU offset is
            # decided. See NoiseSchedule.snr_loss_weight, including why this is
            # NOT the published min-SNR-gamma target.
            w = self.schedule.snr_loss_weight(t, self.min_snr_gamma, self.param)
            w = w.reshape((-1,) + (1,) * (out.ndim - 1))
            mse = (w * (out - tgt) ** 2).mean()
        else:
            mse = F.mse_loss(out, tgt)
        total = mse
        # `*_contrib` mirrors CompositeLoss: the lambda-scaled value actually
        # added to `total`, so "what share of the gradient is this term?" is
        # answerable from history.json. The diffusion MSE carries no lambda, so
        # its contrib equals its raw value and it is the denominator everything
        # else should be read against.
        d = {'mse': float(mse.detach()), 'mse_contrib': float(mse.detach()),
             'organ': 0.0, 'organ_contrib': 0.0,
             'hu_profile': 0.0, 'hu_profile_contrib': 0.0}

        # clip=False, unlike the sampler. `predict_x0` defaults to clamping to
        # [-1,1], which is right when the estimate is fed back into the next DDIM
        # step but wrong in a LOSS: clamp() has zero gradient outside the range,
        # so every saturated voxel silently contributes nothing. The adversarial
        # branch re-applies the bound as a STRAIGHT-THROUGH clamp instead, which
        # keeps D's input in range without throwing the gradient away.
        need_x0 = (self.use_organ or self.use_hu_profile
                   or self.D is not None)
        x0_hat_m = (self.schedule.predict_x0(out, x_t, t, self.param, clip=False)
                    if need_x0 else None)

        if (self.use_organ or self.use_hu_profile) and mask is not None:
            # Back to [0,1] so the organ losses see exactly the domain they were
            # written and weighted for — their lambdas were sized against
            # normalised HU (config.py's LAMBDA_HU_PROFILE comment), and feeding
            # them [-1,1] would double every residual.
            #
            # The unclipped estimate matters here specifically: at moderate-to-
            # high t a large fraction of x0_hat is saturated, so a clamp would
            # mask the aux gradient far more aggressively than the documented
            # t < aux_max_t gate — and the organ losses are the only terms in the
            # diffusion objective that know about anatomy at all.
            x0_hat = from_model(x0_hat_m)
            tgt01 = from_model(x0)
            keep = (t < self.aux_max_t)
            if keep.any():
                sel = keep.nonzero(as_tuple=True)[0]
                # See __init__: without this the aux terms average over `sel`
                # while the MSE averages over the whole batch, so their effective
                # weight is inflated by 1/P(t < aux_max_t) and jitters with the
                # binomial draw of |sel|. `gate_w` puts both on per-sample terms.
                gate_w = (sel.numel() / t.numel()) if self.aux_gate_normalise else 1.0
                if self.use_organ:
                    o = self.organ_loss(x0_hat[sel], tgt01[sel], mask[sel]) * self.lambda_organ * gate_w
                    d['organ'] = d['organ_contrib'] = float(o.detach())
                    total = total + o
                if self.use_hu_profile:
                    h = self.hu_profile_loss(x0_hat[sel], tgt01[sel], mask[sel]) * self.lambda_hu_profile * gate_w
                    d['hu_profile'] = d['hu_profile_contrib'] = float(h.detach())
                    total = total + h

        d['total'] = float(total.detach())
        return total, d, x0_hat_m

    # -----------------------------------------------------------------------
    def _adv_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Bound x0_hat to the valid image range for the critic's benefit.

        'straight_through' (default): the value D sees is clamped to [-1,1] but
        the gradient passes through unchanged. Real x0 is in [-1,1] by
        construction and unclipped x0_hat is not, so without a bound D can
        separate real from fake on RANGE, which has nothing to do with realism;
        with a plain clamp the saturated voxels — the ones actually out of range,
        i.e. exactly the ones the term should be correcting — receive zero
        gradient. This is the same trap the organ losses fell into.
        'hard' reproduces the naive clamp, 'none' disables the bound; both exist
        to be ablated against, not used.
        """
        if self.adv_clip_mode == 'none':
            return x
        if self.adv_clip_mode == 'hard':
            return x.clamp(-1.0, 1.0)
        return x + (x.clamp(-1.0, 1.0) - x).detach()

    def _cond_for_d(self, phase, level) -> Optional[torch.Tensor]:
        """Phase/level conditioning for the DISCRIMINATOR — always DETACHED, and
        deliberately WITHOUT the timestep.

        D has its own timestep embedding (models_disc.py), so reusing the
        generator's `cond_vec` here would sum t in twice, from two different
        embeddings, one of which is being trained by a different objective.

        The detach is not an optimisation. D is a critic; letting its gradients
        reach the generator's phase/level embedding would let G lower the
        adversarial loss by reshaping what it *claims* was requested instead of
        by improving the image.
        """
        if not getattr(self.D, 'cond_dim', 0):
            return None
        parts = []
        if getattr(self.G, 'use_phase_cond', False) and phase is not None:
            parts.append(self.G.phase_emb(phase.reshape(-1).long()))
        if getattr(self.G, 'n_levels', 0) and level is not None:
            parts.append(self.G.level_proj(level.float()))
        if not parts:
            return None
        return sum(parts).detach()

    # -----------------------------------------------------------------------
    def _adversarial_term(self, source, target, x0_hat_m, t, phase, level):
        """The GAN half of the step. Returns (adv_loss_or_None, log_dict).

        Everything is in the MODEL domain [-1,1] — both D's image input and, when
        the D is conditional, the NCCT channel it is paired with. Feeding one in
        [0,1] and the other in [-1,1] would give D a constant offset between the
        two channels that separates nothing and confuses everything.
        """
        # NaN, not 0.0: these channels mean "not measured on this step". A tight
        # `adv_max_t` makes the empty-gate case the COMMON one (at adv_max_t=T/10
        # and batch 8 it is ~43% of steps), and averaging those in as zeros would
        # scale train_adv by the firing rate and read as a decaying loss. See the
        # nanmean in `Trainer.train`.
        nan = float('nan')
        out = {'adv': nan, 'fm': nan, 'disc': nan, 'd_real': nan, 'd_fake': nan, 'n_adv': 0}

        # Gate to the low-t half of the schedule (§4b). The SAME subset goes to
        # both D and G, and the same `t` goes to D's real and fake branches, so
        # the critic can never separate them on their conditioning.
        sel = (t < self.adv_max_t).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None, out
        out['n_adv'] = int(sel.numel())

        real_m = to_model(target)[sel]
        cond_m = to_model(source)[sel]
        fake_m = self._adv_bound(x0_hat_m[sel])
        t_sel  = t[sel]
        dcond  = self._cond_for_d(phase, level)
        if dcond is not None:
            dcond = dcond[sel]

        # D's update, and therefore D's backward, runs OUTSIDE any autocast
        # region — `_disc_step` opens its own for the forward. Calling backward
        # under autocast is what the PyTorch AMP docs tell you not to do, and it
        # is easy to get wrong here because the caller is mid-step.
        if self.global_step % self.disc_freq == 0:
            out['disc'], out['d_real'], out['d_fake'] = self._disc_step(
                real_m, fake_m.detach(), source=cond_m, t=t_sel, cond=dcond)

        # D is frozen here: the generator's backward has no business touching
        # D's weights, and leaving it unfrozen would fill D's .grad with
        # scaler_G-scaled values between its own updates.
        term = None
        lam = self._adv_w()
        with autocast('cuda', enabled=self.use_amp), self._frozen_d():
            logits, real_f, fake_f = self._disc_verdict(
                fake_m, real=real_m, source=cond_m, t=t_sel, cond=dcond)
            if self.use_adv:
                raw = self.criterion_adv.gen_loss(logits)
                # RAW value, not the lambda-scaled contribution, so `train_adv`
                # does not change units the moment the warmup finishes — and so
                # it is still recorded during epoch 0 of the warmup, where the
                # term is measured but weighted to nothing.
                out['adv'] = float(raw.detach())
                # The scaled view, which is what competes with the diffusion MSE
                # and the organ terms. Reading `adv` alone is what let a critic
                # sit at 0.95 for 80 epochs while nobody noticed it was 0.5*0.95
                # against an MSE of order 1e-2 — i.e. dominant and useless.
                out['adv_contrib'] = float(raw.detach()) * lam
                if lam > 0:
                    term = raw * lam
            if self.use_fm:
                fm = self.fm_loss(real_f or [], fake_f) * self.lambda_fm
                out['fm'] = out['fm_contrib'] = float(fm.detach())
                term = fm if term is None else term + fm
        return term, out

    # -----------------------------------------------------------------------
    def _train_step(self, batch: Dict) -> Dict:
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        mask   = batch['mask'].to(self.device) if 'mask' in batch else None
        phase  = self._phase(batch)
        level  = self._level(batch)

        B = source.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (B,), device=self.device)
        noise = offset_noise(target, self.offset_noise)

        self.opt_G.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            loss, d, x0_hat_m = self._diffusion_loss(
                source, target, mask, phase, level, t, noise)

        adv_log = {'adv': 0.0, 'fm': 0.0, 'disc': 0.0, 'd_real': 0.0, 'd_fake': 0.0}
        if self.D is not None:
            # ONE denoiser forward per step: `x0_hat_m` above already carries the
            # generator's graph, and `.detach()` inside is what feeds D. The
            # reference GAN trainer runs its generator a second time under
            # no_grad for this; on a diffusion U-Net that would be close to a 2x
            # slowdown for no benefit. See trainer_adv's docstring, point 1.
            # Not wrapped in autocast here — `_adversarial_term` opens it around
            # the forwards only, so D's backward stays outside one.
            adv_term, adv_log = self._adversarial_term(
                source, target, x0_hat_m, t, phase, level)
            if adv_term is not None:
                loss = loss + adv_term
                d['total'] = float(loss.detach())

        self.scaler_G.scale(loss).backward()
        self.scaler_G.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        if self.ema is not None:
            # After the step, so the average is over post-update iterates.
            self.ema.update(self.G)

        self.global_step += 1
        return {
            'gen_total': d['total'],
            'l1':        d['mse'],           # the fidelity term, whatever its form
            'nll':       0.0,
            'ssim':      0.0, 'grad': 0.0, 'freq': 0.0,
            'organ':     d['organ'],
            'hu_profile': d['hu_profile'],
            'adv':       adv_log['adv'],
            'fm':        adv_log['fm'],
            'disc':      adv_log['disc'],
            'd_real':    adv_log['d_real'],
            'd_fake':    adv_log['d_fake'],
            # Lambda-scaled twins. `l1_contrib` is the diffusion MSE, which
            # carries no lambda — it is the denominator the others are read
            # against, not a term to tune.
            'l1_contrib':         d['mse_contrib'],
            'nll_contrib':        0.0,
            'ssim_contrib':       0.0, 'grad_contrib': 0.0, 'freq_contrib': 0.0,
            'organ_contrib':      d['organ_contrib'],
            'hu_profile_contrib': d['hu_profile_contrib'],
            'adv_contrib':        adv_log.get('adv_contrib', 0.0),
            'fm_contrib':         adv_log.get('fm_contrib', 0.0),
        }

    # -----------------------------------------------------------------------
    def _fixed_val(self, n_items: int, shape, device):
        """Fixed (t, noise) per validation item — the same every epoch.

        `t` walks an EVEN grid over [0, T) rather than being sampled: with a few
        thousand val patches a uniform sample already covers the schedule, but an
        even grid makes the val loss reproducible to the last digit across
        resumes and across machines, which is what makes small epoch-to-epoch
        differences readable.
        """
        if self._val_fixed is None or self._val_fixed['n'] != n_items:
            g = torch.Generator(device='cpu').manual_seed(self._val_seed)
            t = torch.linspace(0, self.schedule.n_steps - 1, n_items).round().long()
            # Same noise distribution as the train step, offset included: a val
            # loss computed against plain randn while training uses offset noise
            # measures a distribution the model is not being fit to.
            noise = offset_noise(torch.empty(n_items, *shape), self.offset_noise,
                                 generator=g)
            self._val_fixed = {'n': n_items, 't': t, 'noise': noise}
        return self._val_fixed

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict:
        """Deterministic denoising loss + (periodically) sampled organ metrics.

        Runs under the EMA weights when EMA is on, so the curves, the sample
        grids and the checkpoint that inference loads all describe one set of
        weights. Validating the live iterate while shipping the average would
        mean selecting a checkpoint on a model that is never used.
        """
        if self.ema is not None:
            with self.ema.swapped(self.G):
                return self._validate_weights(val_loader)
        return self._validate_weights(val_loader)

    @torch.no_grad()
    def _validate_weights(self, val_loader: DataLoader) -> Dict:
        """The body of `_validate`, on whatever weights are currently loaded.

        The returned dict keeps the parent's key names so `_update_history`,
        `_selection_score` and the plots need no special-casing. `val_loss` is
        the denoising MSE, NOT an image MAE — the two are not comparable across
        model families, which is why the diffusion runs are never put in the same
        `val_loss` column as the deterministic ones.
        """
        self.G.eval()
        losses, organ_losses = [], []
        n_seen = 0
        fixed = None
        for batch in val_loader:
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device) if 'mask' in batch else None
            B = src.shape[0]
            if fixed is None:
                fixed = self._fixed_val(len(val_loader.dataset), tgt.shape[1:], self.device)
            sl = slice(n_seen, n_seen + B)
            t = fixed['t'][sl].to(self.device)
            noise = fixed['noise'][sl].to(self.device)
            n_seen += B
            if t.numel() != B:                     # last partial batch guard
                continue
            with autocast('cuda', enabled=self.use_amp):
                _, d, _ = self._diffusion_loss(src, tgt, mask, self._phase(batch),
                                               self._level(batch), t, noise)
            losses.append(d['mse'])
            organ_losses.append(d['organ'])

        out = {
            'val_loss': float(np.mean(losses)) if losses else 0.0,
            'val_mae': 0.0, 'val_psnr': 0.0, 'val_ssim': 0.0, 'val_ncc': 0.0,
            'val_raps_hf': 0.0, 'val_grad_w1': 0.0, 'val_org_grad_w1': 0.0,
            'n_organ_items': 0,
            'val_sampled': 0.0,
        }
        for m in _METRICS:
            out[f'val_org_{m}'] = 0.0

        # Sampled metrics, occasionally — the expensive, informative half.
        if (self.current_epoch % max(1, self.sample_every) == 0
                or self.current_epoch == self.cfg.get('epochs', 0)):
            out.update(self._sampled_metrics(val_loader))
            out['val_sampled'] = 1.0
        elif self._last_sampled:
            # Carry the last sampled values forward so the curves are continuous
            # instead of dropping to zero between sampling epochs. They are
            # STALE, not fresh — `val_sampled` stays 0.0 so `_plot_history` draws
            # them as a faint connecting line and puts a marker only on the
            # epochs actually measured. Without that flag these rows read as
            # 10-epoch plateaus in curves.png that no model behaviour produced.
            out.update(self._last_sampled)

        self.G.train()
        return out

    @torch.no_grad()
    def _sampled_metrics(self, val_loader: DataLoader, n_items: int = 16) -> Dict:
        """Run a short DDIM sample on a few val patches and score it.

        Fixed patches and a fixed x_T (seeded once), so the number moves only
        when the model does. Patch-level, not volume-level: this is a training
        curve, and the volume numbers come from infer_volume.py + benchmark.py.
        """
        ds = val_loader.dataset
        n = min(n_items, len(ds))
        idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
        batch = torch.utils.data.default_collate([ds[i] for i in idx])
        src = batch['source'].to(self.device)
        tgt = batch['target'].to(self.device)
        mask = batch.get('mask')

        g = torch.Generator(device='cpu').manual_seed(self._val_seed + 1)
        x_T = torch.randn(tgt.shape, generator=g).to(self.device)
        sampler = DDIMSampler(self.G, self.schedule, n_steps=self.sample_steps,
                              eta=0.0, guidance=1.0)
        with autocast('cuda', enabled=self.use_amp):
            x = sampler.sample(to_model(src), x_T,
                               self._phase(batch), self._level(batch))
        fake = from_model(x.float()).clamp(0, 1)

        glob = {m: [] for m in _METRICS}
        org = {m: [] for m in _METRICS}
        d_pred, d_tgt, d_mask = [], [], []
        for i in range(fake.shape[0]):
            d_pred.append(_center_2d(fake[i]))
            d_tgt.append(_center_2d(tgt[i]))
            p = _center_flat(fake[i]).astype(np.float64)
            t_ = _center_flat(tgt[i]).astype(np.float64)
            for m, v in _metric_set(p, t_).items():
                glob[m].append(v)
            if mask is not None:
                d_mask.append(_center_2d(mask[i]))
                mk = _center_flat(mask[i]) > 0
                if mk.sum() >= _MIN_MASK_VOXELS:
                    for m, v in _metric_set(p[mk], t_[mk]).items():
                        org[m].append(v)

        def _mean(xs): return float(np.mean(xs)) if xs else 0.0
        res = {f'val_{m}': _mean(glob[m]) for m in _METRICS}
        res.update({f'val_org_{m}': _mean(org[m]) for m in _METRICS})
        res['n_organ_items'] = len(org['mae'])
        # The detail axis, on the DDIM sample rather than a forward pass — a
        # one-step denoise would score as pure blur regardless of the model.
        res.update(_detail_metric_set(d_pred, d_tgt, d_mask or None))
        self._last_sampled = dict(res)
        # The sample grid is drawn from the same tensors, so it shows exactly the
        # patches these numbers describe.
        self._sampled_grid = (src, fake, tgt, idx, ds)
        return res

    # -----------------------------------------------------------------------
    def _selection_label(self) -> str:
        if self.selection_mode == 'detail':
            return 'selection: |raps_hf-1| + org gradW1 (sampled epochs only)'
        return 'selection: val_loss (denoising MSE)'

    def _selection_score(self, val: Dict) -> float:
        """MINIMISE, on sampling epochs only under `diffusion_selection=detail`.

        `val_loss` — the denoising MSE — is available every epoch and is
        deterministic by construction (`_fixed_val`), which is why it was the
        original choice. But it is a per-voxel loss, and every per-voxel loss is
        minimised by the conditional mean: selecting on it picks the epoch whose
        samples are BLURREST. That is the defect this project is trying to fix,
        so it cannot also be the checkpoint rule.

        `detail` mode scores what the samples actually look like:

            |raps_hf - 1|  +  val_org_grad_w1

        raps_hf is a ratio (1.0 = the real CECT's high-frequency amplitude; <1
        blur, >1 hallucinated noise) so it is scored as a distance from 1. The
        organ-region gradW1 is a distance already, and is restricted to the
        anatomy that matters rather than to the body wall and air that dominate
        a whole-patch metric. Both come from the DDIM sample, so both exist only
        on sampling epochs — a stale row returns +inf and is simply not a
        candidate, rather than being compared against a fresh one.

        Early stopping does NOT use this (see `_early_stop_score`): patience has
        to count on a signal that exists every epoch.
        """
        if self.selection_mode != 'detail':
            return float(val['val_loss'])

        raps = val.get('val_raps_hf') or 0.0
        gw1 = val.get('val_org_grad_w1') or val.get('val_grad_w1') or 0.0
        if val.get('val_sampled') and raps > 0.0:
            self._have_detail_score = True
            return abs(raps - 1.0) + float(gw1)

        if self._have_detail_score:
            return float('inf')                  # stale row: not a candidate

        # No sampled epoch has happened YET. Returning +inf here would mean a run
        # whose sampling cadence never fires (diffusion_sample_every > epochs, or
        # a crash before the first sample) finishes with no best_model.pth at all
        # — and infer_volume.py requires that file, so the run would be dead
        # without ever having said so. Fall back to the denoising loss, offset far
        # enough that the first REAL detail score always supersedes it.
        if not self._warned_selection:
            log.warning(
                "diffusion_selection='detail' but no DDIM sample has been taken "
                "yet — best_model.pth is provisionally selected on val_loss and "
                "will be replaced at the first sampling epoch. If "
                f"diffusion_sample_every ({self.sample_every}) exceeds the epoch "
                "count, that never happens and selection stays on val_loss.")
            self._warned_selection = True
        return _PROVISIONAL_SELECTION_OFFSET + float(val['val_loss'])

    def _early_stop_score(self, val: Dict, selection_score: float) -> float:
        """Patience counts against the denoising loss, which exists every epoch.

        Under `detail` selection the selection score is +inf on the ~90% of
        epochs that carry no fresh sample; counting patience against that would
        stop every run after `early_stop_patience` epochs regardless of how it
        was training.
        """
        return float(val['val_loss'])

    # -----------------------------------------------------------------------
    def _save_samples(self, val_loader: DataLoader, epoch: int):
        """Draw the grid from the last DDIM sample rather than a forward pass.

        The parent's version calls `self.G(src)` directly, which for a diffusion
        model is one denoising step from pure noise — a grey blur that says
        nothing. When no sample has been taken yet this epoch, skip the grid
        entirely rather than draw a misleading one.
        """
        import matplotlib.pyplot as plt
        got = getattr(self, '_sampled_grid', None)
        if got is None:
            return
        src, fake, tgt, idx, ds = got
        n = src.shape[0]
        case_ids = getattr(ds, 'case_ids', None)

        def _mid(t):
            a = t.detach().float().cpu().squeeze().numpy()
            return a[a.shape[0] // 2] if a.ndim == 3 else a

        fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
        if n == 1:
            axes = axes[None]
        for i in range(n):
            s, f, t_ = _mid(src[i]), _mid(fake[i]), _mid(tgt[i])
            err = np.abs(f - t_)
            for j, (img, kw) in enumerate([
                (s,   dict(cmap='gray', vmin=0, vmax=1)),
                (f,   dict(cmap='gray', vmin=0, vmax=1)),
                (t_,  dict(cmap='gray', vmin=0, vmax=1)),
                (err, dict(cmap='inferno', vmin=0,
                           vmax=self.cfg.get('sample_err_vmax', 0.15))),
            ]):
                ax = axes[i, j]
                ax.imshow(img, **kw)
                if i == 0:
                    ax.set_title(['NCCT', 'DDIM sample', 'Real CECT', '|error|'][j],
                                 fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            m = _metric_set(f.ravel(), t_.ravel())
            axes[i, 0].set_ylabel(f"{case_ids[idx[i]]}" if case_ids else f"patch {idx[i]}",
                                  fontsize=7)
            axes[i, 3].set_xlabel(f"PSNR {m['psnr']:.1f}   SSIM {m['ssim']:.3f}",
                                  fontsize=8)
        plt.suptitle(f'Epoch {epoch}  (DDIM {self.sample_steps} steps, eta=0)',
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(self.samples / f'ep{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()
        self._sampled_grid = None

        keep = self.cfg.get('keep_last_n_sample_epochs', 5)
        imgs = sorted(self.samples.glob('ep*.png'), key=lambda p: p.stat().st_mtime)
        for old in imgs[:-keep]:
            old.unlink(missing_ok=True)


class _NoCurriculum:
    """Stand-in for `CompositeLoss` where the shared loop only needs its two
    scheduling hooks. Not a loss: the diffusion loss lives in
    `DiffusionTrainer._diffusion_loss`, which the overridden `_train_step` calls
    directly."""

    def set_epoch(self, epoch: int):
        pass

    def _l1_w(self) -> float:
        return 0.0
