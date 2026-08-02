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
from models_diffusion import (DDIMSampler, build_diffusion, from_model, to_model)
from trainer import Trainer, _center_flat, _metric_set, _METRICS, _MIN_MASK_VOXELS

log = logging.getLogger(__name__)


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

        betas = config.get('betas', (0.5, 0.999))
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

        self.use_amp = config.get('use_mixed_precision', True) and self.device == 'cuda'
        self.scaler_G = GradScaler('cuda', enabled=self.use_amp)

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        from trainer import EarlyStopping
        self.early_stop = EarlyStopping(config.get('early_stop_patience', 30))
        self._warned_selection = False

        self.history: Dict[str, List] = {k: [] for k in [
            'epoch', 'lr_gen', 'lambda_l1',
            'train_gen_total', 'train_l1', 'train_nll',
            'train_ssim', 'train_grad', 'train_freq', 'train_organ', 'train_hu_profile',
            'val_loss',
            'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',
            'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc',
        ] + [f'gamma_{s}' for s in ('enc1', 'enc2', 'enc3', 'enc4', 'bottleneck',
                                    'dec4', 'dec3', 'dec2', 'dec1')]}

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
                 f"aux_max_t={self.aux_max_t} | "
                 f"sample every {self.sample_every} ep at {self.sample_steps} DDIM steps")
        log.info(f"Active losses: MSE({self.param})"
                 + (" + organ" if self.use_organ else "")
                 + (" + hu_profile" if self.use_hu_profile else ""))

    # -----------------------------------------------------------------------
    def _split_out(self, out):
        """Parent-contract shim: diffusion has no variance channel."""
        return out, None

    # -----------------------------------------------------------------------
    def _diffusion_loss(self, source, target, mask, phase, level, t, noise):
        """Shared by the train step and the deterministic val pass.

        Returns (total, dict). `t` and `noise` are passed in rather than drawn
        here precisely so validation can hold them fixed.
        """
        x0 = to_model(target)
        cond = to_model(source)
        x_t = self.schedule.q_sample(x0, t, noise)

        out = self.G(x_t, cond, t, phase, level)
        tgt = self.schedule.to_target(x0, noise, t, self.param)
        mse = F.mse_loss(out, tgt)
        total = mse
        d = {'mse': float(mse.detach()), 'organ': 0.0, 'hu_profile': 0.0}

        if (self.use_organ or self.use_hu_profile) and mask is not None:
            # Back to [0,1] so the organ losses see exactly the domain they were
            # written and weighted for — their lambdas were sized against
            # normalised HU (config.py's LAMBDA_HU_PROFILE comment), and feeding
            # them [-1,1] would double every residual.
            x0_hat = from_model(self.schedule.predict_x0(out, x_t, t, self.param))
            tgt01 = from_model(x0)
            keep = (t < self.aux_max_t)
            if keep.any():
                sel = keep.nonzero(as_tuple=True)[0]
                if self.use_organ:
                    o = self.organ_loss(x0_hat[sel], tgt01[sel], mask[sel]) * self.lambda_organ
                    d['organ'] = float(o.detach());  total = total + o
                if self.use_hu_profile:
                    h = self.hu_profile_loss(x0_hat[sel], tgt01[sel], mask[sel]) * self.lambda_hu_profile
                    d['hu_profile'] = float(h.detach());  total = total + h

        d['total'] = float(total.detach())
        return total, d

    # -----------------------------------------------------------------------
    def _train_step(self, batch: Dict) -> Dict:
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        mask   = batch['mask'].to(self.device) if 'mask' in batch else None
        phase  = self._phase(batch)
        level  = self._level(batch)

        B = source.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (B,), device=self.device)
        noise = torch.randn_like(target)

        self.opt_G.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            loss, d = self._diffusion_loss(source, target, mask, phase, level, t, noise)

        self.scaler_G.scale(loss).backward()
        self.scaler_G.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        self.global_step += 1
        return {
            'gen_total': d['total'],
            'l1':        d['mse'],           # the fidelity term, whatever its form
            'nll':       0.0,
            'ssim':      0.0, 'grad': 0.0, 'freq': 0.0,
            'organ':     d['organ'],
            'hu_profile': d['hu_profile'],
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
            noise = torch.randn(n_items, *shape, generator=g)
            self._val_fixed = {'n': n_items, 't': t, 'noise': noise}
        return self._val_fixed

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict:
        """Deterministic denoising loss + (periodically) sampled organ metrics.

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
                _, d = self._diffusion_loss(src, tgt, mask, self._phase(batch),
                                            self._level(batch), t, noise)
            losses.append(d['mse'])
            organ_losses.append(d['organ'])

        out = {
            'val_loss': float(np.mean(losses)) if losses else 0.0,
            'val_mae': 0.0, 'val_psnr': 0.0, 'val_ssim': 0.0, 'val_ncc': 0.0,
            'n_organ_items': 0,
        }
        for m in _METRICS:
            out[f'val_org_{m}'] = 0.0

        # Sampled metrics, occasionally — the expensive, informative half.
        if (self.current_epoch % max(1, self.sample_every) == 0
                or self.current_epoch == self.cfg.get('epochs', 0)):
            out.update(self._sampled_metrics(val_loader))
        elif self._last_sampled:
            # Carry the last sampled values forward so the curves are continuous
            # instead of dropping to zero between sampling epochs. They are
            # STALE, not fresh — never used for selection, which reads val_loss.
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
        for i in range(fake.shape[0]):
            p = _center_flat(fake[i]).astype(np.float64)
            t_ = _center_flat(tgt[i]).astype(np.float64)
            for m, v in _metric_set(p, t_).items():
                glob[m].append(v)
            if mask is not None:
                mk = _center_flat(mask[i]) > 0
                if mk.sum() >= _MIN_MASK_VOXELS:
                    for m, v in _metric_set(p[mk], t_[mk]).items():
                        org[m].append(v)

        def _mean(xs): return float(np.mean(xs)) if xs else 0.0
        res = {f'val_{m}': _mean(glob[m]) for m in _METRICS}
        res.update({f'val_org_{m}': _mean(org[m]) for m in _METRICS})
        res['n_organ_items'] = len(org['mae'])
        self._last_sampled = dict(res)
        # The sample grid is drawn from the same tensors, so it shows exactly the
        # patches these numbers describe.
        self._sampled_grid = (src, fake, tgt, idx, ds)
        return res

    # -----------------------------------------------------------------------
    def _selection_score(self, val: Dict) -> float:
        """MINIMISE the deterministic denoising loss.

        NOT `val_org_ssim`, the parent's default: that needs a deterministic
        forward pass, and the sampled version of it is only computed every
        `diffusion_sample_every` epochs, so selecting on it would compare fresh
        values against stale carried-forward ones. The denoising loss is
        available every epoch, is deterministic by construction (see
        `_fixed_val`), and is the standard diffusion selection signal.
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
