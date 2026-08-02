"""
Trainer for the DETERMINISTIC NCCT → CECT models — the L1/organ baseline and the
heteroscedastic (mu, log sigma^2) baseline of DIFFUSION_PLAN.md §6 step 2.

The diffusion model has its own trainer, `trainer_diffusion.py`, because its
training step (sample t, add noise, regress the parameterisation target) has
nothing in common with this one. Everything downstream of the step — checkpoints,
history, curves, sample grids, early stopping — is shared by inheritance.

Batch format (from CTPairDataset):
    {'source': Tensor(B, 1, H, W)  or (B, 1, D, H, W),
     'target': Tensor(B, 1, H, W)  or (B, 1, D, H, W)}

The trainer is completely independent of the dimensionality of the patches —
it delegates all shape-awareness to the models and losses.

FORK NOTE: the PatchGAN discriminator branch (`_disc_step`, `_d_in`,
`_cond_for_d`, opt_D, scaler_D) was removed along with the adversarial losses.
It is still in `../synthetic_CECT/trainer.py` if an adversarial run is ever needed
again.
"""

import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import CompositeLoss
from models import UNetGenerator
from models_hetero import HeteroGenerator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().squeeze().numpy()


def _psnr(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _ssim(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = pred.mean(), target.mean()
    s1, s2   = pred.std(), target.std()
    s12      = np.mean((pred - mu1) * (target - mu2))
    return float(((2*mu1*mu2 + C1) * (2*s12 + C2)) /
                 ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)))


def _psnr_ssim(pred: torch.Tensor, target: torch.Tensor):
    """Compute PSNR and SSIM on first item. For 3-D tensors use centre slice."""
    p = _np(pred[0])
    t = _np(target[0])
    if p.ndim == 3:                        # (D, H, W) — take centre slice
        mid = p.shape[0] // 2
        p, t = p[mid], t[mid]
    # Fixed data_range=1.0: both tensors are globally normalised to [0, 1] by
    # dataset.py (see hu_min/hu_max clip + rescale), so PSNR/SSIM must use
    # that same fixed range. Using the per-sample target min/max instead (as
    # this used to) makes the metric incomparable across patches/epochs —
    # a near-uniform patch (small true dynamic range) would get an arbitrary
    # data_range that has nothing to do with the network's actual output scale.
    return _psnr(p, t, 1.0), _ssim(p, t, 1.0)


# Metrics reported by _validate, and the minimum voxel count for a masked
# (organ-region) metric to be meaningful (std/NCC need a few voxels).
_METRICS = ('mae', 'psnr', 'ssim', 'ncc')
_MIN_MASK_VOXELS = 16


def _center_flat(t: torch.Tensor) -> np.ndarray:
    """A single tensor item (1,H,W) or (1,D,H,W) → 1-D voxel array. For 3-D the
    centre slice is taken (same convention as _psnr_ssim), so pred/target/mask
    are always flattened over the identical voxels."""
    a = _np(t)
    if a.ndim == 3:                         # (D, H, W) — centre slice
        a = a[a.shape[0] // 2]
    return a.ravel()


def _metric_set(p: np.ndarray, t: np.ndarray, data_range: float = 1.0) -> Dict[str, float]:
    """MAE, PSNR, SSIM, NCC over a set of paired voxel values (1-D arrays).

    SSIM/NCC use the global (non-windowed) single-value formulae so they apply
    unchanged to an arbitrary voxel subset (e.g. an organ mask), not just a
    rectangular image — the windowed SSIM used elsewhere can't be restricted to
    a mask cleanly."""
    diff = p - t
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = 100.0 if mse == 0 else float(10.0 * np.log10(data_range ** 2 / mse))
    mu1, mu2 = float(p.mean()), float(t.mean())
    s1, s2 = float(p.std()), float(t.std())
    s12 = float(np.mean((p - mu1) * (t - mu2)))
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim = float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                 ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2)))
    ncc = float(s12 / (s1 * s2 + 1e-8))
    return {'mae': mae, 'psnr': psnr, 'ssim': ssim, 'ncc': ncc}


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best     = float('inf')
        self.counter  = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training loop for the literature baseline.

    Works for 2-D (patch_depth=1, dims=2) and 3-D (patch_depth>1, dims=3)
    without any code change — the models and losses handle the difference.
    """

    def __init__(self, config: Dict):
        self.cfg        = config
        self.device     = config['device']
        self.out        = Path(config['output_dir'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples    = self.out / 'samples'
        self.samples.mkdir(exist_ok=True)

        dims = config.get('dims', 2)

        # ── models ──────────────────────────────────────────────────────────
        # n_phases comes from the configured phase list, so a 3-phase run needs
        # no code change here.
        _phases = (config.get('target_phases')
                   or [config.get('target_phase', 'venous')])
        # The heteroscedastic head is a different CLASS, not a flag on the
        # generator, so a run without it draws exactly the RNG sequence and holds
        # exactly the parameters the deterministic baseline does.
        self.use_hetero = config.get('use_hetero', False)
        _G = HeteroGenerator if self.use_hetero else UNetGenerator
        self.G = _G(
            dims          = dims,
            base_channels = config['generator_base_channels'],
            dropout       = config.get('generator_dropout', 0.2),
            norm          = config.get('generator_norm', 'instance'),
            in_channels   = config.get('in_channels', 1),
            use_phase_cond= config.get('use_phase_cond', False),
            n_phases      = max(2, len(_phases)),
            cond_dim      = config.get('phase_cond_dim', 64),
            n_levels      = len(config.get('cond_organs') or []),
        ).to(self.device)

        # ── optimisers ──────────────────────────────────────────────────────
        betas = config.get('betas', (0.5, 0.999))
        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr           = config.get('learning_rate', 2e-4),
            betas        = betas,
            weight_decay = config.get('weight_decay', 1e-5),
        )
        # ── scheduler ───────────────────────────────────────────────────────
        if config.get('use_cosine_schedule', True):
            self.sched_G = sched.CosineAnnealingWarmRestarts(
                self.opt_G,
                T_0    = config.get('cosine_t0', 15),
                T_mult = config.get('cosine_tmult', 2),
                eta_min= config.get('cosine_eta_min', 5e-7),
            )
        else:
            self.sched_G = None

        # ── loss ────────────────────────────────────────────────────────────
        self.criterion  = CompositeLoss(config).to(self.device)
        # ── AMP ─────────────────────────────────────────────────────────────
        # use_mixed_precision defaults True independent of device; gate it on
        # actual CUDA availability too, since GradScaler('cuda', enabled=True)
        # / autocast('cuda', enabled=True) assume a CUDA context and error out
        # on a CPU-only machine even though `enabled` looks like a runtime switch.
        self.use_amp  = config.get('use_mixed_precision', True) and self.device == 'cuda'
        self.scaler_G = GradScaler('cuda', enabled=self.use_amp)

        # ── state ───────────────────────────────────────────────────────────
        self.global_step   = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')    # now holds _selection_score, not just MAE
        self.early_stop    = EarlyStopping(config.get('early_stop_patience', 12))
        self._warned_selection = False

        self.history: Dict[str, List] = {k: [] for k in [
            'epoch', 'lr_gen', 'lambda_l1',   # lambda_l1 varies under the decay curriculum
            'train_gen_total', 'train_l1', 'train_nll',
            'train_ssim', 'train_grad', 'train_freq', 'train_organ', 'train_hu_profile',
            'val_loss',
            'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',              # global
            'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc',  # organ-region
        ] + ([
            # Mean |gamma| per FiLM site. This is a RESULT, not diagnostics: if
            # gamma converges to ~0 the decoder learned to ignore the phase
            # input, i.e. conditioning bought nothing.
            f'gamma_{s}' for s in ('bottleneck', 'dec4', 'dec3', 'dec2', 'dec1')
        ] if config.get('use_phase_cond', False) else [])}

        # Per-organ metrics: id→name map (from the CTPhase-XGBoost dump so organ
        # names match that phase model). Missing/unset → per-organ reported by
        # raw label id. Only active when report_per_organ_metrics is on.
        self.per_organ = config.get('report_per_organ_metrics', False)
        self.organ_id_to_name = self._load_organ_id_map(config.get('organ_label_map_json'))
        self.per_organ_history: List[Dict] = []   # [{epoch, organs:{name:{mae,psnr,ssim,ncc,n_items}}}]

        self._log_active_losses()

    # -----------------------------------------------------------------------
    @staticmethod
    def _load_organ_id_map(path) -> Dict[int, str]:
        """Load {label_id: organ_name} from a {name: id} JSON (as dumped by
        retrain_xgb.py). Returns {} on any failure — callers then fall back to
        naming organs by their raw label id."""
        if not path or not Path(path).exists():
            return {}
        try:
            name_to_id = json.loads(Path(path).read_text())
            return {int(v): str(k) for k, v in name_to_id.items()}
        except Exception as e:
            log.warning(f"could not read organ_label_map_json '{path}': {e}")
            return {}

    # -----------------------------------------------------------------------
    def _log_active_losses(self):
        flags = ['ssim', 'gradient', 'frequency', 'organ', 'hu_profile']
        base = ['GaussianNLL'] if self.use_hetero else ['L1']
        active = base + [f for f in flags if self.cfg.get(f'use_{f}')]
        log.info(f"Active losses: {' + '.join(active)}")
        log.info(f"Dims: {self.cfg.get('dims', 2)}-D  |  "
                 f"patch_depth={self.cfg.get('patch_depth', 1)}  |  "
                 f"patch_size={self.cfg.get('patch_size')}")

    # -----------------------------------------------------------------------
    def _level(self, batch: Dict):
        """Level vector for this batch, or None when the generator has none."""
        if not getattr(self.G, 'n_levels', 0):
            return None
        lv = batch.get('level')
        if lv is None:
            raise KeyError("generator has n_levels>0 but the batch has no 'level' "
                           "key — stale patch cache? delete it and re-preload")
        return lv.to(self.device)

    def _phase(self, batch: Dict):
        """Phase ids for this batch, or None when the generator is unconditioned.

        The dataset always emits 'phase', so this is the single place that
        decides whether it is used — passing it to a generator built with
        use_phase_cond=False raises rather than being silently ignored.
        """
        if not getattr(self.G, 'use_phase_cond', False):
            return None
        ph = batch.get('phase')
        if ph is None:
            raise KeyError("generator has use_phase_cond=True but the batch has "
                           "no 'phase' key — stale patch cache? delete it and re-preload")
        return ph.to(self.device)

    def _split_out(self, out):
        """(mu, log_var) under the heteroscedastic head, (pred, None) otherwise.

        Kept in one place because _train_step, _validate and _save_samples all
        need the mean channel and only the first needs the variance one.
        """
        if self.use_hetero:
            return out[:, :1], out[:, 1:2]
        return out, None

    def _train_step(self, batch: Dict) -> Dict:
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        mask   = batch['mask'].to(self.device) if 'mask' in batch else None
        phase  = self._phase(batch)
        level  = self._level(batch)

        self.opt_G.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            fake, log_var = self._split_out(self.G(source, phase, level))
            loss_G, ld = self.criterion(
                pred    = fake,
                target  = target,
                mask    = mask,
                log_var = log_var,
            )

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        self.global_step += 1
        return {
            'gen_total': ld['total'],
            'l1':        ld.get('l1', 0),
            'nll':       ld.get('nll', 0),
            'ssim':      ld.get('ssim', 0),
            'grad':      ld.get('gradient', 0),
            'freq':      ld.get('frequency', 0),
            'organ':     ld.get('organ', 0),
            'hu_profile': ld.get('hu_profile', 0),
        }

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict:
        """Per-item MAE / PSNR / SSIM / NCC, globally and (if a 'mask' is present
        in the batch) restricted to the organ-region voxels. Averaged over all
        validation items."""
        self.G.eval()
        glob = {m: [] for m in _METRICS}     # whole-patch
        org  = {m: [] for m in _METRICS}     # organ-mask region (union) only
        # per-organ: {label_id: {metric: [values]}} — populated only when the
        # multi-label mask is available and report_per_organ_metrics is on.
        per: Dict[int, Dict[str, list]] = {}
        for batch in val_loader:
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            mask = batch.get('mask')
            with autocast('cuda', enabled=self.use_amp):
                fake, _ = self._split_out(
                    self.G(src, self._phase(batch), self._level(batch)))
            for i in range(src.size(0)):
                p = _center_flat(fake[i]).astype(np.float64)
                t = _center_flat(tgt[i]).astype(np.float64)
                for m, v in _metric_set(p, t).items():
                    glob[m].append(v)
                if mask is not None:
                    mvox = _center_flat(mask[i])
                    mk = mvox > 0
                    if mk.sum() >= _MIN_MASK_VOXELS:
                        for m, v in _metric_set(p[mk], t[mk]).items():
                            org[m].append(v)
                    if self.per_organ:
                        # one metric set per distinct organ label present here.
                        # Round once and select on the rounded array — comparing
                        # rounded ids against raw floats (`mvox == lid`) only
                        # worked while the mask held exact integer floats, and
                        # would silently drop voxels under any interpolation.
                        mvox_lbl = np.round(mvox).astype(int)
                        for lid in np.unique(mvox_lbl[mk]):
                            sel = mvox_lbl == lid
                            if sel.sum() >= _MIN_MASK_VOXELS:
                                d = per.setdefault(int(lid), {mm: [] for mm in _METRICS})
                                for m, v in _metric_set(p[sel], t[sel]).items():
                                    d[m].append(v)
        self.G.train()

        def _mean(xs): return float(np.mean(xs)) if xs else 0.0
        out = {
            # val_loss stays = global MAE: it's the early-stopping / best-model
            # metric and existing history/plots key.
            'val_loss': _mean(glob['mae']),
            'val_mae':  _mean(glob['mae']),
            'val_psnr': _mean(glob['psnr']),
            'val_ssim': _mean(glob['ssim']),
            'val_ncc':  _mean(glob['ncc']),
        }
        # Organ-region (union) metrics: 0.0 when no masks were loaded.
        # `n_organ_items` disambiguates "0 because no masks" from a genuine 0.
        for m in _METRICS:
            out[f'val_org_{m}'] = _mean(org[m])
        out['n_organ_items'] = len(org['mae'])

        # Per-organ breakdown, keyed by organ name (id→name if a map was loaded).
        if self.per_organ and per:
            out['per_organ'] = {
                self.organ_id_to_name.get(lid, f'label_{lid}'): {
                    **{m: _mean(d[m]) for m in _METRICS},
                    'n_items': len(d['mae']),
                }
                for lid, d in sorted(per.items())
            }
        return out

    # -----------------------------------------------------------------------
    def _sample_indices(self, ds, n: int, epoch: int) -> List[int]:
        """Pick n validation patch indices to display.

        'fixed'  → the first n patches, every epoch (the legacy behaviour: lets
                   you watch one patch sharpen over epochs, but shows one slice
                   of one case and tells you nothing about the rest).
        'random' → n patches drawn fresh each epoch, spread over as many DISTINCT
                   cases as possible when per-patch provenance is available.
                   Seeded by epoch, so a given epoch's grid is reproducible.
        """
        n = min(n, len(ds))
        if self.cfg.get('sample_mode', 'random') == 'fixed':
            return list(range(n))

        rng = np.random.default_rng(self.cfg.get('seed', 42) * 100003 + epoch)
        case_ids = getattr(ds, 'case_ids', None)
        if not case_ids:                       # old cache: plain random patches
            return [int(i) for i in rng.choice(len(ds), n, replace=False)]

        by_case: Dict[str, List[int]] = {}
        for i, c in enumerate(case_ids):
            by_case.setdefault(str(c), []).append(i)
        cases = sorted(by_case)
        rng.shuffle(cases)
        # One patch per case first, so n distinct patients appear before any
        # patient is shown twice; wrap around only if there are fewer cases than
        # rows (otherwise the grid would silently under-fill).
        picked = []
        while len(picked) < n:
            for c in cases:
                if len(picked) >= n:
                    break
                picked.append(int(rng.choice(by_case[c])))
        return picked

    def _save_samples(self, val_loader: DataLoader, epoch: int):
        """Save a (NCCT | generated | real CECT | abs error) grid.

        Rows are drawn per `sample_mode` — by default random patches from
        distinct validation cases each epoch, rather than the same first batch
        forever. Each row is annotated with its case id, source slice and that
        patch's own PSNR/SSIM, so the figure can be read on its own.
        """
        ds = val_loader.dataset
        idx = self._sample_indices(ds, self.cfg.get('sample_n', 4), epoch)
        if not idx:
            return

        self.G.eval()
        with torch.no_grad():
            batch = torch.utils.data.default_collate([ds[i] for i in idx])
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            fake, _ = self._split_out(
                self.G(src, self._phase(batch), self._level(batch)))

        n = src.size(0)
        case_ids = getattr(ds, 'case_ids', None)
        patch_z  = getattr(ds, 'patch_z', None)

        def _mid(t):
            arr = _np(t)
            return arr[arr.shape[0] // 2] if arr.ndim == 3 else arr

        fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
        if n == 1: axes = axes[None]
        for i in range(n):
            s, f, t = _mid(src[i]), _mid(fake[i]), _mid(tgt[i])
            err = np.abs(f - t)
            for j, (img, kw) in enumerate([
                (s,   dict(cmap='gray', vmin=0, vmax=1)),
                (f,   dict(cmap='gray', vmin=0, vmax=1)),
                (t,   dict(cmap='gray', vmin=0, vmax=1)),
                # Fixed error scale: a per-image autoscale would make every row
                # look equally bad and hide progress across epochs.
                (err, dict(cmap='inferno', vmin=0,
                           vmax=self.cfg.get('sample_err_vmax', 0.15))),
            ]):
                ax = axes[i, j]
                ax.imshow(img, **kw)
                if i == 0:
                    ax.set_title(['NCCT', 'Generated', 'Real CECT', '|error|'][j],
                                 fontsize=9)
                # Ticks/spines off rather than axis('off'), so the row label and
                # the per-row metrics below still render.
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            m = _metric_set(f.ravel(), t.ravel())
            tag = f"{case_ids[idx[i]]}" if case_ids else f"patch {idx[i]}"
            if patch_z:
                tag += f"\nz={patch_z[idx[i]]}"
            axes[i, 0].set_ylabel(tag, fontsize=7)
            # Below the image, not a title — a title on row 0 collides with the
            # '|error|' column header.
            axes[i, 3].set_xlabel(f"PSNR {m['psnr']:.1f}   SSIM {m['ssim']:.3f}",
                                  fontsize=8)
        mode = self.cfg.get('sample_mode', 'random')
        plt.suptitle(f'Epoch {epoch}  ({mode} val samples)', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.samples / f'ep{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

        # Sort by mtime, not filename: a fresh (non-resumed) run restarts its
        # epoch counter at 1, and a filename/epoch-number sort would keep
        # stale high-numbered images left over from an earlier, longer run of
        # this same scenario instead of the current run's own latest samples.
        keep = self.cfg.get('keep_last_n_sample_epochs', 5)
        imgs = sorted(self.samples.glob('ep*.png'), key=lambda p: p.stat().st_mtime)
        for old in imgs[:-keep]:
            old.unlink(missing_ok=True)

        self.G.train()

    # -----------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool):
        # Everything needed to resume *exactly* where we left off: not just the
        # weights, but the optimiser, LR scheduler, AMP scalers, step counter,
        # metric history and early-stopping state. Without the scheduler/history
        # a resume would restart the cosine LR cycle and drop the loss curves.
        state = {
            'epoch':       epoch,
            'global_step': self.global_step,
            'G_state':     self.G.state_dict(),
            'opt_G':       self.opt_G.state_dict(),
            'best_val':    self.best_val_loss,
            'history':     self.history,
            # Without this, organ_metrics.json restarts from an empty list on
            # resume and every pre-resume epoch's per-organ breakdown is lost.
            'per_organ_history': self.per_organ_history,
            'early_stop':  {'best': self.early_stop.best,
                            'counter': self.early_stop.counter},
        }
        if self.sched_G is not None:
            state['sched_G'] = self.sched_G.state_dict()
        if self.use_amp:
            state['scaler_G'] = self.scaler_G.state_dict()
        # Write to a temp file and rename into place. torch.save straight to the
        # final path leaves a truncated, unloadable .pth if the process dies
        # mid-write (second Ctrl-C during the emergency save, OOM-kill, full
        # disk) — rename is atomic on the same filesystem, so a checkpoint that
        # exists is always complete.
        path = self.out / f'ckpt_ep{epoch:03d}.pth'
        self._atomic_save(state, path)
        if is_best:
            self._atomic_save(state, self.out / 'best_model.pth')
            log.info(f"  ★ best model saved (epoch {epoch})")
        # Same mtime-not-filename reasoning as _save_samples above.
        keep = self.cfg.get('keep_last_n_checkpoints', 3)
        ckpts = sorted(self.out.glob('ckpt_ep*.pth'), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-keep]:
            old.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save(state: Dict, path: Path):
        """torch.save via a temp file + rename, so `path` is never a partial write."""
        tmp = path.with_suffix(path.suffix + '.tmp')
        try:
            with open(tmp, 'wb') as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())      # rename is only safe once bytes are on disk
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        self.G.load_state_dict(state['G_state'])
        self.opt_G.load_state_dict(state['opt_G'])
        if 'sched_G' in state and self.sched_G is not None:
            self.sched_G.load_state_dict(state['sched_G'])
        if self.use_amp:
            if 'scaler_G' in state:
                self.scaler_G.load_state_dict(state['scaler_G'])
        self.best_val_loss = state.get('best_val', float('inf'))
        self.global_step   = state.get('global_step', 0)
        if state.get('history'):
            # Keep only the epochs at/below the resume point so re-training an
            # already-recorded epoch overwrites rather than duplicates it.
            ep = state.get('epoch', 0)
            hist = state['history']
            if hist.get('epoch'):
                keep = sum(1 for e in hist['epoch'] if e <= ep)
                self.history = {k: list(v[:keep]) for k, v in hist.items()}
            else:
                self.history = hist
        if state.get('per_organ_history'):
            ep = state.get('epoch', 0)
            self.per_organ_history = [r for r in state['per_organ_history']
                                      if r.get('epoch', 0) <= ep]
        es = state.get('early_stop')
        if es:
            self.early_stop.best    = es.get('best', float('inf'))
            self.early_stop.counter = es.get('counter', 0)
        ep = state.get('epoch', 0)
        log.info(f"Resumed from checkpoint epoch {ep} "
                 f"(global_step={self.global_step}, best_val={self.best_val_loss:.6f})")
        return ep

    # -----------------------------------------------------------------------
    def _selection_score(self, val: Dict) -> float:
        """Value to MINIMISE for best-checkpoint selection and early stopping.

        Default is `val_org_ssim` (negated, since higher is better). The previous
        default — global `val_loss`, i.e. whole-patch MAE — structurally favours
        the blurriest epoch: L1-optimal predictions are the conditional mean, and
        MAE rewards exactly that. It also scores mostly background (global PSNR
        runs ~3.6 dB above organ-region PSNR here), so it barely discriminates on
        the anatomy that matters. Organ-region SSIM is texture-sensitive and
        restricted to the mask.

        Falls back to `val_loss` if the organ metric is unavailable (no masks),
        so runs without segmentation still train.
        """
        metric = self.cfg.get('selection_metric', 'val_org_ssim')
        if metric == 'val_loss':
            return float(val['val_loss'])
        v = val.get(metric)
        if not v:                       # missing or 0.0 → no organ voxels scored
            if not self._warned_selection:
                log.warning(f"selection_metric '{metric}' unavailable "
                            f"(no organ mask?) — falling back to val_loss")
                self._warned_selection = True
            return float(val['val_loss'])
        return -float(v)                # higher SSIM is better

    # -----------------------------------------------------------------------
    def _update_history(self, epoch, avgs, val):
        h = self.history
        h['epoch'].append(epoch)
        h['lr_gen'].append(self.opt_G.param_groups[0]['lr'])
        h['lambda_l1'].append(self.criterion._l1_w())
        for k in ['gen_total', 'l1', 'nll',
                  'ssim', 'grad', 'freq', 'organ', 'hu_profile']:
            h[f'train_{k}'].append(avgs.get(k, 0.0))
        for k in ['val_loss',
                  'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',
                  'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc']:
            h[k].append(val.get(k, 0.0))
        for k, v in self.G.film_stats().items():
            if k in h:
                h[k].append(v)

    def _save_history(self):
        hist = {k: [float(v) for v in vl] for k, vl in self.history.items()}
        with open(self.out / 'history.json', 'w') as f:
            json.dump(hist, f, indent=2)

    def _record_organ_metrics(self, epoch: int, per_organ: Dict):
        """Append this epoch's per-organ metrics and dump organ_metrics.json.
        Logs a compact per-organ MAE line (which organs the generator does
        best/worst on — organs missing from the mask simply don't appear)."""
        self.per_organ_history.append({'epoch': epoch, 'organs': per_organ})
        with open(self.out / 'organ_metrics.json', 'w') as f:
            json.dump(self.per_organ_history, f, indent=2)
        line = "  ".join(f"{name}: MAE={d['mae']:.3f} SSIM={d['ssim']:.3f} "
                         f"NCC={d['ncc']:.3f}(n={d['n_items']})"
                         for name, d in per_organ.items())
        log.info(f"  per-organ | {line}")

    def _plot_history(self):
        if not self.history['epoch']:
            return
        ep = self.history['epoch']
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        ax = axes[0, 0]
        ax.plot(ep, self.history['train_gen_total'], label='Gen total')
        ax.set_title('Total loss'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for k, lbl in [('train_l1', 'L1'), ('train_nll', 'GaussNLL')]:
            if any(v > 0 for v in self.history[k]):
                ax.plot(ep, self.history[k], label=lbl)
        ax.set_title('Fidelity term'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        for k, lbl in [('train_ssim','SSIM'), ('train_grad','Grad'),
                        ('train_freq','Freq'), ('train_organ','Organ'),
                        ('train_hu_profile','HUprof')]:
            if any(v > 0 for v in self.history[k]):
                ax.plot(ep, self.history[k], label=lbl)
        ax.set_title('Extra losses'); ax.legend(); ax.grid(alpha=0.3)

        axes[0, 3].plot(ep, self.history['lr_gen']); axes[0, 3].set_title('LR (gen)')
        axes[0, 3].grid(alpha=0.3)

        # Bottom row: each of the 4 val metrics, global vs organ-region overlaid.
        # Only draw the organ line if organ metrics were actually recorded.
        has_organ = any(v > 0 for v in self.history['val_org_mae'])
        for col, (m, title) in enumerate([('mae', 'Val MAE'), ('psnr', 'Val PSNR'),
                                          ('ssim', 'Val SSIM'), ('ncc', 'Val NCC')]):
            ax = axes[1, col]
            ax.plot(ep, self.history[f'val_{m}'], label='global')
            if has_organ:
                ax.plot(ep, self.history[f'val_org_{m}'], label='organ')
            ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.out / 'curves.png', dpi=120, bbox_inches='tight')
        plt.close()

    # -----------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        start_epoch:  int = 0,
    ):
        log.info(f"Training {start_epoch + 1} → {epochs}  |  device={self.device}")
        save_every = self.cfg.get('save_samples_interval', 1)

        for epoch in range(start_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            self.criterion.set_epoch(epoch)
            self.G.train()

            accum: Dict[str, List] = {k: [] for k in [
                'gen_total', 'l1', 'nll',
                'ssim', 'grad', 'freq', 'organ', 'hu_profile'
            ]}

            pbar = tqdm(train_loader, desc=f'Ep {epoch}/{epochs}', leave=False)
            for batch in pbar:
                step = self._train_step(batch)
                for k in accum:
                    accum[k].append(step.get(k, 0.0))
                pbar.set_postfix(
                    total=f"{step['gen_total']:.4f}",
                    l1=f"{step['l1']:.4f}",
                )

            avgs = {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}

            if self.sched_G:
                self.sched_G.step()

            val = self._validate(val_loader)

            msg = (
                f"Ep {epoch:3d}/{epochs} | "
                f"total={avgs['gen_total']:.4f}  l1={avgs['l1']:.4f}  "
                f"nll={avgs['nll']:.4f}  organ={avgs['organ']:.4f} | "
                f"MAE={val['val_mae']:.4f}  PSNR={val['val_psnr']:.2f}  "
                f"SSIM={val['val_ssim']:.4f}  NCC={val['val_ncc']:.4f}"
            )
            if val.get('n_organ_items', 0) > 0:
                msg += (f" | organ: MAE={val['val_org_mae']:.4f}  "
                        f"PSNR={val['val_org_psnr']:.2f}  SSIM={val['val_org_ssim']:.4f}  "
                        f"NCC={val['val_org_ncc']:.4f}")
            msg += f" | lr={self.opt_G.param_groups[0]['lr']:.2e}"
            log.info(msg)

            self._update_history(epoch, avgs, val)
            self._save_history()
            if val.get('per_organ'):
                self._record_organ_metrics(epoch, val['per_organ'])
            if epoch % 5 == 0 or epoch == epochs:
                self._plot_history()

            # Model selection score, as a value to MINIMISE (see _selection_score).
            score = self._selection_score(val)
            is_best = score < self.best_val_loss
            if is_best:
                self.best_val_loss = score
            self._save_checkpoint(epoch, is_best)

            if epoch % save_every == 0:
                self._save_samples(val_loader, epoch)

            if self.early_stop.step(score):
                log.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        log.info(f"Done. Best val loss: {self.best_val_loss:.6f}")
