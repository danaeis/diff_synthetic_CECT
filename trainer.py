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

ADVERSARIAL BRANCH: the discriminator half of the step lives in
`trainer_adv.AdversarialMixin`, which this class mixes in and every subclass
therefore inherits — including `DiffusionTrainer`, which is the reason it is a
mixin rather than inline code here. Read that module's docstring for what it
changes relative to the `../synthetic_CECT/trainer.py` original.
"""

import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

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
from metrics import grad_hist_distance, raps_hf_ratio
from models import UNetGenerator
from models_hetero import HeteroGenerator
from trainer_adv import AdversarialMixin

log = logging.getLogger(__name__)


# Generator-side loss terms that get a `train_<k>_contrib` history channel.
#
# WHY THIS LIST EXISTS. The legacy `train_<k>` channels are NOT in consistent
# units and cannot be: `train_l1` and `train_adv` hold RAW values while
# `train_organ`, `train_ssim`, `train_grad`, `train_freq`, `train_hu_profile`,
# `train_nll` and `train_fm` hold LAMBDA-SCALED ones (see CompositeLoss.forward,
# which documents why each is the way it is). Those units are frozen — changing
# them would silently redefine every curve in every history.json already on
# disk — so the fix is a parallel set of channels that are all scaled, all
# comparable, and all summing to `train_gen_total`.
#
# That makes the question a lambda is actually chosen by — "what fraction of the
# gradient is this term carrying?" — a division rather than a guess:
#
#     share_k = train_<k>_contrib / train_gen_total
#
# `disc` is deliberately absent: it is the discriminator's own objective under a
# different optimiser, not a term in the generator total, so it has no share.
CONTRIB_TERMS = ('l1', 'nll', 'ssim', 'grad', 'freq',
                 'organ', 'hu_profile', 'adv', 'fm')


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


def _center_2d(t: torch.Tensor) -> np.ndarray:
    """A single tensor item (1,H,W) or (1,D,H,W) → the 2-D centre slice.

    `_center_flat` is the same selection followed by `.ravel()`. The detail
    metrics need the 2-D structure kept: a spectrum and a spatial gradient are
    both undefined on a flattened voxel list."""
    a = _np(t)
    if a.ndim == 3:                         # (D, H, W) — centre slice
        a = a[a.shape[0] // 2]
    return a


# Detail metrics are O(N) FFTs; 64 slices is `metrics.raps`'s own default and is
# plenty for a training curve. Beyond it the estimate stops moving and the val
# pass starts costing more than the epoch.
_DETAIL_MAX_SLICES = 64


def _detail_metric_set(preds: List[np.ndarray], tgts: List[np.ndarray],
                       masks: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    """raps_hf / grad_w1 over a stack of paired 2-D slices.

    This is the axis `_metric_set` cannot see. MAE, PSNR, SSIM and NCC are all
    minimised by the conditional mean, so they rank a blurry copy of the input
    ABOVE a correctly-textured sample — which is exactly the failure these runs
    exhibit. `raps_hf` reads the high-frequency amplitude ratio (1.0 = correct,
    <1 = blur, >1 = noise/hallucination) and `grad_w1` the W1 distance between
    edge-strength distributions (0 = correct). Neither is capped by residual
    registration error, because neither depends on where the edges are.

    The stack is passed as a pseudo-volume with `slice_axis=0`: `metrics.raps`
    and `metrics.grad_hist_distance` both average over 2-D slices, so a batch of
    unrelated patches is a valid input as long as pred and target are paired
    slice-for-slice.
    """
    if not preds:
        return {'val_raps_hf': 0.0, 'val_grad_w1': 0.0, 'val_org_grad_w1': 0.0}
    g = np.stack(preds[:_DETAIL_MAX_SLICES]).astype(np.float64)
    r = np.stack(tgts[:_DETAIL_MAX_SLICES]).astype(np.float64)

    def _f(x):                        # NaN (degenerate stack) → 0.0, the "unset"
        return float(x) if np.isfinite(x) else 0.0   # sentinel the plots already use

    out = {
        'val_raps_hf':  _f(raps_hf_ratio(g, r, slice_axis=0)),
        'val_grad_w1':  _f(grad_hist_distance(g, r, slice_axis=0)),
        'val_org_grad_w1': 0.0,
    }
    if masks:
        m = np.stack(masks[:_DETAIL_MAX_SLICES])
        out['val_org_grad_w1'] = _f(
            grad_hist_distance(g, r, mask=(m > 0), slice_axis=0))
    return out


class WeightEMA:
    """Exponential moving average of the model weights.

    Absent from this repo entirely before now, which is unusual for a diffusion
    model — EMA is near-universal in the DDPM lineage and is normally worth a
    large margin on exactly the axis this project is failing at, sample quality.
    A single-step SGD iterate is a noisy point in weight space; the average over
    the last ~1/(1-decay) steps is not, and the difference shows up as texture
    rather than as loss.

    Warmup: at step 0 an EMA initialised from the weights and decayed at 0.999
    lags ~1000 steps behind, so early samples would be of a near-random model.
    `min(decay, (1 + step) / (10 + step))` ramps the effective decay from 0.1 up
    to `decay`, which is the standard fix.

    Buffers are copied, not averaged: they are counters and running statistics
    (and, here, the schedule's alpha tables), where an average is meaningless.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> Dict:
        return {'decay': self.decay, 'step': self.step, 'shadow': self.shadow}

    def load_state_dict(self, state: Dict):
        self.decay = state.get('decay', self.decay)
        self.step = state.get('step', 0)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in state['shadow'].items()}

    def averaged_state(self, model: nn.Module) -> Dict:
        """A full state_dict with the averaged weights in place, dtypes preserved.

        This is what gets written to the checkpoint's `G_state`, so
        `infer_volume.load_model` uses the EMA weights without knowing EMA
        exists. The live weights go to `G_raw_state` for exact resume.
        """
        out = {}
        for k, v in model.state_dict().items():
            out[k] = (self.shadow[k].to(v.dtype) if k in self.shadow
                      else v.detach().clone())
        return out

    class _Swap:
        def __init__(self, ema, model):
            self.ema, self.model = ema, model
            self.backup = None

        def __enter__(self):
            self.backup = {k: v.detach().clone()
                           for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.averaged_state(self.model))
            return self.model

        def __exit__(self, *exc):
            self.model.load_state_dict(self.backup)
            self.backup = None
            return False

    def swapped(self, model: nn.Module) -> '_Swap':
        """Context manager: EMA weights inside, live weights restored on exit.

        Used for every validation and sampling pass, so the curves and the sample
        grids describe the same weights that inference will load.
        """
        return WeightEMA._Swap(self, model)


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

class Trainer(AdversarialMixin):
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

        # ── discriminator (optional) ────────────────────────────────────────
        # Built last, and strictly inside its own flag, so a non-adversarial run
        # draws the RNG sequence it drew before this branch existed. No timestep
        # conditioning here: this generator has no timestep. Its phase/level
        # vector is `phase_cond_dim` wide, and is only fed to D when the D is
        # conditional at all — an unconditional texture critic that also got the
        # requested phase would be judging a pairing it cannot see.
        self._init_adversarial(
            config, dims=dims,
            source_channels=config.get('in_channels', 1),
            cond_dim=(config.get('phase_cond_dim', 64)
                      if (config.get('use_cond_disc', False)
                          and getattr(self.G, 'use_cond', False)) else 0),
            use_t_cond=False,
        )

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
            # Adversarial branch. train_adv is the RAW generator-side GAN loss
            # (not lambda-scaled — see CompositeLoss) and train_disc is D's own
            # loss. Read them TOGETHER: train_adv falling while train_disc also
            # falls means D is winning and G is not, which is the collapse case;
            # both hovering is the healthy one. Constant 0 when the flag is off.
            # train_d_real / train_d_fake are D's raw logit means (see
            # _disc_step) and exist to resolve the one thing train_disc cannot:
            # under hinge loss, "D is confused/inverted" (real << fake) and
            # "D has collapsed to a constant" (real ~= fake ~= 0) both write
            # train_disc == 1.0 (its value at the origin) — a healthy but not
            # yet confident D lands in that same neighbourhood too. The raw
            # logits tell those apart; the scalar loss does not.
            'train_adv', 'train_fm', 'train_disc', 'train_d_real', 'train_d_fake',
            'lambda_adv',
            'val_loss',
        ] + [
            # See CONTRIB_TERMS. One channel per generator-side term holding the
            # LAMBDA-SCALED value it actually contributed, because the legacy
            # channels above are a mix of raw and scaled and cannot be compared
            # with each other. Divide by train_gen_total for the gradient share.
            f'train_{k}_contrib' for k in CONTRIB_TERMS
        ] + [
            'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',              # global
            'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc',  # organ-region
            # Detail/texture, the axis the per-voxel metrics above cannot see. A
            # model that synthesises correct texture LOSES on val_mae relative to
            # a blurry copy, so these are the ones to read for sample quality.
            # raps_hf is a ratio scored as |x-1|; grad_w1 is a distance, lower is
            # better. See metrics.raps_hf_ratio / metrics.grad_hist_distance.
            'val_raps_hf', 'val_grad_w1', 'val_org_grad_w1',
            # 1.0 when the val metrics on this row were freshly computed, 0.0 when
            # they were carried forward from an earlier epoch (the diffusion path
            # samples only every `diffusion_sample_every` epochs). Without this the
            # curves show flat 10-epoch steps that read as plateaus but are not.
            'val_sampled',
        ] + ([
            # Mean |gamma| per FiLM site. This is a RESULT, not diagnostics: if
            # gamma converges to ~0 the decoder learned to ignore the phase
            # input, i.e. conditioning bought nothing.
            f'gamma_{s}' for s in ('bottleneck', 'dec4', 'dec3', 'dec2', 'dec1')
        ] if config.get('use_phase_cond', False) else [])}
        self._hist_keys = list(self.history)

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
        flags = ['ssim', 'gradient', 'frequency', 'organ', 'hu_profile',
                 'adversarial', 'feature_matching']
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

    def _cond_for_d(self, phase, level):
        """Conditioning vector for the DISCRIMINATOR — always DETACHED.

        It is produced by the generator's embedding, so it must be cut from that
        graph: D is a critic, and letting its gradients reach G's conditioning
        embedding would let G lower the adversarial loss by reshaping what it
        *claims* was requested instead of by improving the image.
        """
        if not getattr(self.D, 'cond_dim', 0):
            return None
        return self.G.cond_vec(phase, level).detach()

    def _train_step(self, batch: Dict) -> Dict:
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        mask   = batch['mask'].to(self.device) if 'mask' in batch else None
        phase  = self._phase(batch)
        level  = self._level(batch)
        dcond  = self._cond_for_d(phase, level) if self.D is not None else None

        # ONE generator forward. `fake.detach()` feeds D; the same graph is then
        # reused for the G step. See trainer_adv's docstring, point 1, for why
        # the reference implementation's second forward is not needed.
        self.opt_G.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            fake, log_var = self._split_out(self.G(source, phase, level))

        # NaN = "not measured this step", so the epoch mean is over the steps D
        # actually ran on rather than being divided by disc_update_freq. See the
        # nanmean in `train`.
        loss_D_val = d_real_val = d_fake_val = 0.0 if self.D is None else float('nan')
        if self.D is not None and (self.global_step % self.disc_freq == 0):
            loss_D_val, d_real_val, d_fake_val = self._disc_step(
                target, fake.detach(), source=source, cond=dcond)

        with autocast('cuda', enabled=self.use_amp):
            adv_logits = real_feats = fake_feats = None
            if self.D is not None:
                with self._frozen_d():
                    adv_logits, real_feats, fake_feats = self._disc_verdict(
                        fake, real=target, source=source, cond=dcond)

            loss_G, ld = self.criterion(
                pred    = fake,
                target  = target,
                mask    = mask,
                log_var = log_var,
                adv_fake_logits = adv_logits if self.use_adv else None,
                adv_weight      = self._adv_w() if self.use_adv else None,
                real_features   = real_feats,
                fake_features   = fake_feats,
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
            'adv':       ld.get('adversarial', 0),
            'fm':        ld.get('feature_matching', 0),
            'disc':      loss_D_val,
            'd_real':    d_real_val,
            'd_fake':    d_fake_val,
            # Lambda-scaled twins. The short names here differ from
            # CompositeLoss's long ones (grad/gradient, freq/frequency,
            # adv/adversarial, fm/feature_matching), so the mapping is spelled
            # out rather than derived — a silent rename would zero a curve.
            'l1_contrib':         ld.get('l1_contrib', 0),
            'nll_contrib':        ld.get('nll_contrib', 0),
            'ssim_contrib':       ld.get('ssim_contrib', 0),
            'grad_contrib':       ld.get('gradient_contrib', 0),
            'freq_contrib':       ld.get('frequency_contrib', 0),
            'organ_contrib':      ld.get('organ_contrib', 0),
            'hu_profile_contrib': ld.get('hu_profile_contrib', 0),
            'adv_contrib':        ld.get('adversarial_contrib', 0),
            'fm_contrib':         ld.get('feature_matching_contrib', 0),
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
        # 2-D centre slices for the detail metrics, capped (see _DETAIL_MAX_SLICES).
        d_pred: List[np.ndarray] = []
        d_tgt:  List[np.ndarray] = []
        d_mask: List[np.ndarray] = []
        for batch in val_loader:
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            mask = batch.get('mask')
            with autocast('cuda', enabled=self.use_amp):
                fake, _ = self._split_out(
                    self.G(src, self._phase(batch), self._level(batch)))
            for i in range(src.size(0)):
                if len(d_pred) < _DETAIL_MAX_SLICES:
                    d_pred.append(_center_2d(fake[i]))
                    d_tgt.append(_center_2d(tgt[i]))
                    if mask is not None:
                        d_mask.append(_center_2d(mask[i]))
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
        out.update(_detail_metric_set(d_pred, d_tgt, d_mask or None))

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
        # With EMA on, `G_state` holds the AVERAGED weights and `G_raw_state` the
        # live ones. That way `infer_volume.load_model`, which reads `G_state`,
        # gets the EMA weights without knowing EMA exists, while `load_checkpoint`
        # prefers `G_raw_state` so a resume continues from the real iterate rather
        # than from its own average. Checkpoints written before EMA existed have
        # only `G_state` and load unchanged.
        ema = getattr(self, 'ema', None)
        state = {
            'epoch':       epoch,
            'global_step': self.global_step,
            'G_state':     (ema.averaged_state(self.G) if ema is not None
                            else self.G.state_dict()),
            'opt_G':       self.opt_G.state_dict(),
            'best_val':    self.best_val_loss,
            'history':     self.history,
            # Without this, organ_metrics.json restarts from an empty list on
            # resume and every pre-resume epoch's per-organ breakdown is lost.
            'per_organ_history': self.per_organ_history,
            'early_stop':  {'best': self.early_stop.best,
                            'counter': self.early_stop.counter},
        }
        if ema is not None:
            state['G_raw_state'] = self.G.state_dict()
            state['ema'] = ema.state_dict()
        if self.sched_G is not None:
            state['sched_G'] = self.sched_G.state_dict()
        if self.use_amp:
            state['scaler_G'] = self.scaler_G.state_dict()
        # D, its optimiser and its scaler, when there is one. Resuming an
        # adversarial run without them would restart the critic from scratch
        # against a fully-trained generator — a step change in the objective
        # that is indistinguishable, in the curves, from a training collapse.
        self._adv_checkpoint_state(state)
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
        # Prefer the live weights over the averaged ones: resuming from the EMA
        # would restart training from a point the optimiser state does not match.
        self.G.load_state_dict(state.get('G_raw_state') or state['G_state'])
        ema = getattr(self, 'ema', None)
        if ema is not None:
            if state.get('ema'):
                ema.load_state_dict(state['ema'])
            else:
                # Resuming a pre-EMA run, or one trained with ema_decay=0. Seed
                # the average from the loaded weights instead of leaving it at a
                # random init that would take ~1/(1-decay) steps to wash out.
                log.info("  no EMA state in checkpoint — seeding EMA from the "
                         "loaded weights")
                ema.__init__(self.G, ema.decay)
        self.opt_G.load_state_dict(state['opt_G'])
        self._adv_load_state(state)
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
            # Backfill channels this build records but the checkpoint predates
            # (the adversarial ones, most recently). Without it `_update_history`
            # raises KeyError on the first post-resume epoch, i.e. adding ANY new
            # history channel silently breaks every existing checkpoint.
            n = len(self.history.get('epoch', []))
            for k in self._history_keys():
                if k not in self.history:
                    self.history[k] = [0.0] * n
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
    def _selection_label(self) -> str:
        """Human-readable name of what `_selection_score` minimises, for the plot
        title. Subclasses that override `_selection_score` should override this
        too, so the figure never claims a selection rule the code does not use."""
        return f"selection: {self.cfg.get('selection_metric', 'val_org_ssim')}"

    def _early_stop_score(self, val: Dict, selection_score: float) -> float:
        """Value to MINIMISE for early-stopping patience. Same as the selection
        score unless a subclass needs a signal available on every epoch."""
        return selection_score

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
        h['lambda_adv'].append(self._adv_w())
        for k in ['gen_total', 'l1', 'nll',
                  'ssim', 'grad', 'freq', 'organ', 'hu_profile',
                  'adv', 'fm', 'disc', 'd_real', 'd_fake']:
            h[f'train_{k}'].append(avgs.get(k, 0.0))
        # Lambda-scaled contributions, recorded next to the legacy channels
        # rather than replacing them (those units are frozen — see
        # CompositeLoss.forward). `train_<k>_contrib / train_gen_total` is the
        # gradient share of term k, which is the number to size a lambda by.
        # `disc` has none: it is D's own objective, optimised by a different
        # optimiser, and is not a term in the generator total.
        for k in CONTRIB_TERMS:
            key = f'train_{k}_contrib'
            if key in h:
                h[key].append(avgs.get(f'{k}_contrib', 0.0))
        for k in ['val_loss',
                  'val_mae', 'val_psnr', 'val_ssim', 'val_ncc',
                  'val_org_mae', 'val_org_psnr', 'val_org_ssim', 'val_org_ncc',
                  'val_raps_hf', 'val_grad_w1', 'val_org_grad_w1']:
            h[k].append(val.get(k, 0.0))
        # Deterministic trainers validate every epoch, so a missing flag means
        # "fresh". Only the diffusion path ever carries values forward.
        h['val_sampled'].append(float(val.get('val_sampled', 1.0)))
        for k, v in self.G.film_stats().items():
            if k in h:
                h[k].append(v)

    def _history_keys(self) -> List[str]:
        """Channel names this build records, captured at construction.

        Needed because `load_checkpoint` replaces `self.history` wholesale with
        whatever dict the checkpoint holds, which may predate a channel.
        """
        return list(getattr(self, '_hist_keys', ()) or self.history)

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
        """Three rows: train losses, per-voxel val metrics, detail/texture metrics.

        Two things this deliberately does NOT do, both of which made the earlier
        two-row version misleading:

        1. It never draws a carried-forward val value as if it were measured. The
           diffusion path samples only every `diffusion_sample_every` epochs and
           repeats the last value in between; plotted as a line that reads as a
           10-epoch plateau. Rows 2 and 3 plot markers at the epochs actually
           measured (`val_sampled == 1`) with a faint connecting line.
        2. It no longer hides `val_loss`, which is what drives checkpoint
           selection and early stopping on the diffusion path and was previously
           absent from the figure entirely.

        Row 3 is the axis that matters for sample quality: `val_mae` and friends
        cannot see blur-vs-texture, and rank a blurry copy above a correctly
        textured sample. Read `raps_hf` as |x - 1| (<1 blur, >1 noise) and
        `grad_w1` as a distance.
        """
        if not self.history['epoch']:
            return
        ep = self.history['epoch']
        # Epochs whose val row was freshly measured. Older history.json files
        # predate the flag; treat every row as fresh so they still plot.
        fresh = self.history.get('val_sampled') or [1.0] * len(ep)
        fresh = [bool(v) for v in fresh[:len(ep)]]
        fresh += [True] * (len(ep) - len(fresh))

        def _legend(ax):
            """Only when something was actually drawn — a bare ax.legend() on an
            empty axis warns once per panel per plot, and these panels are empty
            for any run predating the detail metrics."""
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

        def _val_plot(ax, key, label, color=None):
            """Markers at measured epochs, faint line for the carried-forward shape.

            Exact 0.0 is this codebase's "not measured" sentinel for every val
            metric (`_validate` seeds them at 0.0, `_detail_metric_set` maps NaN
            to 0.0, `n_organ_items` exists to disambiguate a real 0). Drawing it
            as a value rescales the axis to include a number no model produced —
            on a diffusion run it drags the PSNR panel from a 25-28 dB range down
            to 0-28 and flattens the entire curve. Masked to NaN so matplotlib
            leaves a gap.
            """
            ys = self.history.get(key)
            if not ys:
                return
            n = min(len(ys), len(ep))
            ys = [float('nan') if y == 0.0 else y for y in ys[:n]]
            if not any(y == y for y in ys):        # all-NaN → nothing measured
                return
            xs_m = [e for e, f, y in zip(ep, fresh, ys) if f and y == y]
            ys_m = [y for f, y in zip(fresh, ys) if f and y == y]
            ln, = ax.plot(ep[:n], ys, alpha=0.25, lw=1, color=color)
            ax.plot(xs_m, ys_m, 'o', ms=3, label=label, color=ln.get_color())

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        ax = axes[0, 0]
        ax.plot(ep, self.history['train_gen_total'], label='Gen total')
        ax.set_title('Total loss'); _legend(ax); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for k, lbl in [('train_l1', 'L1'), ('train_nll', 'GaussNLL')]:
            if any(v > 0 for v in self.history[k]):
                ax.plot(ep, self.history[k], label=lbl)
        # Both are RAW, un-lambda-scaled, so the panel means the same thing in
        # every mode. Multiply L1 by history['lambda_l1'] for its contribution.
        ax.set_title('Fidelity term (raw, unweighted)'); _legend(ax); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        # Adv and Disc are drawn on this panel deliberately, and dashed: they are
        # the only pair here that has to be read TOGETHER. G's adversarial loss
        # falling on its own means nothing; falling while D's loss also falls
        # means D is winning and the term is injecting noise, which is the
        # failure this panel exists to make visible within one epoch of it
        # starting. Multiply Adv by history['lambda_adv'] for its contribution.
        for k, lbl in [('train_ssim','SSIM'), ('train_grad','Grad'),
                        ('train_freq','Freq'), ('train_organ','Organ'),
                        ('train_hu_profile','HUprof'), ('train_fm','FM')]:
            if any(v > 0 for v in (self.history.get(k) or [])):
                ax.plot(ep, self.history[k], label=lbl)
        for k, lbl in [('train_adv','Adv (G)'), ('train_disc','Disc (D)')]:
            if any(v > 0 for v in (self.history.get(k) or [])):
                ax.plot(ep, self.history[k], ls='--', label=lbl)
        # D's raw logit means, not just its loss — see _disc_step. Real well
        # above fake (both past the ±1 hinge margin) is a discriminating D;
        # both flat near 0 is a dead one; train_disc alone reads the same in
        # both cases.
        for k, lbl in [('train_d_real','D(real)'), ('train_d_fake','D(fake)')]:
            vs = self.history.get(k) or []
            if any(v != 0 for v in vs):
                ax.plot(ep, vs, ls=':', label=lbl)
        ax.set_title('Extra losses'); _legend(ax); ax.grid(alpha=0.3)

        axes[0, 3].plot(ep, self.history['lr_gen']); axes[0, 3].set_title('LR (gen)')
        axes[0, 3].grid(alpha=0.3)

        # Row 2: the 4 per-voxel val metrics, global vs organ-region overlaid.
        # Only draw the organ line if organ metrics were actually recorded.
        has_organ = any(v > 0 for v in self.history['val_org_mae'])
        for col, (m, title) in enumerate([('mae', 'Val MAE'), ('psnr', 'Val PSNR'),
                                          ('ssim', 'Val SSIM'), ('ncc', 'Val NCC')]):
            ax = axes[1, col]
            _val_plot(ax, f'val_{m}', 'global')
            if has_organ:
                _val_plot(ax, f'val_org_{m}', 'organ')
            ax.set_title(f'{title}  (per-voxel)'); _legend(ax); ax.grid(alpha=0.3)

        # Row 3: the detail axis, plus the quantity selection actually reads.
        ax = axes[2, 0]
        _val_plot(ax, 'val_raps_hf', 'raps_hf')
        if any(v > 0 for v in self.history.get('val_raps_hf', [])):
            ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'not recorded\n(run predates detail metrics)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, alpha=0.5)
        ax.set_title('Val RAPS-hf  (1.0 = correct detail)')
        _legend(ax); ax.grid(alpha=0.3)

        ax = axes[2, 1]
        _val_plot(ax, 'val_grad_w1', 'global')
        _val_plot(ax, 'val_org_grad_w1', 'organ')
        ax.set_title('Val gradW1  (0 = correct edge stats)')
        _legend(ax); ax.grid(alpha=0.3)

        ax = axes[2, 2]
        ax.plot(ep, self.history['val_loss'], label='val_loss')
        ax.set_title(f'Val loss  ({self._selection_label()})')
        _legend(ax); ax.grid(alpha=0.3)

        # The loss-balance panel. Every OTHER loss panel plots terms in units
        # that are not comparable to each other (see CONTRIB_TERMS), so none of
        # them can answer the question a lambda is actually chosen by: what
        # fraction of the gradient is this term carrying? This one does.
        #
        # Read it for two failures that are invisible elsewhere:
        #   * a band that fills the plot — one term owns the objective and the
        #     others are decoration. lambda_adv=0.5 against a diffusion MSE of
        #     order 1e-2 looks exactly like this.
        #   * a band pinned near zero — that term's lambda is too small to do
        #     anything, which is what config.py records happening to lambda_organ
        #     at 5.0 ("reached only 5% of the L1 term early").
        ax = axes[2, 3]
        shares, labels = [], []
        for k in CONTRIB_TERMS:
            ys = self.history.get(f'train_{k}_contrib')
            if ys and any(abs(v) > 0 for v in ys):
                shares.append([abs(float(v)) for v in ys[:len(ep)]])
                labels.append(k)
        if shares:
            # Normalise to the sum of the drawn bands rather than to
            # train_gen_total: the two agree when every active term is recorded,
            # and when they disagree the difference is a term this build does not
            # log, which would otherwise show up as an unexplained white gap.
            tot = [sum(col) or 1.0 for col in zip(*shares)]
            frac = [[v / t for v, t in zip(row, tot)] for row in shares]
            ax.stackplot(ep[:len(frac[0])], *frac, labels=labels, alpha=0.85)
            ax.set_ylim(0, 1)
            ax.set_ylabel('share of generator loss')
            ax.set_title('Loss balance  (lambda-scaled contributions)')
            _legend(ax)
        else:
            ax.text(0.5, 0.5, 'no contribution channels\n(run predates them)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, alpha=0.5)
            ax.set_title('Loss balance')
        ax.grid(alpha=0.3)

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
                'ssim', 'grad', 'freq', 'organ', 'hu_profile',
                'adv', 'fm', 'disc', 'd_real', 'd_fake',
            ] + [f'{k}_contrib' for k in CONTRIB_TERMS]}

            pbar = tqdm(train_loader, desc=f'Ep {epoch}/{epochs}', leave=False)
            for batch in pbar:
                step = self._train_step(batch)
                for k in accum:
                    accum[k].append(step.get(k, 0.0))
                pbar.set_postfix(
                    total=f"{step['gen_total']:.4f}",
                    l1=f"{step['l1']:.4f}",
                )

            # nanmean, not mean. A step reports NaN for a channel it did not
            # MEASURE, as opposed to 0.0 for one that was measured as zero — the
            # discriminator only updates every `disc_update_freq` steps, and on
            # the diffusion path the adversarial term only fires for the items
            # whose drawn timestep is under `adv_max_t`, which for a tight gate
            # is most steps. Averaging the un-measured steps in as zeros would
            # scale those curves by the measured fraction and make `train_adv`
            # look like it was decaying when only its firing rate changed.
            # Collapsed back to 0.0 for storage so history.json stays valid JSON.
            avgs = {}
            for k, v in accum.items():
                m = float(np.nanmean(v)) if v and not np.all(np.isnan(v)) else 0.0
                avgs[k] = 0.0 if m != m else m

            if self.sched_G:
                self.sched_G.step()

            val = self._validate(val_loader)

            msg = (
                f"Ep {epoch:3d}/{epochs} | "
                f"total={avgs['gen_total']:.4f}  l1={avgs['l1']:.4f}  "
                f"nll={avgs['nll']:.4f}  organ={avgs['organ']:.4f} | "
                + (f"adv={avgs['adv']:.4f}  disc={avgs['disc']:.4f}  "
                   f"D(real)={avgs['d_real']:.2f}  D(fake)={avgs['d_fake']:.2f}  "
                   f"(lam={self._adv_w():.2f}) | " if self.D is not None else "")
                +
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

            # Early stopping reads its own signal. They coincide here, but the
            # diffusion path selects on a metric that only exists on sampling
            # epochs while needing a per-epoch signal to count patience against;
            # collapsing the two would stop a still-improving run after
            # `patience` non-sampling epochs. See DiffusionTrainer.
            if self.early_stop.step(self._early_stop_score(val, score)):
                log.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        log.info(f"Done. Best val loss: {self.best_val_loss:.6f}")
