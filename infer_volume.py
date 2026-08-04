"""
Volume-level inference + phase-fidelity manifest, for all three model families.

Pixel metrics (MAE/PSNR/SSIM/NCC) reward L1 blur and can't tell whether a
generator produced the RIGHT contrast phase. To answer that we reconstruct a FULL
synthetic CECT volume and score it with the 97%-OOF CTPhase-XGBoost model (via
orgFeatXGB_CTPhase/phase_eval.py).

The model only ever saw patches, so reconstruction goes through the SAME patch
pipeline it was trained on (dataset.py geometry + HU normalisation) — no
whole-slice or organ-crop shortcuts:
  • tile the NCCT volume into patches (patch_size / overlap / patch_depth from the
    run's own run_config.json),
  • normalise each patch exactly as dataset.py does (clip [hu_min,hu_max] → [0,1]),
  • run the model (batched), stitch back with weighted overlap-blending (--blend),
  • de-normalise [0,1] → HU, save as NIfTI on the source grid.

Dims-parametric: `patch_depth==1` → 2-D per-slice; `patch_depth>1` → 3-D sub-volume.

THREE MODEL FAMILIES, ONE TILING LOOP
-------------------------------------
The tiling, blending and de-normalisation are identical for all of them; only
what happens to a batch of patches differs. That is expressed as a `PatchPredictor`
— `begin_volume()` then `__call__(batch, coords)`. Adding a model family means
adding a predictor, not touching the geometry.

  deterministic  one forward pass          → {case}_syn.nii.gz
  heteroscedastic one forward pass, 2 ch   → {case}_syn.nii.gz + {case}_std.nii.gz
  diffusion      N x (DDIM sampling loop)  → mean, std, first sample, organ medians

VOLUME-WIDE NOISE — the one structurally important thing here
-------------------------------------------------------------
A diffusion sample is a deterministic function of its initial noise x_T (at
eta=0). If every tile drew its own x_T, every tile would land at its own contrast
level: the case-level offset would average out across ~49 tiles per slice, the
variance ratio would stay near the deterministic model's 0.176, seams would
worsen, and z would flicker. Diffusion would then fail at the one thing
DIFFUSION_PLAN.md §3 chose it for — for a plumbing reason, not a scientific one.

So ONE noise field is drawn per (case, sample) at the full padded volume shape,
and each tile CROPS its x_T from it. Overlapping tiles then share noise exactly
where they overlap, and the whole volume is one coherent draw. Seeded by
(sample_seed, case_id, sample index), so a rerun reproduces it exactly.

`--sample_seed` is deliberately separate from the training seed: sampling variance
stacks on top of seed variance, and the featHU 2 sigma gate is 0.84 HU, so the two
have to be separable to attribute a difference to either.

Outputs per scenario (diffusion):
  <out_dir>/<case>_syn.nii.gz     mean of N samples  → benchmark.py / phase_eval.py
  <out_dir>/<case>_syn1.nii.gz    sample k=0         → §7's "single DDIM sample" row
  <out_dir>/<case>_std.nii.gz     per-voxel std across samples
  <out_dir>/<case>_s{k}.nii.gz    every sample, only with --save_samples
  <out_dir>/organ_medians.json    per-organ median HU per sample → calibration_eval.py
  <out_dir>/manifest.csv          the mean volume
  <out_dir>/manifest_single.csv   sample k=0, so benchmark.py scores it as its own model

Usage:
    # deterministic / heteroscedastic
    python infer_volume.py --scenario_dir <run> --split test

    # diffusion
    python infer_volume.py --scenario_dir <run> --split test \
        --n_samples 8 --ddim_steps 25 --overlap 0.25 --eta 0.0 --guidance 1.0
"""

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch

from dataset import _load_vol, find_pairs_and_split, standardised_level

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading — no full Trainer, so nothing builds an optimiser or a loss
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, cfg: Dict, device: str):
    """Return (model, kind) for whichever family this run_config describes.

    `kind` is read from the config, not guessed from the checkpoint: a
    run_config.json is the record of what was trained, and inferring the family
    from state-dict keys would silently succeed on a mismatched pair.
    """
    kind = ('diffusion' if cfg.get('use_diffusion') else
            'hetero' if cfg.get('use_hetero') else 'deterministic')
    _phases = cfg.get('target_phases') or [cfg.get('target_phase', 'venous')]

    if kind == 'diffusion':
        from models_diffusion import build_diffusion
        model, schedule = build_diffusion(cfg)
        model = model.to(device)
        schedule = schedule.to(device)
    else:
        from models import UNetGenerator
        from models_hetero import HeteroGenerator
        cls = HeteroGenerator if kind == 'hetero' else UNetGenerator
        model = cls(
            dims          = cfg.get('dims', 2),
            base_channels = cfg['generator_base_channels'],
            dropout       = cfg.get('generator_dropout', 0.2),
            norm          = cfg.get('generator_norm', 'instance'),
            in_channels   = cfg.get('in_channels', 1),
            use_phase_cond= cfg.get('use_phase_cond', False),
            n_phases      = max(2, len(_phases)),
            cond_dim      = cfg.get('phase_cond_dim', 64),
            n_levels      = len(cfg.get('cond_organs') or []),
        ).to(device)
        schedule = None

    # weights_only=False: checkpoints also carry non-tensor state (history,
    # early_stop, ...); we only read G_state but must be able to unpickle them.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['G_state'])
    model.eval()
    log.info(f"Loaded {kind} model from {ckpt_path} (epoch {state.get('epoch', '?')})")
    return model, kind, schedule


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _starts(length: int, patch: int, stride: int) -> List[int]:
    """Tile-start indices covering [0, length) with the given patch/stride,
    always including a final flush-to-edge patch so the whole extent is covered."""
    if length <= patch:
        return [0]
    s = list(range(0, length - patch + 1, stride))
    if s[-1] != length - patch:
        s.append(length - patch)
    return s


def _axis_window(n: int, mode: str, margin: int,
                 at_lo: bool, at_hi: bool) -> np.ndarray:
    """1-D blend weight along one tile axis.

    `margin` discards the outer band of the tile, where every output voxel was
    computed partly from the zero-padding the convolutions invented outside the
    tile (padding=1 nested 9 deep in the U-Net ⇒ a systematically corrupted
    band ~20-30 px wide). The margin is NOT applied on a side that lies on the
    volume boundary — there is no neighbouring tile to supply those voxels, so
    trimming them would leave the volume's own outer shell uncovered.
    """
    if mode == 'uniform':
        w = np.ones(n, np.float32)
    elif mode == 'hann':
        # n+2 then trim: a plain Hann is exactly 0 at both ends, which would make
        # the outermost row of every tile weightless and, on a border tile,
        # unrecoverable.
        w = np.hanning(n + 2)[1:-1].astype(np.float32)
    elif mode == 'gaussian':
        x = (np.arange(n, dtype=np.float32) - (n - 1) / 2) / max(1.0, (n - 1) / 2)
        w = np.exp(-0.5 * (x / 0.35) ** 2).astype(np.float32)
    else:
        raise ValueError(f'unknown blend mode: {mode}')

    if margin > 0 and n > 2 * margin:
        if not at_lo:
            w[:margin] = 0.0
        if not at_hi:
            w[-margin:] = 0.0
    return w


# ---------------------------------------------------------------------------
# Patch predictors — the only thing that differs between model families
# ---------------------------------------------------------------------------

class PatchPredictor:
    """Turn a batch of NCCT patches in [0,1] into a batch of outputs in [0,1].

    n_out          how many blended volumes come back. 1 for an image, 2 for
                   (image, per-voxel sd).
    begin_volume() called once per volume, before any tile, with the PADDED
                   (D,H,W). This is where a diffusion predictor builds its noise
                   field, which is why the hook exists at all.
    __call__       (batch (B,C,...), coords [(d0,y0,x0)]) -> (B, n_out, ...).
    """

    n_out = 1

    def begin_volume(self, padded_shape: Tuple[int, int, int]):
        pass

    def __call__(self, batch: torch.Tensor,
                 coords: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
        raise NotImplementedError


class DeterministicPredictor(PatchPredictor):
    """One forward pass. Reproduces the original behaviour exactly."""

    n_out = 1

    def __init__(self, G, device, use_amp, phase_id=None, level=None):
        self.G, self.device, self.use_amp = G, device, use_amp
        self.phase_id, self.level = phase_id, level

    def _cond_kwargs(self, B):
        kw = {}
        # Phase and level are constant across the volume, so they are broadcast
        # here rather than carried through the tiling bookkeeping.
        if getattr(self.G, 'use_phase_cond', False):
            kw['phase'] = torch.full((B,), int(self.phase_id), dtype=torch.long,
                                     device=self.device)
        if getattr(self.G, 'n_levels', 0):
            kw['level'] = (torch.as_tensor(self.level, dtype=torch.float32,
                                           device=self.device)
                           .reshape(1, -1).expand(B, -1))
        return kw

    @torch.no_grad()
    def __call__(self, batch, coords):
        with torch.autocast('cuda', enabled=self.use_amp):
            out = self.G(batch, **self._cond_kwargs(batch.shape[0]))
        return out.float()


class HeteroPredictor(DeterministicPredictor):
    """Two channels out: mu, and sd = exp(log_var/2) in [0,1] units.

    The sd field is blended with the SAME Hann weights as the mean. That is an
    approximation — a weighted mean of standard deviations is not the standard
    deviation of a weighted mean — but sigma is a smooth, slowly-varying field
    here, and the alternative (propagating variances through the blend) would
    assume the overlapping tiles' errors are independent, which they are not.
    Stated because the number is read as a calibration quantity downstream.
    """

    n_out = 2

    @torch.no_grad()
    def __call__(self, batch, coords):
        with torch.autocast('cuda', enabled=self.use_amp):
            out = self.G(batch, **self._cond_kwargs(batch.shape[0]))
        out = out.float()
        return torch.cat([out[:, :1], torch.exp(0.5 * out[:, 1:2])], dim=1)


class DiffusionPredictor(PatchPredictor):
    """DDIM sampling per tile, with x_T cropped from ONE volume-wide noise field.

    See this module's docstring for why the noise field is not per-tile. The
    field is rebuilt by `set_sample()` for each of the N samples, so sample k is
    reproducible from (sample_seed, case_id, k) alone.
    """

    n_out = 1

    def __init__(self, model, schedule, device, use_amp, cfg,
                 ddim_steps=25, eta=0.0, guidance=1.0,
                 phase_id=None, level=None, sample_seed=0, case_id=''):
        from models_diffusion import DDIMSampler, from_model, to_model
        self.model, self.schedule = model, schedule
        self.device, self.use_amp = device, use_amp
        self.to_model, self.from_model = to_model, from_model
        self.sampler = DDIMSampler(model, schedule, n_steps=ddim_steps,
                                   eta=eta, guidance=guidance)
        self.phase_id, self.level = phase_id, level
        self.pd = int(cfg.get('patch_depth', 1))
        ps = cfg['patch_size']
        self.ph, self.pw = ((int(ps), int(ps)) if isinstance(ps, int)
                            else (int(ps[0]), int(ps[1])))
        self.sample_seed = int(sample_seed)
        self.case_id = case_id
        # Read from the run's own config, never from a CLI default: a checkpoint
        # trained with offset noise and sampled without it (or the reverse) is a
        # train/test mismatch in the one channel this project measures, and it
        # would look like a modelling result rather than a flag.
        self.offset_noise = float(cfg.get('diffusion_offset_noise', 0.0))
        self.k = 0
        self._shape = None
        self._noise = None

    def begin_volume(self, padded_shape):
        self._shape = padded_shape
        self._build_noise()

    def set_sample(self, k: int):
        """Switch to sample k and redraw the whole field."""
        self.k = int(k)
        if self._shape is not None:
            self._build_noise()

    def _build_noise(self):
        # Seed from the case id as well as the sample index, so two cases never
        # share a noise field — sharing one would correlate their sampled
        # contrast levels and quietly shrink the across-case variance that §7's
        # headline metric measures.
        h = int(hashlib.sha256(self.case_id.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device='cpu').manual_seed(
            (self.sample_seed * 1_000_003 + h * 7919 + self.k) % (2 ** 63 - 1))
        # Built on CPU and kept there: a full-volume float32 field is ~80 MB and
        # only one tile's worth is needed on the device at a time.
        self._noise = torch.randn(self._shape, generator=g, dtype=torch.float32)
        if self.offset_noise:
            # ONE constant for the entire padded volume, not one per tile. Every
            # tile of a case therefore crops the same DC offset and they all
            # agree on the contrast level — the same coherence argument that
            # makes the field volume-wide at all (see the module docstring).
            # Per-tile offsets would give each tile its own level, average the
            # case-level offset away across ~49 tiles per slice, and leave the
            # variance ratio near 0.176 for a plumbing reason.
            #
            # Drawn from the same generator, after the field, so (sample_seed,
            # case_id, k) still reproduces the draw exactly.
            off = torch.randn((1,), generator=g, dtype=torch.float32)
            self._noise = self._noise + self.offset_noise * off

    def _x_T(self, coords) -> torch.Tensor:
        crops = []
        for (d0, y0, x0) in coords:
            if self.pd > 1:
                c = self._noise[d0:d0+self.pd, y0:y0+self.ph, x0:x0+self.pw]
            else:
                c = self._noise[d0, y0:y0+self.ph, x0:x0+self.pw]
            crops.append(c)
        return torch.stack(crops).unsqueeze(1).to(self.device)     # (B,1,...)

    def _cond_kwargs(self, B):
        kw = {}
        if getattr(self.model, 'use_phase_cond', False):
            kw['phase'] = torch.full((B,), int(self.phase_id), dtype=torch.long,
                                     device=self.device)
        if getattr(self.model, 'n_levels', 0):
            kw['level'] = (torch.as_tensor(self.level, dtype=torch.float32,
                                           device=self.device)
                           .reshape(1, -1).expand(B, -1))
        return kw

    @torch.no_grad()
    def __call__(self, batch, coords):
        x_T = self._x_T(coords)
        with torch.autocast('cuda', enabled=self.use_amp):
            x = self.sampler.sample(self.to_model(batch), x_T,
                                    **self._cond_kwargs(batch.shape[0]))
        return self.from_model(x.float()).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# The tiling loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_volume(predictor: PatchPredictor, vol_dhw: np.ndarray, cfg: Dict,
                 device: str, batch_size: int = 32, blend: str = 'hann',
                 edge_margin: int = 0, overlap: Optional[float] = None,
                 count_only: bool = False):
    """Reconstruct full output volume(s) (HU, (D,H,W)) from an NCCT volume.

    Returns (n_out, D, H, W) in HU for channel 0 and, where a channel is a
    standard deviation rather than a level, in HU-scale units — i.e. multiplied by
    the window width but NOT offset by hu_min, because a spread has no origin.

    Weighted overlap-blending. Patches are normalised/denormalised with the run's
    own HU window so the model sees exactly the training input distribution.

    `blend='uniform'` reproduces the original flat average, whose blend weight
    jumps discontinuously wherever a tile starts — measured at seam=1.27 against
    a seam=1.04 floor from a real untouched scan (`metrics.seam_energy`). A
    tapered window removes that discontinuity.

    NOTE this cannot fix a seam caused by the model itself: InstanceNorm rescales
    each tile by that tile's own statistics, so overlapping tiles disagree by a
    content-dependent DC offset that no blending weight can reconcile. Blending is
    necessary, not sufficient; see the norm ablation.

    `overlap` overrides the training-time value from the config — inference
    overlap is free to differ from what training used, and for diffusion it is
    normally LOWER, because the cost is n_tiles x ddim_steps x n_samples.

    `count_only=True` returns the tile count without running the model. That is
    the dry run the compute estimate needs before committing to a full pass.
    """
    pd = int(cfg.get('patch_depth', 1))
    # 2.5-D: gather 2k+1 adjacent slices as channels, still emit one slice. Output
    # depth is unchanged, so all the blending/windowing below is untouched.
    n_in = int(cfg.get('n_input_slices', 1))
    k_sl = (n_in - 1) // 2
    ps = cfg['patch_size']
    ph, pw = (int(ps), int(ps)) if isinstance(ps, int) else (int(ps[0]), int(ps[1]))
    hu_min = float(cfg.get('hu_min', -200)); hu_max = float(cfg.get('hu_max', 400))
    ov = cfg.get('overlap', 0.5) if overlap is None else float(overlap)
    ratio = 1.0 - ov
    sh = max(1, int(ph * ratio)); sw = max(1, int(pw * ratio))
    sd = max(1, int(pd * ratio)) if pd > 1 else 1

    D, H, W = vol_dhw.shape
    # Pad tiled dims up to patch size if a volume is smaller (rare; 3-D depth).
    pad_d = max(0, pd - D) if pd > 1 else 0
    pad_h = max(0, ph - H); pad_w = max(0, pw - W)
    if pad_d or pad_h or pad_w:
        vol_dhw = np.pad(vol_dhw, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
    Dp, Hp, Wp = vol_dhw.shape

    d_starts = _starts(Dp, pd, sd) if pd > 1 else list(range(Dp))
    h_starts = _starts(Hp, ph, sh)
    w_starts = _starts(Wp, pw, sw)
    n_tiles = len(d_starts) * len(h_starts) * len(w_starts)
    if count_only:
        return n_tiles, (len(h_starts) * len(w_starts))

    predictor.begin_volume((Dp, Hp, Wp))
    n_out = predictor.n_out

    # Normalise once (elementwise → identical to per-patch clip+rescale).
    vn = np.clip(vol_dhw, hu_min, hu_max)
    vn = ((vn - hu_min) / (hu_max - hu_min)).astype(np.float32)

    out_sum = np.zeros((n_out, Dp, Hp, Wp), np.float32)
    out_cnt = np.zeros((Dp, Hp, Wp), np.float32)

    # Blend windows depend only on which volume edges a tile touches, so there are
    # at most a handful of distinct ones — build each once and reuse.
    _wcache: Dict = {}

    def _window(d0, y0, x0):
        key = (d0 == 0, d0 + pd >= Dp, y0 == 0, y0 + ph >= Hp, x0 == 0, x0 + pw >= Wp)
        if key not in _wcache:
            wh = _axis_window(ph, blend, edge_margin, key[2], key[3])
            ww = _axis_window(pw, blend, edge_margin, key[4], key[5])
            if pd > 1:
                wd = _axis_window(pd, blend, edge_margin, key[0], key[1])
                w = wd[:, None, None] * wh[None, :, None] * ww[None, None, :]
            else:
                w = wh[:, None] * ww[None, :]
            _wcache[key] = w.astype(np.float32)
        return _wcache[key]

    buf_patch, buf_coord = [], []

    def _flush():
        if not buf_patch:
            return
        batch = torch.from_numpy(np.stack(buf_patch)).to(device)
        # 2.5-D patches already carry the slice stack in the channel axis.
        if k_sl == 0:
            batch = batch.unsqueeze(1)                                  # (B,1,...)
        out = predictor(batch, buf_coord).cpu().numpy()                 # (B,n_out,...)
        for (d0, y0, x0), o in zip(buf_coord, out):
            w = _window(d0, y0, x0)
            if pd > 1:
                out_sum[:, d0:d0+pd, y0:y0+ph, x0:x0+pw] += o * w
                out_cnt[d0:d0+pd, y0:y0+ph, x0:x0+pw] += w
            else:
                out_sum[:, d0, y0:y0+ph, x0:x0+pw] += o * w
                out_cnt[d0, y0:y0+ph, x0:x0+pw] += w
        buf_patch.clear(); buf_coord.clear()

    for d0 in d_starts:
        for y0 in h_starts:
            for x0 in w_starts:
                if pd > 1:
                    patch = vn[d0:d0+pd, y0:y0+ph, x0:x0+pw]     # (pd,ph,pw)
                elif k_sl:
                    # Edge-clamped, matching dataset.py's _crop_src exactly, so
                    # the model sees the same boundary behaviour it trained on.
                    zs = np.clip(np.arange(d0 - k_sl, d0 + k_sl + 1), 0, Dp - 1)
                    patch = vn[zs, y0:y0+ph, x0:x0+pw]           # (2k+1,ph,pw)
                else:
                    patch = vn[d0, y0:y0+ph, x0:x0+pw]           # (ph,pw)
                buf_patch.append(patch); buf_coord.append((d0, y0, x0))
                if len(buf_patch) >= batch_size:
                    _flush()
    _flush()

    # With edge-aware margins every voxel keeps positive weight, so a zero here
    # means a real coverage bug rather than a benign edge case — say so instead of
    # silently emitting hu_min.
    n_zero = int((out_cnt <= 0).sum())
    if n_zero:
        log.error(f"  {n_zero} voxel(s) received no tile weight "
                  f"(blend={blend}, edge_margin={edge_margin}) — output invalid there; "
                  f"reduce --edge_margin or raise --overlap")
        out_cnt[out_cnt <= 0] = 1.0
    syn01 = out_sum / out_cnt[None]
    span = hu_max - hu_min
    out_hu = syn01 * span
    # Channel 0 is a LEVEL and needs the window offset; channels 1+ are SPREADS,
    # scaled by the window width but never offset — a standard deviation has no
    # origin, and adding hu_min to one would make it negative.
    out_hu[0] += hu_min
    return out_hu[:, :D, :H, :W]        # crop away any padding


# ---------------------------------------------------------------------------
# Organ medians — the quantity calibration_eval.py scores
# ---------------------------------------------------------------------------

def _organ_label_map(cfg: Dict) -> Tuple[Dict[str, int], List[str]]:
    """({organ_name: label_id}, ordered organ names) for the 16 organs the phase
    model reads. The ORDER is the model's trained feature order, imported rather
    than re-listed so this and featHU can never score a different organ set."""
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / 'orgFeatXGB_CTPhase'))
    from organ_features import ORGANS
    p = Path(cfg.get('organ_label_map_json') or
             here / 'orgFeatXGB_CTPhase' / 'retrain_out_full' / 'ts_label_map_total.json')
    name_to_id = json.loads(p.read_text())
    return {o: int(name_to_id[o]) for o in ORGANS if o in name_to_id}, list(ORGANS)


def organ_medians(vol_hu: np.ndarray, mask: np.ndarray,
                  label_map: Dict[str, int], organs: List[str],
                  min_voxels: int = 64) -> List[float]:
    """Per-organ median HU, in the phase model's organ order.

    Median, not mean, because that is exactly what the XGBoost phase model reads
    and therefore what featHU is computed from. Absent or tiny organs give NaN,
    which propagates rather than being imputed — an imputed level would be
    indistinguishable from a predicted one downstream.
    """
    out = []
    for o in organs:
        lid = label_map.get(o)
        if lid is None:
            out.append(float('nan')); continue
        sel = mask == lid
        out.append(float(np.median(vol_hu[sel])) if sel.sum() >= min_voxels
                   else float('nan'))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _dry_run(sdir: Path, cfg: Dict, pairs: List[Dict], split: str, device: str,
             overlap: Optional[float], ddim_steps: int, n_samples: int,
             guidance: float):
    """Tile count and sampling budget, without touching the model.

    Deliberately reachable WITHOUT a checkpoint. Tile counts come from the source
    volume shapes and the run config alone, and the whole point of "run --dry_run
    first" is to size the budget before committing to it — which has to work while
    the run is still training, or before you have decided which checkpoint to
    sample with. It used to sit after `load_model`, so the one instruction the
    docs give as a pre-flight check needed the thing it was meant to precede.
    """
    kind = ('diffusion' if cfg.get('use_diffusion') else
            'hetero' if cfg.get('use_hetero') else 'deterministic')
    ov = overlap if overlap is not None else cfg.get('overlap', 0.5)
    log.info(f"[dry run] {len(pairs)} {split} case(s) for scenario '{sdir.name}' "
             f"[{kind}] at overlap={ov}")

    total = n_skipped = 0
    for pair in pairs:
        if not pair.get('seg_path'):
            n_skipped += 1          # same skip rule as the real loop
            continue
        src_dhw = _load_vol(pair['source_path'])
        n_tiles, per_slice = infer_volume(PatchPredictor(), src_dhw, cfg, device,
                                          overlap=overlap, count_only=True)
        total += n_tiles
        log.info(f"  {pair['case_id']}: shape {src_dhw.shape} → {n_tiles} tiles "
                 f"({per_slice} per slice)")

    # For a deterministic or hetero run a tile IS a forward pass. For diffusion the
    # budget is the number that decides whether the run is affordable at all.
    if kind == 'diffusion':
        passes = total * ddim_steps * n_samples * (2 if guidance != 1.0 else 1)
        log.info(f"[dry run] {total} tiles → {passes:,} forward passes "
                 f"(x{ddim_steps} ddim_steps x{n_samples} n_samples"
                 + (", x2 for guidance != 1" if guidance != 1.0 else "") + ")")
    else:
        log.info(f"[dry run] {total} tiles → {total:,} forward passes")
    if n_skipped:
        log.info(f"[dry run] {n_skipped} case(s) skipped: no organ mask")


def run(scenario_dir: str, split: str, out_dir: str, ckpt_name: str,
        batch_size: int, device: str, blend: str = 'hann',
        edge_margin: int = 0, overlap: Optional[float] = None,
        level_mode: str = 'oracle', level: Optional[float] = None,
        n_samples: int = 1, sample_seed: int = 0, ddim_steps: int = 25,
        eta: float = 0.0, guidance: float = 1.0, save_samples: bool = False,
        dry_run: bool = False):
    sdir = Path(scenario_dir)
    cfg = json.loads((sdir / 'run_config.json').read_text())

    train_pairs, val_pairs, test_pairs = find_pairs_and_split(cfg)
    pairs = {'val': val_pairs, 'test': test_pairs,
             'both': val_pairs + test_pairs}[split]

    if dry_run:
        _dry_run(sdir, cfg, pairs, split, device, overlap, ddim_steps, n_samples,
                 guidance)
        return

    ckpt = sdir / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model, kind, schedule = load_model(str(ckpt), cfg, device)
    use_amp = (device == 'cuda')

    log.info(f"Inferring {len(pairs)} {split} case(s) for scenario '{sdir.name}' "
             f"[{kind}]")

    if kind != 'diffusion' and n_samples > 1:
        raise ValueError(f"--n_samples {n_samples} is only meaningful for a "
                         f"diffusion run; this is a {kind} model, whose output "
                         f"is deterministic.")

    # ── level conditioning at inference ──────────────────────────────────────
    # Same checkpoint, three modes. The REPORTABLE RESULT is the gap between
    # 'oracle' and 'population' — oracle reads the answer off the real CECT and
    # can never be a headline number on its own.
    #
    #   oracle      the case's true standardised level      -> the ceiling
    #   population  all zeros == the training-set mean      -> must match baseline
    #   fixed       --level L, standardised, for every case -> the deployable mode
    cond_organs = list(cfg.get('cond_organs') or [])
    levels_db = None
    if cond_organs:
        if level_mode not in ('oracle', 'population', 'fixed'):
            raise ValueError(f"unknown level_mode {level_mode!r}")
        lp = Path(cfg.get('levels_json', 'splits/levels.json'))
        if level_mode == 'oracle' and not lp.exists():
            raise FileNotFoundError(f"level_mode=oracle needs {lp}")
        if lp.exists():
            levels_db = json.loads(lp.read_text())
        log.info(f"level conditioning: {len(cond_organs)} organ(s), mode={level_mode}"
                 + (f", level={level}" if level_mode == 'fixed' else ""))

    def _level_for(case_id: str, phase: str):
        if not cond_organs:
            return None
        n = len(cond_organs)
        if level_mode == 'population':
            return np.zeros(n, np.float32)      # standardised mean
        idx = [levels_db['organs'].index(o) for o in cond_organs]
        st = levels_db['standardize']
        if level_mode == 'fixed':
            return np.array([(level - st['mean'][i]) / (st['std'][i] or 1.0)
                             for i in idx], np.float32)
        return np.array(standardised_level(
            levels_db, case_id, phase, cond_organs, idx,
            [st['mean'][i] for i in idx],
            [st['std'][i] or 1.0 for i in idx]), np.float32)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    default_phase = cfg.get('target_phase', 'venous')
    # One manifest per phase. A multi-phase model emits several volumes per case,
    # which would collide on '{case_id}_syn.nii.gz' in a single directory, and
    # each phase has to be phase-scored against its own target anyway. Runs with
    # one phase keep exactly the previous flat layout.
    phases = sorted({p.get('phase', default_phase) for p in pairs})
    multi = len(phases) > 1
    rows_by_phase = {ph: [] for ph in phases}
    single_by_phase = {ph: [] for ph in phases}
    n_no_mask = 0

    label_map, organs = _organ_label_map(cfg) if kind == 'diffusion' else ({}, [])
    med_out = {'organs': organs, 'n_samples': int(n_samples), 'cases': {}}

    for pair in pairs:
        case_id = pair['case_id']
        seg_path = pair.get('seg_path')
        ph = pair.get('phase', default_phase)
        if not seg_path:
            n_no_mask += 1
            log.warning(f"  {case_id} [{ph}]: no {cfg.get('seg_suffix','_seg_reg')} "
                        f"mask — skipping (can't phase-score)")
            continue
        pdir = (out / ph) if multi else out
        pdir.mkdir(parents=True, exist_ok=True)
        src_nii = nib.load(pair['source_path'])
        src_dhw = _load_vol(pair['source_path'])
        lv = _level_for(case_id, ph)

        common = dict(cfg=cfg, device=device, batch_size=batch_size, blend=blend,
                      edge_margin=edge_margin, overlap=overlap)

        if kind == 'diffusion':
            pred = DiffusionPredictor(model, schedule, device, use_amp, cfg,
                                      ddim_steps=ddim_steps, eta=eta,
                                      guidance=guidance,
                                      phase_id=pair.get('phase_id', 0), level=lv,
                                      sample_seed=sample_seed, case_id=case_id)
            mask = _load_vol(seg_path)
            mask = np.round(mask).astype(np.int32)
            # Welford over samples: N full volumes at once is ~80 MB each and the
            # only thing needed from them is the mean, the variance, sample 0 and
            # each sample's organ medians — all of which stream.
            mean = None; m2 = None; meds = []
            t0 = time.time()
            for k in range(n_samples):
                pred.set_sample(k)
                syn = infer_volume(pred, src_dhw, **common)[0]     # (D,H,W) HU
                meds.append(organ_medians(syn, mask, label_map, organs))
                if k == 0:
                    _save(pdir / f'{case_id}_syn1.nii.gz', syn, src_nii)
                    mean = syn.astype(np.float64); m2 = np.zeros_like(mean)
                else:
                    d = syn - mean
                    mean += d / (k + 1)
                    m2 += d * (syn - mean)
                if save_samples:
                    _save(pdir / f'{case_id}_s{k}.nii.gz', syn, src_nii)
            syn_mean = mean.astype(np.float32)
            std = (np.sqrt(m2 / max(1, n_samples - 1)).astype(np.float32)
                   if n_samples > 1 else np.zeros_like(syn_mean))
            _save(pdir / f'{case_id}_syn.nii.gz', syn_mean, src_nii)
            _save(pdir / f'{case_id}_std.nii.gz', std, src_nii)
            med_out['cases'][case_id] = {
                'real': organ_medians(_load_vol(pair['target_path']), mask,
                                      label_map, organs),
                'samples': meds,
            }
            single_by_phase[ph].append({
                'gen_path': str(pdir / f'{case_id}_syn1.nii.gz'),
                'real_path': pair['target_path'],
                'mask_path': seg_path, 'target_phase': ph})
            log.info(f"  {case_id} [{ph}]: {n_samples} sample(s) in "
                     f"{time.time() - t0:.0f}s → mean/std/organ medians")
        else:
            pred = (HeteroPredictor if kind == 'hetero' else DeterministicPredictor)(
                model, device, use_amp, phase_id=pair.get('phase_id', 0), level=lv)
            res = infer_volume(pred, src_dhw, **common)
            _save(pdir / f'{case_id}_syn.nii.gz', res[0], src_nii)
            if kind == 'hetero':
                _save(pdir / f'{case_id}_std.nii.gz', res[1], src_nii)
            log.info(f"  {case_id} [{ph}]: saved {case_id}_syn.nii.gz")

        rows_by_phase[ph].append({'gen_path': str(pdir / f'{case_id}_syn.nii.gz'),
                                  'real_path': pair['target_path'],
                                  'mask_path': seg_path, 'target_phase': ph})

    for ph, rows in rows_by_phase.items():
        pdir = (out / ph) if multi else out
        _write_manifest(pdir / 'manifest.csv', rows)
        log.info(f"Wrote {len(rows)} rows → {pdir / 'manifest.csv'}"
                 + (f"  ({n_no_mask} skipped, no mask)" if n_no_mask else ""))
        if single_by_phase[ph]:
            _write_manifest(pdir / 'manifest_single.csv', single_by_phase[ph])
            log.info(f"Wrote {len(single_by_phase[ph])} rows → "
                     f"{pdir / 'manifest_single.csv'} (single DDIM sample, §7)")
        log.info(f"Next: python orgFeatXGB_CTPhase/phase_eval.py "
                 f"--weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl "
                 f"--manifest {pdir/'manifest.csv'} --gen_in_hu "
                 f"--hu_min {cfg.get('hu_min',-200)} --hu_max {cfg.get('hu_max',400)} "
                 f"--out_json {pdir/'phase_eval_report.json'}")

    if kind == 'diffusion':
        (out / 'organ_medians.json').write_text(json.dumps(med_out, indent=2))
        log.info(f"Wrote per-organ medians for {len(med_out['cases'])} case(s) → "
                 f"{out / 'organ_medians.json'}")
        log.info(f"Next: python scripts/calibration_eval.py --mode ensemble "
                 f"--dir {out} --out analysis/calibration_{sdir.name}.json")


def _save(path: Path, vol_dhw: np.ndarray, src_nii):
    """Write a (D,H,W) volume on the source grid.

    Back to native (X,Y,Z) orientation to match the real/mask volumes that
    phase_eval loads raw.
    """
    xyz = np.transpose(vol_dhw, (2, 1, 0)).astype(np.float32)
    nib.save(nib.Nifti1Image(xyz, src_nii.affine, src_nii.header), str(path))


def _write_manifest(path: Path, rows: List[Dict]):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path', 'mask_path',
                                          'target_phase'])
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Volume inference + phase-fidelity manifest')
    ap.add_argument('--scenario_dir', required=True,
                    help='a trained scenario output dir (has run_config.json + best_model.pth)')
    ap.add_argument('--split', default='test', choices=['val', 'test', 'both'])
    ap.add_argument('--out_dir', default=None, help='default: <scenario_dir>/phase_infer')
    ap.add_argument('--ckpt_name', default='best_model.pth')
    ap.add_argument('--level_mode', default='oracle',
                    choices=['oracle', 'population', 'fixed'],
                    help="oracle = the case's true level (the CEILING, leaks the "
                         "target); population = training mean (must reproduce the "
                         "unconditioned baseline); fixed = --level for every case. "
                         "Report the oracle-vs-population GAP, never oracle alone.")
    ap.add_argument('--level', type=float, default=None,
                    help='HU value for --level_mode fixed')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--blend', default='hann', choices=['uniform', 'hann', 'gaussian'],
                    help="tile blend window. 'uniform' is the original flat average "
                         "(measured seam=1.27 vs a 1.04 floor); default 'hann' tapers "
                         "the weight so it is continuous across a tile boundary.")
    ap.add_argument('--edge_margin', type=int, default=0,
                    help='discard this many voxels from each tile side before '
                         'blending (the conv zero-padding band). Not applied on '
                         'sides lying on the volume boundary. Needs enough overlap '
                         'to stay covered: margin < patch_size*overlap/2.')
    ap.add_argument('--overlap', type=float, default=None,
                    help='override the training-time overlap for inference only. '
                         'For diffusion the cost is n_tiles x ddim_steps x '
                         'n_samples, so 0.25 rather than 0.5 is the default '
                         'starting point.')

    g = ap.add_argument_group('diffusion sampling')
    g.add_argument('--n_samples', type=int, default=1,
                   help='draws per case; each gets its own volume-wide noise '
                        'field. CRPS and the variance ratio work at any N, but a '
                        '90%% coverage interval is not EXPRESSIBLE below N=19 '
                        '(metrics.min_samples_for_level) and is reported as NaN. '
                        'Use 8 to explore, 20 for the calibration run.')
    g.add_argument('--sample_seed', type=int, default=0,
                   help='SEPARATE from the training seed on purpose: sampling '
                        'variance stacks on seed variance and the featHU 2-sigma '
                        'gate is 0.84 HU, so the two must be separable.')
    g.add_argument('--ddim_steps', type=int, default=25)
    g.add_argument('--eta', type=float, default=0.0,
                   help='0 = deterministic DDIM (keeps the volume-wide noise '
                        'field coherent); 1 = ancestral.')
    g.add_argument('--guidance', type=float, default=1.0,
                   help='classifier-free guidance scale. 1.0 = plain conditional '
                        'and one forward pass; anything else costs two. Sweep it '
                        'to trace the var-ratio-vs-featHU trade-off; do not tune it.')
    g.add_argument('--save_samples', action='store_true',
                   help='also write every individual sample volume — needed for '
                        'the per-voxel block of calibration_eval.py.')
    ap.add_argument('--dry_run', action='store_true',
                    help='print the tile count per volume and the total forward-pass '
                         'budget, then exit. Run this FIRST: the sampling cost '
                         'estimate depends on it. Needs only run_config.json and the '
                         'data — NOT a checkpoint, so it works mid-training.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or str(Path(args.scenario_dir) / 'phase_infer')
    run(args.scenario_dir, args.split, out_dir, args.ckpt_name, args.batch_size,
        device, blend=args.blend, edge_margin=args.edge_margin,
        overlap=args.overlap, level_mode=args.level_mode, level=args.level,
        n_samples=args.n_samples, sample_seed=args.sample_seed,
        ddim_steps=args.ddim_steps, eta=args.eta, guidance=args.guidance,
        save_samples=args.save_samples, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
