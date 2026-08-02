"""
Self-contained paired NCCT/CECT dataset.

Follows the autoenc_fresh pattern exactly:
  1. Index all valid patch coordinates across all pairs.
  2. Randomly sub-sample up to max_patches.
  3. Preload BOTH source and target patches into a Python list in RAM.
  __getitem__ is a plain list lookup — zero disk I/O during training.

Supports 2-D slices (patch_depth=1, default) and 3-D patches (patch_depth>1)
via a single flag so the model can be extended to 3D without changing this file.

HU normalisation follows autoenc_fresh:
  clip [hu_min, hu_max] → rescale to [0, 1]

Validity filter (applied in HU space on the SOURCE/NCCT patch):
  patch.std()  >= min_patch_std
  patch.mean() >= min_patch_mean
  max(src,tgt).max() >= min_patch_max   (at least some tissue present)
"""

import functools
import hashlib
import json
import logging
from random import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume loader — LRU-cached, no duplication of disk reads
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _load_vol(path: str) -> np.ndarray:
    """Return (D, H, W) float32 volume in original HU values."""
    vol = nib.load(path).get_fdata().astype(np.float32)
    return np.transpose(vol, (2, 1, 0))          # (X,Y,Z) → (D,H,W)


# ---------------------------------------------------------------------------
# Phase inference (fallback when no CSV provided)
# ---------------------------------------------------------------------------

_PHASE_KW: Dict[str, List[str]] = {
    'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline',
                     'native', 'nc', 'nce', 'noncon'],
    'arterial':     ['arterial', 'art', 'early', 'phase1', 'p1'],
    'venous':       ['venous', 'portal', 'pv', 'phase2', 'p2', 'late'],
    'delayed':      ['delayed', 'delay', 'equilibrium', 'phase3', 'p3'],
}

def _infer_phase(name: str) -> str:
    n = name.lower()
    for phase, kws in _PHASE_KW.items():
        if any(k in n for k in kws):
            return phase
    return 'unknown'


# ---------------------------------------------------------------------------
# Pair finder — replicates find_and_split from autoenc_fresh but for pairs
# ---------------------------------------------------------------------------

def find_pairs_and_split(
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Scan data_dir, match NCCT + target-phase volumes in each case directory,
    then split CASES into train / val / test.

    Returns (train_pairs, val_pairs, test_pairs)
    Each pair is {'source_path', 'target_path', 'seg_path', 'case_id',
                  'phase', 'phase_id'}

    Multi-phase (`cfg['target_phases'] = ['venous', 'arterial']`) emits one pair
    per (case, available phase). Two properties this function must guarantee:

      * NO PATIENT LEAKAGE. The split is over cases, and every phase of a case
        lands in the same split. Splitting the expanded pair list instead would
        put a patient's arterial scan in train and their venous in test.
      * The single-phase split is UNCHANGED. The case list is built from cases
        having NCCT + phases[0] and permuted exactly as before, so a run with
        `target_phases=['venous']` reproduces the existing 97/20/20 split
        case-for-case. That is what keeps multi-phase runs comparable to every
        result already in analysis/.
    """
    data_dir     = Path(cfg['data_dir'])
    # 'target_phases' (list) generalises 'target_phase' (scalar); the scalar is
    # still what config.py and every existing run_config.json set.
    target_phases = cfg.get('target_phases') or [cfg.get('target_phase', 'venous')]
    if isinstance(target_phases, str):
        target_phases = [target_phases]
    target_phase = target_phases[0]
    file_tag     = cfg.get('file_tag', '_deeds')
    seg_suffix   = cfg.get('seg_suffix', '_seg_reg')   # e.g. '_seg_full' for the
                                                        # regenerated full TS masks
    val_split    = cfg.get('val_split',  0.15)
    test_split   = cfg.get('test_split', 0.15)
    seed         = cfg.get('data_seed', cfg.get('seed', 42))

    # Optional labels CSV
    phase_map: Dict[str, Dict[str, str]] = {}
    labels_csv = cfg.get('labels_csv', '')
    if labels_csv and Path(labels_csv).exists():
        import pandas as pd
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            case  = str(row['StudyInstanceUID'])
            ser   = str(row['SeriesInstanceUID'])
            label = str(row['Label']).lower()
            phase_map.setdefault(case, {})[ser] = label
        log.info(f"Loaded labels CSV: {len(phase_map)} cases")

    pairs: List[Dict] = []
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name

        # Find all registered volumes, skip segmentation masks
        vols: Dict[str, str] = {}
        for f in sorted(case_dir.glob(f'*{file_tag}.nii.gz')):
            if '_seg' in f.name:
                continue
            stem = f.name.replace(f'{file_tag}.nii.gz', '').replace('.nii', '')
            parts = stem.split('_')
            series_id = parts[1] if len(parts) >= 2 else stem

            # Phase from CSV or filename
            if case_id in phase_map and series_id in phase_map[case_id]:
                phase = phase_map[case_id][series_id]
            else:
                phase = _infer_phase(f.name)

            vols[phase] = str(f)

        # Case membership is decided by phases[0] alone, so adding a second phase
        # never changes which cases exist or in what order — only how many pairs
        # each case contributes.
        if 'non-contrast' in vols and target_phase in vols:
            case_pairs = []
            for pid, ph in enumerate(target_phases):
                if ph not in vols:
                    continue                     # e.g. the one case with no arterial
                target_path = vols[ph]
                # Organ/vessel segmentation mask for target_path (same grid).
                # '..._deeds.nii.gz' -> '..._deeds{seg_suffix}.nii.gz' — use
                # '_seg_full' for the regenerated full TS masks (aorta/heart/IVC).
                seg_path = target_path.replace(f'{file_tag}.nii.gz',
                                               f'{file_tag}{seg_suffix}.nii.gz')
                case_pairs.append({
                    'source_path': vols['non-contrast'],
                    'target_path': target_path,
                    'seg_path':    seg_path if Path(seg_path).exists() else None,
                    'case_id':     case_id,
                    'phase':       ph,
                    'phase_id':    pid,
                })
            pairs.append(case_pairs)             # grouped by case, flattened below

    if not pairs:
        log.error(
            f"No NCCT/{target_phase} pairs found under {data_dir}\n"
            f"  file_tag='{file_tag}', phase_map loaded={bool(phase_map)}\n"
            f"  Tip: check that '_deeds.nii.gz' files exist and labels.csv has "
            f"  'non-contrast' and '{target_phase}' labels."
        )
        return [], [], []

    n_pairs = sum(len(cp) for cp in pairs)
    log.info(f"Found {len(pairs)} cases with NCCT/{target_phase}"
             + (f" — {n_pairs} pairs over phases {target_phases}"
                if len(target_phases) > 1 else f" ({n_pairs} pairs)"))

    # Case-level split. `pairs` is a list of PER-CASE lists, so permuting it
    # keeps every phase of a case together — the leakage guarantee. Split sizes
    # are computed on the CASE count, and with one phase that is the pair count,
    # so the single-phase split is bit-identical to before.
    n_test = max(1, int(len(pairs) * test_split))
    n_val  = max(1, int(len(pairs) * val_split))
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(pairs))

    def _flat(sel):
        return [p for i in sel for p in pairs[i]]

    test_pairs  = _flat(idx[:n_test])
    val_pairs   = _flat(idx[n_test: n_test + n_val])
    train_pairs = _flat(idx[n_test + n_val:])

    log.info(f"  cases  train={len(idx) - n_test - n_val}  val={n_val}  test={n_test}")
    log.info(f"  pairs  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Dataset — index → subsample → RAM preload
# ---------------------------------------------------------------------------

class CTPairDataset(Dataset):
    """
    Paired NCCT/CECT patch dataset with full RAM preloading.

    Patch shape returned:
      patch_depth=1  → (1, H, W)       2-D slice
      patch_depth>1  → (1, D, H, W)    3-D sub-volume

    Validity is checked in HU space on the SOURCE (NCCT) patch.
    Both source and target are then clipped and normalised to [0, 1].

    Organ/vessel segmentation mask (from 'seg_path') is only read from disk
    and preloaded if cfg['use_organ'] or cfg['use_seg_consistency'] is True;
    __getitem__ then also returns a 'mask' key (binarised, > 0).

    Args:
        pairs        List of {'source_path', 'target_path', 'seg_path',
                     'case_id'} dicts ('seg_path' may be None).
        cfg          Config dict (see config.py for all keys).
        max_patches  Maximum patches to preload into RAM.
                     None = use all valid patches.
        split_name   'train' | 'val' | 'test' (for logging only).
    """

    def __init__(
        self,
        pairs:       List[Dict],
        cfg:         Dict,
        max_patches: Optional[int] = None,
        split_name:  str = 'split',
    ):
        self.cfg        = cfg
        self.split_name = split_name

        # Patch geometry
        ps = cfg['patch_size']
        if isinstance(ps, int):
            self.ph = self.pw = ps
        else:
            self.ph, self.pw = int(ps[0]), int(ps[1])

        self.patch_depth = int(cfg.get('patch_depth', 1))   # 1 = 2D, >1 = 3D
        self.half_h = self.ph // 2
        self.half_w = self.pw // 2
        self.half_d = self.patch_depth // 2

        # 2.5-D: feed 2k+1 adjacent axial slices as CHANNELS and predict the
        # centre slice. The source patch is (2k+1, H, W) while the target stays
        # (H, W) — the one place in this file where src and tgt shapes differ.
        # Motivation: the aorta runs 258 mm in z but patch_depth=1 shows the
        # model 1.5 mm of it (../synthetic_CECT/PROJECT_PLAN.md 1.5).
        # Level conditioning. `cond_organs` names which organs' median HU form the
        # conditioning vector; the values come from splits/levels.json, which is
        # ORACLE information read from the real CECT (see ../synthetic_CECT/scripts/dump_levels.py).
        # Missing entries fall back to the population mean, which after
        # standardisation is a vector of zeros.
        self.cond_organs = list(cfg.get('cond_organs') or [])
        self._levels = None
        if self.cond_organs:
            lp = Path(cfg.get('levels_json', 'splits/levels.json'))
            if not lp.exists():
                raise FileNotFoundError(
                    f"cond_organs={self.cond_organs} needs {lp} — run "
                    f"`../synthetic_CECT/scripts/dump_levels.py` first")
            self._levels = json.loads(lp.read_text())
            missing = [o for o in self.cond_organs if o not in self._levels['organs']]
            if missing:
                raise ValueError(f"cond_organs not in {lp}: {missing}")
            self._lv_idx = [self._levels['organs'].index(o) for o in self.cond_organs]
            st = self._levels['standardize']
            self._lv_mean = [st['mean'][i] for i in self._lv_idx]
            self._lv_std  = [st['std'][i] or 1.0 for i in self._lv_idx]

        self.n_input_slices = int(cfg.get('n_input_slices', 1))
        if self.n_input_slices % 2 == 0:
            raise ValueError(f"n_input_slices must be odd, got {self.n_input_slices}")
        self.k_slices = (self.n_input_slices - 1) // 2
        if self.k_slices and self.patch_depth != 1:
            raise ValueError("n_input_slices>1 is a 2.5-D mode and needs patch_depth=1; "
                             f"got patch_depth={self.patch_depth}")

        stride_ratio = 1.0 - cfg.get('overlap', 0.5)
        self.stride_h = max(1, int(self.ph * stride_ratio))
        self.stride_w = max(1, int(self.pw * stride_ratio))

        # Validity thresholds (HU space)
        self.min_std  = cfg.get('min_patch_std',  10.0)
        self.min_mean = cfg.get('min_patch_mean', -800.0)
        self.min_max  = cfg.get('min_patch_max',  -500.0)

        # Normalisation
        self.hu_min = float(cfg.get('hu_min', -200))
        self.hu_max = float(cfg.get('hu_max',  300))

        # Organ-focused sampling: bias a fraction of patch centres onto organ/
        # vessel voxels so patches actually CONTAIN the structures the organ and
        # phase-consistency losses care about (a uniform 128² grid drops most
        # patches on parenchyma/fat/bowel and almost never on the aorta/portal
        # vein/IVC, making those losses near-degenerate). 0.0 = legacy uniform
        # grid behaviour. `organ_focus_labels` (list of int label ids) restricts
        # the focus to specific organs when the mask is TotalSegmentator
        # MULTILABEL; None → any segmented voxel (mask > 0).
        self.organ_focus_frac   = float(cfg.get('organ_focus_frac', 0.0))
        self.organ_focus_labels = cfg.get('organ_focus_labels', None)
        self.max_focus_cand     = int(cfg.get('max_focus_candidates_per_vol', 3000))

        # Organ/vessel mask is only loaded from disk if a loss that consumes it
        # is enabled, OR organ-focused sampling needs it to place patch centres,
        # OR organ-region validation metrics are requested (val/test only — the
        # train split never needs masks just for metric reporting, so we don't
        # pay the extra RAM/cache there).
        report_organ_metrics = cfg.get('report_organ_metrics', False) and \
            split_name in ('val', 'test')
        self.load_mask = bool(cfg.get('use_organ', False)
                              or cfg.get('use_seg_consistency', False)
                              or self.organ_focus_frac > 0.0
                              or report_organ_metrics)

        # Keep the mask MULTI-LABEL (raw TotalSegmentator ids, not binarised >0)
        # when per-organ metrics or label-restricted organ-focus need to tell
        # organs apart. SegmentationConsistencyLoss does `mask.clamp(0,1)`, so a
        # multi-label mask still behaves as a binary foreground for it.
        #
        # `organ_weights` is the reason this also has to hold on the TRAIN split:
        # per-organ loss weights key on the raw label id, and a binarised mask
        # would collapse every organ onto the single label-1 weight — silently
        # applying the aorta's weight to bowel, bone and everything else. The two
        # conditions above are val/test-only, so without this the per-organ
        # weighting simply could not work where it matters.
        self.mask_multilabel = bool(report_organ_metrics
                                    or self.organ_focus_labels is not None
                                    or cfg.get('organ_weights'))

        rng = np.random.default_rng(cfg.get('data_seed', cfg.get('seed', 42)))
        self._rng = rng

        # ── Cache: skip indexing + preload entirely if a prior run already
        # produced the identical patch set (same source/target files, patch
        # geometry, validity thresholds, HU window, subsample size and mask
        # requirement). Keyed independently of `output_dir` so it's shared
        # across scenario runs that only differ in loss flags.
        self._cache_file = self._cache_path(pairs, cfg, max_patches)
        if self._cache_file is not None and self._cache_file.exists():
            log.info(f"[{split_name}] Loading cached patches from {self._cache_file}")
            data = np.load(self._cache_file)
            self.src_patches  = list(data['src'])
            self.tgt_patches  = list(data['tgt'])
            self.mask_patches = list(data['mask']) if self.load_mask and 'mask' in data else \
                                 ([] if self.load_mask else None)
            # Per-patch provenance (case id + source slice), used to draw sample
            # grids from DISTINCT cases. Deliberately NOT part of the cache key:
            # caches written before this existed stay valid and simply yield no
            # provenance, rather than forcing a full re-preload.
            self.case_ids = list(data['case_id']) if 'case_id' in data else None
            self.patch_z  = list(data['patch_z'])  if 'patch_z'  in data else None
            # Pre-multi-phase caches carry no phase; they are single-phase by
            # construction, so phase 0 for every patch is the correct reading.
            self.phase_ids = (list(data['phase_id']) if 'phase_id' in data
                              else [0] * len(self.src_patches))
            self.level_vecs = (list(data['level']) if 'level' in data else [])
            if self.case_ids is None:
                log.info(f"  [{split_name}] cache predates per-patch provenance — "
                         f"sample grids will be random but unlabelled "
                         f"(delete the cache to regenerate with case labels)")
            log.info(f"  [{split_name}] {len(self.src_patches)} patch pairs loaded from cache"
                     + (f"  (+ organ masks)" if self.load_mask else ""))
            # 'output_dir' is the key config.py actually defines; 'out_dir' is
            # accepted for callers (smoke tests) that pass it. Without the
            # fallback this silently wrote every scenario's grid into the cwd,
            # where each run overwrote the last.
            self._save_patch_grid(cfg.get('out_dir') or cfg.get('output_dir'),
                                  split_name)
            return

        # ── Step 1: index valid coordinates ──────────────────────────────────
        coords: List[Tuple] = []          # uniform-grid candidates
        focus_coords: List[Tuple] = []    # organ/vessel-centred candidates
        n_skip = 0
        log.info(f"[{split_name}] Indexing {len(pairs)} pairs "
                 f"(patch={self.ph}×{self.pw}×{self.patch_depth}, "
                 f"stride={self.stride_h}×{self.stride_w}"
                 + (f", organ_focus={self.organ_focus_frac:.2f}"
                    if self.organ_focus_frac > 0 else "") + ") …")

        n_missing_seg = 0
        # source path → case id, so the preload loop can tag each patch with the
        # case it came from without widening the coord tuples (which are built in
        # two places: here and _organ_centred_coords).
        path_to_case: Dict[str, str] = {}
        # Phase keys on the TARGET path, not the source: under multi-phase the
        # same NCCT is the source for both the arterial and the venous pair, so
        # src_path does not identify the phase.
        path_to_phase: Dict[str, int] = {}
        path_to_level: Dict[str, List[float]] = {}
        for pair in pairs:
            src_path = pair['source_path']
            tgt_path = pair['target_path']
            seg_path = pair.get('seg_path')
            case_id  = pair['case_id']
            path_to_case[src_path] = case_id
            path_to_phase[tgt_path] = int(pair.get('phase_id', 0))
            if self.cond_organs:
                path_to_level[tgt_path] = standardised_level(
                    self._levels, case_id, pair.get('phase', 'venous'),
                    self.cond_organs, self._lv_idx, self._lv_mean, self._lv_std)
            if self.load_mask and seg_path is None:
                n_missing_seg += 1
            try:
                src_vol = _load_vol(src_path)
                tgt_vol = _load_vol(tgt_path)
            except Exception as e:
                log.warning(f"  Skip {case_id}: {e}")
                continue

            if src_vol.shape != tgt_vol.shape:
                log.warning(f"  Shape mismatch {case_id}: "
                            f"{src_vol.shape} vs {tgt_vol.shape}, skipping")
                continue

            D, H, W = src_vol.shape

            # z range — respect patch_depth
            z_start = self.half_d
            z_end   = D - self.half_d
            if z_start >= z_end:
                log.warning(f"  Volume too thin ({D} slices) for depth={self.patch_depth}: {case_id}")
                continue

            ys = range(self.half_h, H - self.half_h + 1, self.stride_h)
            xs = range(self.half_w, W - self.half_w + 1, self.stride_w)

            if not ys or not xs:
                log.warning(f"  Volume too small ({H}×{W}) for patch {self.ph}×{self.pw}: {case_id}")
                continue

            for z in range(z_start, z_end):
                for y in ys:
                    for x in xs:
                        src_patch = src_vol[z,
                                            y - self.half_h: y + self.half_h,
                                            x - self.half_w: x + self.half_w]
                        if src_patch.shape != (self.ph, self.pw):
                            n_skip += 1; continue
                        # Validity in HU space
                        if src_patch.std()  < self.min_std:
                            n_skip += 1; continue
                        if src_patch.mean() < self.min_mean:
                            n_skip += 1; continue
                        if src_patch.max()  < self.min_max:
                            n_skip += 1; continue
                        coords.append((src_path, tgt_path, seg_path, z, y, x))

            # Organ/vessel-centred candidates for this pair (only when enabled).
            if self.organ_focus_frac > 0.0 and seg_path is not None:
                focus_coords.extend(
                    self._organ_centred_coords(src_vol, seg_path, src_path, tgt_path,
                                               z_start, z_end, H, W))

        log.info(f"  {len(coords)} valid grid coords"
                 + (f", {len(focus_coords)} organ-centred coords" if self.organ_focus_frac > 0 else "")
                 + f", {n_skip} rejected")
        if self.load_mask and n_missing_seg:
            log.warning(f"  [{split_name}] {n_missing_seg}/{len(pairs)} pairs have no "
                        f"segmentation mask on disk — those patches will use a zero mask")

        if not coords and not focus_coords:
            self._debug(pairs)
            self.src_patches: List[np.ndarray] = []
            self.tgt_patches: List[np.ndarray] = []
            self.mask_patches = [] if self.load_mask else None
            self.case_ids = []
            self.patch_z = []
            self.phase_ids = []
            self.level_vecs = []
            return

        # ── Step 2: sub-sample (mixing organ-focused + grid candidates) ───────
        if self.organ_focus_frac > 0.0:
            coords = self._mix_focus_and_grid(coords, focus_coords, max_patches, rng)
        elif max_patches and len(coords) > max_patches:
            chosen = rng.choice(len(coords), max_patches, replace=False)
            coords = [coords[i] for i in chosen]
            log.info(f"  Sub-sampled to {len(coords)} patches for RAM preload")

        # Sub-sampling above shuffles coords across cases; sort by (src, tgt)
        # path so the preload loop below touches each volume in one
        # contiguous run instead of thrashing `_load_vol`'s LRU cache (this
        # was the actual cause of multi-hour preloads on large splits — see
        # IMPLEMENTATION.md).
        coords.sort(key=lambda c: (c[0], c[1]))

        # ── Step 3: preload both source and target patches into RAM ───────────
        log.info(f"  Preloading {len(coords)} patch pairs into RAM …")
        self.src_patches  = []
        self.tgt_patches  = []
        self.mask_patches = [] if self.load_mask else None
        self.case_ids     = []
        self.patch_z      = []
        self.phase_ids    = []
        self.level_vecs   = []

        def _crop(vol, z, y, x):
            if self.patch_depth == 1:
                return vol[z,
                           y - self.half_h: y + self.half_h,
                           x - self.half_w: x + self.half_w].copy()
            return vol[z - self.half_d: z + self.half_d,
                       y - self.half_h:  y + self.half_h,
                       x - self.half_w:  x + self.half_w].copy()

        def _crop_src(vol, z, y, x):
            """Source crop. 2.5-D returns (2k+1, H, W) with EDGE-CLAMPED z.

            Clamping rather than shrinking the z range: inference has to produce
            slice 0 and slice D-1 regardless, so the window must be defined
            there. Using the same rule in training avoids a train/test mismatch
            at the volume ends, and costs no slices.
            """
            if self.k_slices == 0:
                return _crop(vol, z, y, x)
            zs = np.clip(np.arange(z - self.k_slices, z + self.k_slices + 1),
                         0, vol.shape[0] - 1)
            return vol[zs,
                       y - self.half_h: y + self.half_h,
                       x - self.half_w: x + self.half_w].copy()

        for src_path, tgt_path, seg_path, z, y, x in tqdm(coords, desc=f'Preload [{split_name}]', leave=False):
            src_vol = _load_vol(src_path)
            tgt_vol = _load_vol(tgt_path)

            sp = _crop_src(src_vol, z, y, x)
            tp = _crop(tgt_vol, z, y, x)      # target is always the centre slice

            sp = np.clip(sp, self.hu_min, self.hu_max)
            sp = ((sp - self.hu_min) / (self.hu_max - self.hu_min)).astype(np.float32)
            tp = np.clip(tp, self.hu_min, self.hu_max)
            tp = ((tp - self.hu_min) / (self.hu_max - self.hu_min)).astype(np.float32)

            self.src_patches.append(sp)
            self.tgt_patches.append(tp)
            self.case_ids.append(path_to_case.get(src_path, 'unknown'))
            self.patch_z.append(int(z))
            self.phase_ids.append(path_to_phase.get(tgt_path, 0))
            if self.cond_organs:
                self.level_vecs.append(path_to_level.get(
                    tgt_path, [0.0] * len(self.cond_organs)))

            if self.load_mask:
                mp = np.zeros_like(sp)
                if seg_path is not None:
                    try:
                        seg_vol = _load_vol(seg_path)
                        if seg_vol.shape == src_vol.shape:
                            crop = _crop(seg_vol, z, y, x)
                            # Multi-label: keep raw ids (for per-organ metrics /
                            # label-restricted focus). Else binarise as before.
                            mp = (crop if self.mask_multilabel
                                  else (crop > 0)).astype(np.float32)
                    except Exception as e:
                        log.warning(f"  Failed to load mask {seg_path}: {e}")
                self.mask_patches.append(mp)

        self._save_patch_grid(cfg.get('out_dir') or cfg.get('output_dir'),
                              split_name)
        log.info(f"  [{split_name}] {len(self.src_patches)} patch pairs in RAM"
                 + (f"  (+ organ masks)" if self.load_mask else ""))

        if self._cache_file is not None and self.src_patches:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {'src': np.stack(self.src_patches), 'tgt': np.stack(self.tgt_patches),
                           'case_id': np.array(self.case_ids),   # plain unicode array, no pickle
                           'patch_z': np.array(self.patch_z, dtype=np.int32),
                           'phase_id': np.array(self.phase_ids, dtype=np.int64)}
            if self.cond_organs:
                save_kwargs['level'] = np.asarray(self.level_vecs, dtype=np.float32)
            if self.load_mask:
                save_kwargs['mask'] = np.stack(self.mask_patches)
            np.savez(self._cache_file, **save_kwargs)
            log.info(f"  [{split_name}] Cached patches to {self._cache_file}")

    # -----------------------------------------------------------------------
    def _organ_centred_coords(self, src_vol, seg_path, src_path, tgt_path,
                              z_start, z_end, H, W) -> List[Tuple]:
        """Patch centres placed ON organ/vessel voxels (validity-filtered).

        Draws up to `max_focus_candidates_per_vol` random voxels from the target
        organ set, clamps each to a legal patch centre, and keeps it if the NCCT
        patch there passes the same HU validity filter as the grid path.
        """
        try:
            seg_vol = _load_vol(seg_path)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  organ-focus: failed to load {seg_path}: {e}")
            return []
        if seg_vol.shape != src_vol.shape:
            return []

        if self.organ_focus_labels:
            organ = np.isin(seg_vol, np.asarray(self.organ_focus_labels))
        else:
            organ = seg_vol > 0
        vox = np.argwhere(organ)                      # (K,3) in (z,y,x)
        if len(vox) == 0:
            return []
        if len(vox) > self.max_focus_cand:
            sel = self._rng.choice(len(vox), self.max_focus_cand, replace=False)
            vox = vox[sel]

        out: List[Tuple] = []
        seen = set()
        for z, y, x in vox:
            z = int(np.clip(z, z_start, z_end - 1))
            y = int(np.clip(y, self.half_h, H - self.half_h))
            x = int(np.clip(x, self.half_w, W - self.half_w))
            key = (z, y, x)
            if key in seen:
                continue
            seen.add(key)
            sp = src_vol[z, y - self.half_h: y + self.half_h,
                         x - self.half_w: x + self.half_w]
            if sp.shape != (self.ph, self.pw):
                continue
            if sp.std() < self.min_std or sp.mean() < self.min_mean or sp.max() < self.min_max:
                continue
            out.append((src_path, tgt_path, seg_path, z, y, x))
        return out

    # -----------------------------------------------------------------------
    def _mix_focus_and_grid(self, grid, focus, max_patches, rng) -> List[Tuple]:
        """Blend organ-centred and grid candidates so ~organ_focus_frac of the
        final patches are organ-centred (backfilling from the other pool if one
        runs short so we still hit max_patches)."""
        budget = max_patches if max_patches else (len(grid) + len(focus))
        n_focus = min(len(focus), int(round(budget * self.organ_focus_frac)))
        n_grid  = budget - n_focus
        if n_grid > len(grid):                      # not enough grid → more focus
            n_focus = min(len(focus), n_focus + (n_grid - len(grid)))
            n_grid  = min(len(grid), budget - n_focus)

        def _take(pool, k):
            if k <= 0 or not pool:
                return []
            if k >= len(pool):
                return list(pool)
            return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

        chosen = _take(focus, n_focus) + _take(grid, n_grid)
        log.info(f"  Organ-focus mix: {n_focus} organ-centred + {n_grid} grid "
                 f"= {len(chosen)} patches (frac={self.organ_focus_frac:.2f})")
        return chosen

    # -----------------------------------------------------------------------
    def _cache_path(self, pairs: List[Dict], cfg: Dict, max_patches: Optional[int]) -> Optional[Path]:
        """
        Deterministic on-disk cache location for this exact preloaded patch
        set. Independent of `output_dir` (fixed via `cfg['cache_dir']`) so
        the cache is shared across scenario runs that only differ in loss
        flags, not in data/geometry config.
        """
        cache_dir = cfg.get('cache_dir')
        if not cache_dir:
            return None
        key_data = {
            'pairs':        sorted((p['source_path'], p['target_path']) for p in pairs),
            'patch_h':      self.ph,
            'patch_w':      self.pw,
            'patch_depth':  self.patch_depth,
            'stride_h':     self.stride_h,
            'stride_w':     self.stride_w,
            'min_std':      self.min_std,
            'min_mean':     self.min_mean,
            'min_max':      self.min_max,
            'hu_min':       self.hu_min,
            'hu_max':       self.hu_max,
            'max_patches':  max_patches,
            'seed':         cfg.get('data_seed', cfg.get('seed', 42)),
            'load_mask':    self.load_mask,
            # Binary vs raw-label mask changes cached content. This also covers
            # turning per-organ loss weights on/off (they flip mask_multilabel).
            # The weight *values* are deliberately NOT in the key: they are applied
            # in the loss, not baked into the cache, so re-tuning weights must not
            # force a re-preload.
            'mask_multilabel': self.mask_multilabel,
            'seg_suffix':   cfg.get('seg_suffix', '_seg_reg'),  # switching mask set changes cached masks
            'split_name':   self.split_name,
            # The pair list already differs between single- and multi-phase runs
            # (arterial targets are different files), so the digest would change
            # anyway — this is explicit rather than incidental.
            'target_phases': list(cfg.get('target_phases')
                                  or [cfg.get('target_phase', 'venous')]),
            # Changes the SHAPE of every cached source patch.
            'n_input_slices': int(cfg.get('n_input_slices', 1)),
            # Changes the CONTENT of the cached level vectors.
            'cond_organs': list(cfg.get('cond_organs') or []),
        }
        # Only perturb the cache key when organ-focus is on, so existing
        # uniform-grid caches remain valid for legacy (frac==0) runs.
        if self.organ_focus_frac > 0.0:
            key_data['organ_focus_frac']   = self.organ_focus_frac
            key_data['organ_focus_labels'] = (sorted(self.organ_focus_labels)
                                              if self.organ_focus_labels else None)
            key_data['max_focus_cand']     = self.max_focus_cand
        digest = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
        return Path(cache_dir) / f'{self.split_name}_{digest}.npz'

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.src_patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 2.5-D source patches are already (2k+1, H, W) — the slice stack IS the
        # channel axis, so no unsqueeze. Everything else gets a channel axis added.
        sp = self.src_patches[idx]
        src = torch.from_numpy(sp)
        if getattr(self, 'n_input_slices', 1) == 1:
            src = src.unsqueeze(0)                                  # (1, H, W) or (1, D, H, W)
        tgt = torch.from_numpy(self.tgt_patches[idx]).unsqueeze(0)
        item = {'source': src, 'target': tgt}
        # Always emitted (0 for single-phase runs) so the batch schema does not
        # change with the flag; the trainer only forwards it when the generator
        # was built with use_phase_cond.
        item['phase'] = torch.tensor(
            self.phase_ids[idx] if self.phase_ids else 0, dtype=torch.long)
        if self.level_vecs:
            item['level'] = torch.as_tensor(self.level_vecs[idx], dtype=torch.float32)
        if self.mask_patches:
            item['mask'] = torch.from_numpy(self.mask_patches[idx]).unsqueeze(0)
        return item

    # -----------------------------------------------------------------------
    def _save_patch_grid(self, out_dir, split_name: str, n: int = 16):
        """Save a grid of (source | target) slice pairs for visual inspection.

        `out_dir=None` skips writing. Previously this defaulted to Path('.'),
        which dropped patch_grids/ into whatever the cwd happened to be — the
        repo root, for any caller that did not set out_dir/output_dir.
        """
        if not self.src_patches:
            return
        if out_dir is None:
            log.debug(f"  [{split_name}] no out_dir — skipping patch grid")
            return
        out = Path(out_dir) / 'patch_grids'
        out.mkdir(parents=True, exist_ok=True)

        n = min(n, len(self.src_patches))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.src_patches), n, replace=False)

        # For 3D patches take the center slice for display
        def _mid(p: np.ndarray) -> np.ndarray:
            return p[p.shape[0] // 2] if p.ndim == 3 else p

        cols = 8
        rows = max(1, -(-n // (cols // 2)))   # ceil div; avoids rows=0 when n < cols//2
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).flatten()
        for i in range(n):
            sp = _mid(self.src_patches[idx[i]])
            tp = _mid(self.tgt_patches[idx[i]])
            axes[2 * i].imshow(sp, cmap='gray', vmin=0, vmax=1)
            axes[2 * i].set_title('NCCT', fontsize=6); axes[2 * i].axis('off')
            axes[2 * i + 1].imshow(tp, cmap='gray', vmin=0, vmax=1)
            axes[2 * i + 1].set_title('CECT', fontsize=6); axes[2 * i + 1].axis('off')
        plt.suptitle(f'Sample patches [{split_name}] (NCCT | CECT)', fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f'{split_name}_patch_grid.png', dpi=100)
        plt.close()

    def _debug(self, pairs: List[Dict]):
        """Print diagnostic info when no patches were found."""
        if not pairs:
            return
        try:
            p = pairs[0]
            sv = _load_vol(p['source_path'])
            D, H, W = sv.shape
            mid = sv[D // 2]
            log.error(
                f"\n=== DEBUG: no valid patches found ===\n"
                f"  First pair: {p['case_id']}\n"
                f"  Source shape (D,H,W): {D}×{H}×{W}\n"
                f"  mid-slice HU: min={mid.min():.0f}  max={mid.max():.0f}  "
                f"mean={mid.mean():.0f}  std={mid.std():.0f}\n"
                f"  y positions: {list(range(self.half_h, H-self.half_h+1, self.stride_h))[:5]}\n"
                f"  x positions: {list(range(self.half_w, W-self.half_w+1, self.stride_w))[:5]}\n"
                f"  → Check: hu_min/hu_max, min_patch_std ({self.min_std}), "
                f"min_patch_mean ({self.min_mean}), patch_size ({self.ph}×{self.pw})"
            )
        except Exception as e:
            log.error(f"Debug failed: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def standardised_level(levels: Dict, case_id: str, phase: str,
                       organs: List[str], idx: List[int],
                       mean: List[float], std: List[float]) -> List[float]:
    """Standardised per-organ level vector for one (case, phase).

    Any organ that is absent, too small to measure, or missing from the dump
    resolves to 0.0 — which after standardisation IS the population mean. So a
    partially-missing case degrades to 'no information' rather than to a wrong
    number, and 'population' inference mode is just an all-zeros vector.
    """
    rec = (levels.get('cases', {}).get(case_id) or {}).get(phase)
    out = []
    for j, o in enumerate(organs):
        v = float('nan') if rec is None else rec.get('cect', {}).get(o, float('nan'))
        out.append(0.0 if not np.isfinite(v) else (v - mean[j]) / std[j])
    return out


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from config.
    Prints a sample patch grid per split to cfg['out_dir']/patch_grids/.
    """
    train_pairs, val_pairs, _ = find_pairs_and_split(cfg)

    # Capacity probe: shrink the TRAIN set only, leaving val/test as they are so
    # the run stays comparable to every other run's validation numbers. Cases are
    # taken in split order (already a seeded permutation), so the subset is
    # reproducible and nested — max_train_cases=5 is the first 5 of the 10.
    n_cap = cfg.get('max_train_cases')
    if n_cap and n_cap < len(train_pairs):
        log.warning(f"max_train_cases={n_cap}: training on {n_cap} of "
                    f"{len(train_pairs)} cases — CAPACITY PROBE, not a "
                    f"comparable run")
        train_pairs = train_pairs[:n_cap]

    train_ds = CTPairDataset(
        train_pairs,
        cfg,
        max_patches = cfg.get('max_train_patches', 20_000),
        split_name  = 'train',
    )
    val_ds = CTPairDataset(
        val_pairs,
        cfg,
        max_patches = cfg.get('max_val_patches', 4_000),
        split_name  = 'val',
    )

    num_workers = cfg.get('num_workers', 0)   # 0 = fastest when patches are in RAM
    # pin_memory only helps (and is only valid) when copying to a CUDA device;
    # on CPU it does nothing but emit a warning, so gate it on the actual device.
    pin = str(cfg.get('device', 'cpu')).startswith('cuda')

    g = torch.Generator()
    g.manual_seed(int(cfg.get('seed', 42)))

    def _worker_init(worker_id):
        s = int(cfg.get('seed', 42)) + worker_id
        np.random.seed(s % (2 ** 32))
        random.seed(s)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin,
        drop_last   = True,
        generator   = g,
        worker_init_fn = _worker_init if num_workers > 0 else None,
     )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
    )

    log.info(f"Train loader: {len(train_loader)} batches  |  "
             f"Val loader: {len(val_loader)} batches")
    return train_loader, val_loader


__all__ = ['CTPairDataset', 'build_loaders', 'find_pairs_and_split']
