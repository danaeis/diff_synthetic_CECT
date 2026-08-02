"""
ROI-guided organ features for CT contrast-phase classification.

The published model (Hou et al. 2025, "Segment-and-Classify") classifies phase
from ONE feature per organ: the median HU inside that organ's TotalSegmentator
mask. This module reproduces that feature extraction two ways:

  1. From a TotalSegmentator `statistics` .pkl (what the original `train.py`
     consumed) — `features_from_stats_pkl`.
  2. Directly from a (CT volume, multi-label TS mask) pair — `features_from_mask`
     — so we can extract features for images that were never run through TS
     (e.g. a GENERATED CECT), by borrowing the co-registered real mask.

Both return a length-16 vector in the SAME organ order the trained XGBoost model
expects (`ORGANS`), with NaN where an organ is absent (matching the original's
`0.0 -> np.nan`), which XGBoost handles natively.

Organ → TS-label-id mapping is loaded from the *installed* TotalSegmentator at
runtime (`totalsegmentator.map_to_binary.class_map['total']`) — never hardcoded,
so it can't silently drift from your TS version. `inspect_mask_labels()` reports
which labels a real mask actually contains, to verify before trusting a run.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# The 16 organs, in the exact order the XGBoost model was trained on
# (see the original train.py / totalseg_get_phase.py).
ORGANS: List[str] = [
    "liver", "pancreas", "urinary_bladder", "gallbladder",
    "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
    "pulmonary_vein", "brain", "colon", "small_bowel",
]


def load_organ_label_map(explicit: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Return {organ_name: ts_label_id} for the 16 ORGANS.

    Priority: an explicitly-provided dict, else TotalSegmentator's own
    `class_map['total']` (inverted). Raises if TS isn't importable or an organ
    name is missing from the map (version mismatch) — no silent fallback.
    """
    if explicit is not None:
        missing = [o for o in ORGANS if o not in explicit]
        if missing:
            raise KeyError(f"explicit organ map missing organs: {missing}")
        return {o: int(explicit[o]) for o in ORGANS}

    try:
        from totalsegmentator.map_to_binary import class_map
    except Exception as e:
        raise RuntimeError(
            "Could not import TotalSegmentator to resolve organ label ids "
            f"({e}). Install totalsegmentator, or pass an explicit organ→id map "
            "verified with inspect_mask_labels()."
        )
    total = class_map["total"]                       # {id: name}
    name_to_id = {name: int(i) for i, name in total.items()}
    missing = [o for o in ORGANS if o not in name_to_id]
    if missing:
        raise KeyError(
            f"TS class_map['total'] is missing these organ names: {missing}. "
            f"Your TotalSegmentator version differs from the model's; verify with "
            f"inspect_mask_labels() and pass an explicit map."
        )
    return {o: name_to_id[o] for o in ORGANS}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def features_from_stats_pkl(stats_path: str) -> np.ndarray:
    """16-vector of per-organ median HU from a TS `statistics` pickle (the
    format the original pipeline dumped via ts_get_stats.py)."""
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    feats = []
    for organ in ORGANS:
        v = stats.get(organ, {}).get("intensity", 0.0)
        feats.append(np.nan if v == 0.0 else float(v))
    return np.array(feats, dtype=np.float64)


def features_from_mask(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    organ_label_map: Dict[str, int],
    aggregation: str = "median",
) -> np.ndarray:
    """
    16-vector of per-organ HU aggregate, computed directly from a CT volume and
    its co-registered multi-label TS mask.

    ct_volume / mask_volume must be the same shape (same grid). `aggregation`
    matches the trained model: 'median' (default; the model was trained with
    stats_aggregation='median'). Absent organs → NaN.
    """
    if ct_volume.shape != mask_volume.shape:
        raise ValueError(
            f"ct/mask shape mismatch: {ct_volume.shape} vs {mask_volume.shape}. "
            f"The mask must be on the same grid as the CT (co-registered)."
        )
    agg = np.median if aggregation == "median" else np.mean
    mask_volume = mask_volume.astype(np.int32, copy=False)
    feats = []
    for organ in ORGANS:
        lid = organ_label_map[organ]
        vals = ct_volume[mask_volume == lid]
        feats.append(float(agg(vals)) if vals.size > 0 else np.nan)
    return np.array(feats, dtype=np.float64)


def features_from_paths(
    ct_path: str,
    mask_path: str,
    organ_label_map: Dict[str, int],
    aggregation: str = "median",
) -> np.ndarray:
    """Convenience: load CT + mask NIfTIs and extract mask-based features.

    Uses nibabel with no transpose (CT and mask share the same file geometry),
    so voxel correspondence is exact."""
    import nibabel as nib
    ct = nib.load(ct_path).get_fdata().astype(np.float32)
    mk = nib.load(mask_path).get_fdata()
    return features_from_mask(ct, mk, organ_label_map, aggregation)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def inspect_mask_labels(mask_path: str, organ_label_map: Optional[Dict[str, int]] = None):
    """Print the label ids present in a mask and (if resolvable) their organ
    names — run this once on a real mask to confirm the id map before trusting
    a full extraction/eval run."""
    import nibabel as nib
    mk = nib.load(mask_path).get_fdata().astype(np.int32)
    ids = sorted(int(i) for i in np.unique(mk) if i != 0)
    print(f"mask {Path(mask_path).name}: {len(ids)} non-zero labels present")
    id_to_organ = {}
    if organ_label_map:
        id_to_organ = {v: k for k, v in organ_label_map.items()}
    for i in ids:
        tag = f"  <- ORGAN '{id_to_organ[i]}'" if i in id_to_organ else ""
        print(f"  label {i}: {int((mk == i).sum())} voxels{tag}")
    if organ_label_map:
        absent = [o for o in ORGANS if organ_label_map[o] not in ids]
        if absent:
            print(f"  ORGANS with no voxels in this mask (→ NaN feature): {absent}")


__all__ = [
    "ORGANS", "load_organ_label_map",
    "features_from_stats_pkl", "features_from_mask", "features_from_paths",
    "inspect_mask_labels",
]
