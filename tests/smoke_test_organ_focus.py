"""
Synthetic smoke test for organ-focused patch sampling in CTPairDataset.

No real data: writes tiny NIfTI NCCT/CECT/seg triples where the seg marks a
small off-centre organ blob, then checks that with organ_focus_frac>0 a large
share of sampled patches actually overlap the organ (vs the legacy uniform grid,
which mostly misses it). Runs on the box or locally if nibabel is installed.
"""

import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dataset import CTPairDataset

D, H, W = 4, 256, 256
ORGAN_LABEL = 52                      # pretend "aorta"


def _write(dirpath):
    """One case: NCCT + CECT + multilabel seg with a 24x24 organ blob."""
    case = Path(dirpath) / 'caseA'
    case.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # (X,Y,Z) on disk; _load_vol transposes to (D,H,W)
    ncct = rng.normal(40, 30, size=(W, H, D)).astype(np.float32)
    cect = ncct + 60
    seg = np.zeros((W, H, D), np.int16)
    seg[40:64, 40:64, :] = ORGAN_LABEL           # organ blob near a corner
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(ncct, aff), case / 'S_1_deeds.nii.gz')
    nib.save(nib.Nifti1Image(cect, aff), case / 'S_2_deeds.nii.gz')
    nib.save(nib.Nifti1Image(seg, aff),  case / 'S_2_deeds_seg_reg.nii.gz')


def _overlap_frac(ds):
    """Share of preloaded patches whose mask patch contains organ voxels."""
    hits = sum(int(m.sum() > 0) for m in ds.mask_patches)
    return hits / max(1, len(ds.mask_patches))


def main():
    tmp = tempfile.mkdtemp()
    _write(tmp)
    pair = {'source_path': str(Path(tmp) / 'caseA' / 'S_1_deeds.nii.gz'),
            'target_path': str(Path(tmp) / 'caseA' / 'S_2_deeds.nii.gz'),
            'seg_path':    str(Path(tmp) / 'caseA' / 'S_2_deeds_seg_reg.nii.gz'),
            'case_id': 'caseA'}
    base = dict(patch_size=64, patch_depth=1, overlap=0.0, seed=0,
                hu_min=-200, hu_max=400, min_patch_std=1.0, min_patch_mean=-800.0,
                min_patch_max=-500.0, out_dir=tmp, use_organ=True)

    grid = CTPairDataset([pair], {**base, 'organ_focus_frac': 0.0},
                         max_patches=200, split_name='grid')
    foc = CTPairDataset([pair], {**base, 'organ_focus_frac': 0.8,
                                 'organ_focus_labels': [ORGAN_LABEL]},
                        max_patches=200, split_name='focus')

    g, f = _overlap_frac(grid), _overlap_frac(foc)
    print(f"uniform-grid organ-overlap: {g:.1%}  ({len(grid)} patches)")
    print(f"organ-focused organ-overlap: {f:.1%}  ({len(foc)} patches)")
    assert len(foc) > 0, "no focused patches produced"
    assert f > g, f"focus ({f:.1%}) should beat grid ({g:.1%})"
    assert f > 0.5, f"focused overlap unexpectedly low: {f:.1%}"
    print("\nOrgan-focus smoke check passed.")


if __name__ == '__main__':
    main()
