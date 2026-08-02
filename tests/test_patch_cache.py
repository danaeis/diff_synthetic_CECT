"""
Correctness test for CTPairDataset's on-disk patch cache (dataset.py).

Unlike smoke_test.py (which bypasses dataset.py entirely with synthetic
tensors fed straight to Trainer), this exercises the actual index -> preload
-> cache write -> cache read path against small synthetic NIfTI volumes, and
checks:
  1. First construction (no cache yet) preloads from scratch and writes a
     cache file.
  2. A second construction with an IDENTICAL config hits that cache file and
     produces bit-identical src/tgt/mask patch arrays to the first run (not
     just "doesn't crash" — actual data fidelity).
  3. A third construction with one cache-key-relevant field changed
     (patch_size) gets a DIFFERENT cache path — no stale reuse across
     incompatible configs.

Uses tiny random NIfTI volumes written to a temp dir (no real CT data
needed), so this runs anywhere nibabel + numpy + torch are installed.

Usage:
    python tests/test_patch_cache.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dataset import CTPairDataset

N_CASES = 3
VOL_SHAPE_XYZ = (48, 48, 6)  # -> _load_vol transposes to (D,H,W) = (6,48,48)


def _make_case(root: Path, case_id: str, seed: int, with_mask: bool):
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Values well inside the default validity thresholds regardless of the
    # permissive overrides below; shape doesn't matter for that here.
    src = rng.uniform(-50, 150, size=VOL_SHAPE_XYZ).astype(np.float32)
    tgt = rng.uniform(-50, 150, size=VOL_SHAPE_XYZ).astype(np.float32)

    src_path = case_dir / f'{case_id}_nc_deeds.nii.gz'
    tgt_path = case_dir / f'{case_id}_venous_deeds.nii.gz'
    nib.save(nib.Nifti1Image(src, affine=np.eye(4)), str(src_path))
    nib.save(nib.Nifti1Image(tgt, affine=np.eye(4)), str(tgt_path))

    seg_path = None
    if with_mask:
        seg = (rng.uniform(0, 1, size=VOL_SHAPE_XYZ) > 0.7).astype(np.float32)
        seg_path = case_dir / f'{case_id}_venous_deeds_seg_reg.nii.gz'
        nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), str(seg_path))

    return {
        'source_path': str(src_path),
        'target_path': str(tgt_path),
        'seg_path':    str(seg_path) if seg_path else None,
        'case_id':     case_id,
    }


def _base_cfg(cache_dir: Path, out_dir: Path) -> dict:
    return {
        'patch_size':     16,
        'patch_depth':    1,
        'overlap':        0.5,
        'min_patch_std':  0.0,       # permissive: every patch should pass
        'min_patch_mean': -1e9,
        'min_patch_max':  -1e9,
        'hu_min':         -200,
        'hu_max':          400,
        'seed':            42,
        'use_organ':       True,     # forces load_mask=True
        'cache_dir':       str(cache_dir),
        'out_dir':         str(out_dir),
    }


def _stack_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def main():
    tmp = Path(tempfile.mkdtemp(prefix='patch_cache_test_'))
    data_root = tmp / 'data'
    cache_dir = tmp / 'cache'
    out_dir   = tmp / 'out'
    results = []

    try:
        pairs = [_make_case(data_root, f'case{i:03d}', seed=100 + i, with_mask=True)
                 for i in range(N_CASES)]

        cfg = _base_cfg(cache_dir, out_dir)

        # ── 1. First construction: cache miss, preload from scratch ────────
        ds1 = CTPairDataset(pairs, cfg, max_patches=None, split_name='cache_test')
        ok = len(ds1.src_patches) > 0
        results.append(('preload produced patches', ok,
                         f'{len(ds1.src_patches)} patches' if ok else 'no patches produced'))

        cache_file_exists = ds1._cache_file is not None and ds1._cache_file.exists()
        results.append(('cache file written after first preload', cache_file_exists,
                         str(ds1._cache_file)))

        has_mask = ds1.mask_patches is not None and len(ds1.mask_patches) == len(ds1.src_patches)
        results.append(('mask patches preloaded (use_organ=True)', has_mask, ''))

        # ── 2. Second construction, identical config: should hit the cache ──
        ds2 = CTPairDataset(pairs, cfg, max_patches=None, split_name='cache_test')
        same_file = ds2._cache_file == ds1._cache_file
        results.append(('second run resolves to same cache file', same_file,
                         f'{ds2._cache_file} vs {ds1._cache_file}'))

        src_equal  = _stack_equal(ds1.src_patches, ds2.src_patches)
        tgt_equal  = _stack_equal(ds1.tgt_patches, ds2.tgt_patches)
        mask_equal = _stack_equal(ds1.mask_patches, ds2.mask_patches)
        results.append(('cached src patches bit-identical to original preload', src_equal, ''))
        results.append(('cached tgt patches bit-identical to original preload', tgt_equal, ''))
        results.append(('cached mask patches bit-identical to original preload', mask_equal, ''))

        count_equal = len(ds1.src_patches) == len(ds2.src_patches)
        results.append(('cached patch count matches original', count_equal,
                         f'{len(ds1.src_patches)} vs {len(ds2.src_patches)}'))

        # ── 3. Different geometry -> different cache file (no stale reuse) ──
        cfg_diff = dict(cfg)
        cfg_diff['patch_size'] = 8
        ds3 = CTPairDataset(pairs, cfg_diff, max_patches=None, split_name='cache_test')
        different_file = ds3._cache_file != ds1._cache_file
        results.append(('changed patch_size gets a different cache file', different_file,
                         f'{ds3._cache_file} vs {ds1._cache_file}'))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f'  -- {detail}' if detail and not ok else ''))
        if not ok:
            n_fail += 1

    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
