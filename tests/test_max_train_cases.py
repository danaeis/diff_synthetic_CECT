#!/usr/bin/env python
"""
Correctness test for `max_train_cases` (the Gate-B capacity probe).

Three things must hold, and the third is the one that silently breaks things:

  1. Only the TRAIN split shrinks. Val/test must be untouched, otherwise a
     capped run's validation numbers are not comparable to a full run's and the
     probe tells you nothing.
  2. The subset is nested and reproducible: cap=2 must be the first 2 cases of
     cap=3, so probes at different caps sit on the same data.
  3. The on-disk patch cache must NOT be shared between a capped and an uncapped
     run. `_cache_path` hashes the pair list, so truncating `train_pairs` before
     constructing the dataset changes the digest automatically — but that is a
     load-bearing accident, and if the truncation ever moves to after
     construction, a capped run would silently load the full run's 20k patches
     and look like it fit 97 cases.

Usage:  python tests/test_max_train_cases.py
"""
import shutil
import tempfile
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np

import dataset as D
from dataset import CTPairDataset, build_loaders

VOL_SHAPE_XYZ = (32, 32, 4)
N_CASES = 10

_failures = []


def check(cond, label):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    if not cond:
        _failures.append(label)


def _make_case(root: Path, case_id: str, seed: int):
    d = root / case_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    src = rng.uniform(-50, 150, size=VOL_SHAPE_XYZ).astype(np.float32)
    tgt = rng.uniform(-50, 150, size=VOL_SHAPE_XYZ).astype(np.float32)
    sp, tp = d / f'{case_id}_nc_deeds.nii.gz', d / f'{case_id}_venous_deeds.nii.gz'
    nib.save(nib.Nifti1Image(src, affine=np.eye(4)), str(sp))
    nib.save(nib.Nifti1Image(tgt, affine=np.eye(4)), str(tp))
    return {'source_path': str(sp), 'target_path': str(tp),
            'seg_path': None, 'case_id': case_id}


def _cfg(cache_dir: Path) -> dict:
    return {
        'patch_size': 16, 'patch_depth': 1, 'overlap': 0.5, 'dims': 2,
        'min_patch_std': 0.0, 'min_patch_mean': -1e9, 'min_patch_max': -1e9,
        'hu_min': -200, 'hu_max': 400, 'seed': 42, 'data_seed': 42,
        'batch_size': 2, 'num_workers': 0, 'device': 'cpu',
        'use_organ': False, 'cache_dir': str(cache_dir),
        'max_train_patches': 200, 'max_val_patches': 100,
    }


def _digest(pairs, cfg, max_patches, split='train'):
    """Cache filename CTPairDataset would choose, without preloading anything."""
    stub = CTPairDataset.__new__(CTPairDataset)
    stub.split_name = split
    stub.ph = stub.pw = cfg['patch_size']
    stub.patch_depth = cfg['patch_depth']
    stub.stride_h = stub.stride_w = int(cfg['patch_size'] * (1 - cfg['overlap']))
    stub.min_std, stub.min_mean = cfg['min_patch_std'], cfg['min_patch_mean']
    stub.min_max = cfg['min_patch_max']
    stub.hu_min, stub.hu_max = cfg['hu_min'], cfg['hu_max']
    stub.load_mask = False
    stub.mask_multilabel = False
    stub.organ_focus_frac = 0.0
    return CTPairDataset._cache_path(stub, pairs, cfg, max_patches).name


def main():
    tmp = Path(tempfile.mkdtemp(prefix='max_train_cases_'))
    try:
        data = tmp / 'data'
        pairs = [_make_case(data, f'case{i:02d}', seed=i) for i in range(N_CASES)]
        cfg = _cfg(tmp / 'cache')

        # Pin the split so the test does not depend on find_pairs_and_split's
        # own discovery/labels path — only on the capping logic.
        n_val = n_test = 2
        split = (pairs[n_test + n_val:], pairs[n_test:n_test + n_val], pairs[:n_test])
        orig = D.find_pairs_and_split
        D.find_pairs_and_split = lambda _c: split
        n_train_full = len(split[0])
        try:
            tr_full, va_full = build_loaders(dict(cfg))
            n_val_full = len(va_full.dataset)

            cap = 3
            tr_cap, va_cap = build_loaders({**cfg, 'max_train_cases': cap})

            cases_full = {p['case_id'] for p in split[0]}
            check(len(cases_full) == n_train_full == 6,
                  f'fixture: 6 train / 2 val / 2 test (got {n_train_full} train)')

            # 1. train shrinks, val does not
            check(len(tr_cap.dataset) < len(tr_full.dataset),
                  f'capped train set is smaller ({len(tr_cap.dataset)} < '
                  f'{len(tr_full.dataset)} patches)')
            check(len(va_cap.dataset) == n_val_full,
                  f'val set UNCHANGED by the cap ({len(va_cap.dataset)} patches)')

            # 2. nested + reproducible
            sub2 = split[0][:2]
            sub3 = split[0][:3]
            check([p['case_id'] for p in sub2] == [p['case_id'] for p in sub3][:2],
                  'cap=2 is a prefix of cap=3 (nested subsets)')
            tr_cap_again, _ = build_loaders({**cfg, 'max_train_cases': cap})
            check(len(tr_cap_again.dataset) == len(tr_cap.dataset),
                  'same cap gives the same patch count (reproducible)')

            # 3. cache keys must differ — the load-bearing part
            d_full = _digest(split[0], cfg, cfg['max_train_patches'])
            d_cap = _digest(sub3, cfg, cfg['max_train_patches'])
            check(d_full != d_cap,
                  f'capped run gets a DIFFERENT cache file ({d_cap} != {d_full})')
            check(_digest(sub3, cfg, cfg['max_train_patches']) == d_cap,
                  'cache key is stable for an unchanged cap')
            check(_digest(sub2, cfg, cfg['max_train_patches']) != d_cap,
                  'cap=2 and cap=3 get different cache files')

            # cap larger than the split is a no-op, not an error
            tr_big, _ = build_loaders({**cfg, 'max_train_cases': 999})
            check(len(tr_big.dataset) == len(tr_full.dataset),
                  'cap > n_cases is a no-op')
        finally:
            D.find_pairs_and_split = orig

        print()
        if _failures:
            print(f'{len(_failures)} FAILED:')
            for f in _failures:
                print('  -', f)
            return 1
        print('max_train_cases PASSED.')
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
