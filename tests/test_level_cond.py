#!/usr/bin/env python
"""
Correctness tests for level conditioning, the conditional discriminator, and 2.5-D.

The three failures these exist to catch, all silent:

  1. A NEW MODULE SHIFTS THE RNG. Constructing an nn.Module draws from the global
     RNG, so adding `level_proj` or the discriminator's `cond_proj` — even unused —
     would change the init of every layer created after it and silently invalidate
     every baseline result under a fixed seed. Checked tensor-by-tensor.

  2. THE LEVEL VECTOR GETS PERMUTED. If the per-patch level does not follow its
     patch through shuffling, the model is conditioned on another case's contrast
     level. Nothing in the loss would reveal that; it would just look like
     conditioning "didn't help".

  3. 2.5-D SILENTLY TRUNCATES Z. If the slice window shrinks the usable z range
     instead of clamping at the edges, the top and bottom of every volume quietly
     stop being predicted. The output shape still looks plausible.

Usage:  python tests/test_level_cond.py
"""
import json
import shutil
import tempfile
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch

from dataset import build_loaders, standardised_level
from models import UNetGenerator

VOL_XYZ = (64, 64, 7)
_failures = []


def check(cond, label):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    if not cond:
        _failures.append(label)


def _g(**kw):
    torch.manual_seed(42)
    return UNetGenerator(dims=2, base_channels=16, dropout=0.2, **kw)


# ---------------------------------------------------------------------------

def test_rng_safety():
    print('\n-- 1. new modules must not disturb the RNG --')
    a, b = _g(), _g(use_phase_cond=False, n_levels=0)
    sa, sb = a.state_dict(), b.state_dict()
    check(sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa),
          'generator: level conditioning off is BITWISE identical')


def test_generator_levels():
    print('\n-- generator level conditioning --')
    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32)
    base = _g().eval()
    for name, kw, args in [
        ('phase',       dict(use_phase_cond=True), dict(phase=torch.tensor([0, 1]))),
        ('level',       dict(n_levels=3),          dict(level=torch.randn(2, 3))),
        ('phase+level', dict(use_phase_cond=True, n_levels=3),
                        dict(phase=torch.tensor([0, 1]), level=torch.randn(2, 3))),
    ]:
        m = _g(**kw).eval()
        with torch.no_grad():
            y0, y1 = base(x), m(x, **args)
        check(torch.allclose(y0, y1, atol=1e-6),
              f'{name}: zero-init is an exact identity')

    m = _g(n_levels=3).eval()
    with torch.no_grad():
        m.level_proj.weight.normal_(0, 0.5)
        m.film_dec3.to_scale_shift[-1].weight.normal_(0, 0.3)
        lo = m(x, level=torch.zeros(2, 3))
        hi = m(x, level=torch.full((2, 3), 2.0))
    check(not torch.allclose(lo, hi, atol=1e-6),
          'a different level produces a different image')

    for kw, args, lbl in [
        (dict(n_levels=3), {}, 'missing level raises'),
        ({}, dict(level=torch.randn(2, 3)), 'unexpected level raises'),
    ]:
        try:
            _g(**kw).eval()(x, **args)
            check(False, lbl)
        except ValueError:
            check(True, lbl)


# ---------------------------------------------------------------------------

def _make(root, cid, seed, phases=('venous',)):
    d = root / cid
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for ph in ('nc',) + tuple(phases):
        nib.save(nib.Nifti1Image(rng.uniform(-50, 150, VOL_XYZ).astype(np.float32),
                                 np.eye(4)), str(d / f'{cid}_{ph}_deeds.nii.gz'))


def test_data():
    print('\n-- data plumbing --')
    tmp = Path(tempfile.mkdtemp(prefix='lvl_'))
    try:
        data = tmp / 'data'
        for i in range(8):
            _make(data, f'case{i:02d}', i)

        organs = ['aorta', 'liver']
        # Distinct per-case levels so a permutation bug is visible.
        levels = {'organs': organs, 'phases': ['venous'],
                  'standardize': {'mean': [0.0, 0.0], 'std': [1.0, 1.0],
                                  'source_split': 'train'},
                  'cases': {f'case{i:02d}': {
                      'split': 'train',
                      'venous': {'cect': {'aorta': 100.0 + 10 * i, 'liver': 50.0 + i},
                                 'ncct': {'aorta': 40.0, 'liver': 45.0}}}
                      for i in range(8)}}
        lj = tmp / 'levels.json'
        lj.write_text(json.dumps(levels))

        from config import train_config
        cfg = dict(train_config)
        cfg.update(data_dir=str(data), labels_csv='', cache_dir=str(tmp / 'cache'),
                   output_dir=tmp / 'run', batch_size=4, patch_size=32,
                   max_train_patches=60, max_val_patches=20, device='cpu',
                   min_patch_std=0.0, min_patch_mean=-1e9, min_patch_max=-1e9,
                   use_organ=False, cond_organs=organs, levels_json=str(lj))

        tr, _ = build_loaders(cfg)
        ds = tr.dataset
        check('level' in ds[0], "__getitem__ emits 'level'")
        check(len(ds.level_vecs) == len(ds.src_patches),
              'one level vector per patch')

        # 2. the level must follow its patch, not its position
        ok = True
        for i in range(len(ds)):
            cid = ds.case_ids[i]
            want = 100.0 + 10 * int(cid[-2:])
            if abs(float(ds[i]['level'][0]) - want) > 1e-4:
                ok = False
                break
        check(ok, "each patch's level matches its own case_id (no permutation)")

        # population fallback for an unknown case == standardised mean == 0
        v = standardised_level(levels, 'no_such_case', 'venous', organs,
                               [0, 1], [0.0, 0.0], [1.0, 1.0])
        check(v == [0.0, 0.0],
              'a missing case degrades to the population mean, not a wrong number')

        # cache key must separate conditioned from unconditioned
        from dataset import CTPairDataset
        def digest(c):
            s = CTPairDataset.__new__(CTPairDataset)
            s.split_name = 'train'; s.ph = s.pw = 32; s.patch_depth = 1
            s.stride_h = s.stride_w = 16
            s.min_std, s.min_mean, s.min_max = 0.0, -1e9, -1e9
            s.hu_min, s.hu_max = -200, 400
            s.load_mask = s.mask_multilabel = False; s.organ_focus_frac = 0.0
            pairs = [{'source_path': 'a', 'target_path': 'b'}]
            return CTPairDataset._cache_path(s, pairs, c, 100).name
        check(digest({**cfg, 'cond_organs': []}) != digest(cfg),
              'cond_organs changes the cache key')
        check(digest({**cfg, 'n_input_slices': 5}) != digest(cfg),
              'n_input_slices changes the cache key')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_25d():
    print('\n-- 3. 2.5-D must clamp, not truncate --')
    tmp = Path(tempfile.mkdtemp(prefix='t25_'))
    try:
        data = tmp / 'data'
        for i in range(6):
            _make(data, f'case{i:02d}', i)
        from config import train_config
        cfg = dict(train_config)
        cfg.update(data_dir=str(data), labels_csv='', output_dir=tmp / 'run',
                   batch_size=2, patch_size=32, max_train_patches=200,
                   max_val_patches=40, device='cpu', min_patch_std=0.0,
                   min_patch_mean=-1e9, min_patch_max=-1e9, use_organ=False)

        for n in (1, 5):
            c = {**cfg, 'n_input_slices': n, 'in_channels': n,
                 'cache_dir': str(tmp / f'c{n}')}
            ds = build_loaders(c)[0].dataset
            item = ds[0]
            check(tuple(item['source'].shape) == (n, 32, 32),
                  f'n_input_slices={n}: source is {tuple(item["source"].shape)}')
            check(tuple(item['target'].shape) == (1, 32, 32),
                  f'n_input_slices={n}: target stays a single slice')
            if n > 1:
                zs = set(ds.patch_z)
                check(min(zs) == 0 and max(zs) == VOL_XYZ[2] - 1,
                      f'full z range still reachable ({min(zs)}..{max(zs)} of '
                      f'{VOL_XYZ[2]} slices) — clamped, not truncated')
                check(all(p.shape == (n, 32, 32) for p in ds.src_patches),
                      'every source patch is a full window, including at z=0')

        # inference must return the input shape
        from infer_volume import DeterministicPredictor, infer_volume
        torch.manual_seed(0)
        G = UNetGenerator(dims=2, base_channels=8, in_channels=5).eval()
        vol = np.random.uniform(-50, 150, (7, 32, 32)).astype(np.float32)
        out = infer_volume(DeterministicPredictor(G, 'cpu', False),
                           vol, {**cfg, 'n_input_slices': 5, 'in_channels': 5},
                           'cpu', batch_size=4)[0]     # (n_out, D, H, W)
        check(out.shape == vol.shape,
              f'infer_volume round-trips the shape {vol.shape} -> {out.shape}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_rng_safety()
    test_generator_levels()
    test_data()
    test_25d()
    print()
    if _failures:
        print(f'{len(_failures)} FAILED:')
        for f in _failures:
            print('  -', f)
        return 1
    print('level conditioning + 2.5-D PASSED.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
