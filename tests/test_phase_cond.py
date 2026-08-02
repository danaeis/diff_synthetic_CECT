#!/usr/bin/env python
"""
Correctness tests for FiLM phase conditioning and multi-phase data loading.

Four things must hold. The first and third are the ones that fail silently.

  1. WITH CONDITIONING OFF, NOTHING CHANGES — bitwise. Not "produces similar
     results": the state dict must be tensor-for-tensor identical to a model
     built before this feature existed. Constructing an nn.Module draws from the
     global RNG, so instantiating the FiLM layers unconditionally (even unused)
     would shift the init of every layer created afterwards and silently change
     every baseline result under a fixed seed.

  2. ZERO-INIT IS AN EXACT IDENTITY. gamma=beta=0 at step 0, so the conditioned
     model starts equal to the unconditioned one. That is what lets a
     conditioned run be compared against, or fine-tuned from, a baseline.

  3. NO PATIENT LEAKAGE. Both phases of a case must land in the same split.
     Splitting the expanded pair list instead of the case list would put a
     patient's arterial scan in train and their venous in test — a leak that no
     metric would reveal.

  4. THE SINGLE-PHASE SPLIT IS UNCHANGED. A target_phases=['venous'] run must
     reproduce the existing split case-for-case, or multi-phase runs are not
     comparable to anything already in analysis/.

Plus the forward-compatibility contracts for the 2.5-D / 3-D work: a hardcoded
.view(B, C, 1, 1) in FiLM works in 2-D and breaks the first time dims=3 runs, so
it is asserted here rather than left to a comment.

Usage:  python tests/test_phase_cond.py
"""
import shutil
import tempfile
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch

import dataset as D
from dataset import find_pairs_and_split
from models import UNetGenerator

VOL_SHAPE_XYZ = (32, 32, 4)
_failures = []


def check(cond, label):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    if not cond:
        _failures.append(label)


def _build(**kw):
    torch.manual_seed(42)
    return UNetGenerator(dims=2, base_channels=16, dropout=0.2, **kw)


# ---------------------------------------------------------------------------
# 1-2. model
# ---------------------------------------------------------------------------

def test_model():
    print('\n-- generator --')
    base = _build()
    off = _build(use_phase_cond=False)
    sd_a, sd_b = base.state_dict(), off.state_dict()
    identical = (sd_a.keys() == sd_b.keys()
                 and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a))
    check(identical, 'use_phase_cond=False is BITWISE identical to the default '
                     '(no stray RNG draws)')

    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32)
    on = _build(use_phase_cond=True)
    base.eval(); on.eval()
    with torch.no_grad():
        y_off = base(x)
        y_on = on(x, torch.tensor([0, 1]))
    check(torch.allclose(y_off, y_on, atol=1e-6),
          f'zero-init FiLM is an exact identity (max diff '
          f'{(y_off - y_on).abs().max():.1e})')

    # ...and is not merely dead code: perturbing the last-layer WEIGHT (not the
    # bias, which is phase-independent) must make phases diverge.
    with torch.no_grad():
        on.film_dec3.to_scale_shift[-1].weight.normal_(0, 0.3)
        p0 = on(x, torch.tensor([0, 0]))
        p1 = on(x, torch.tensor([1, 1]))
    check(not torch.allclose(p0, p1, atol=1e-6),
          'once gamma depends on the embedding, phase 0 != phase 1')

    check(all(v == 0.0 for v in _build(use_phase_cond=True).film_stats().values()),
          'film_stats() reports gamma=0 at init')
    check(_build(use_phase_cond=False).film_stats() == {},
          'film_stats() is empty when conditioning is off')

    # guards: a mismatch between model and call site must raise, never be ignored
    for kw, arg, label in [
        (dict(use_phase_cond=True), None, 'missing phase raises'),
        (dict(), torch.tensor([0, 1]), 'unexpected phase raises'),
    ]:
        try:
            _build(**kw)(x, arg) if arg is not None else _build(**kw)(x)
            check(False, label)
        except ValueError:
            check(True, label)


def test_forward_compat():
    print('\n-- forward compatibility (2.5-D / 3-D) --')
    torch.manual_seed(0)
    g3 = UNetGenerator(dims=3, base_channels=8, use_phase_cond=True).eval()
    v = torch.randn(2, 1, 8, 32, 32)
    try:
        with torch.no_grad():
            o = g3(v, torch.tensor([0, 1]))
        check(o.shape == v.shape,
              f'dims=3 forward works {tuple(v.shape)} -> {tuple(o.shape)} '
              f'(catches a hardcoded 2-D FiLM broadcast)')
    except RuntimeError as e:
        check(False, f'dims=3 forward: {e}')

    torch.manual_seed(0)
    g25 = UNetGenerator(dims=2, base_channels=8, in_channels=5,
                        use_phase_cond=True).eval()
    s = torch.randn(2, 5, 32, 32)
    with torch.no_grad():
        o = g25(s, torch.tensor([0, 1]))
    check(o.shape == (2, 1, 32, 32),
          f'in_channels=5 -> 1-channel output {tuple(s.shape)} -> {tuple(o.shape)} '
          f'(the 2.5-D contract)')


# ---------------------------------------------------------------------------
# 3-4. data
# ---------------------------------------------------------------------------

def _make_case(root: Path, case_id: str, seed: int, phases):
    d = root / case_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for ph in ['nc'] + list(phases):
        vol = rng.uniform(-50, 150, size=VOL_SHAPE_XYZ).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine=np.eye(4)),
                 str(d / f'{case_id}_{ph}_deeds.nii.gz'))


def test_data():
    print('\n-- multi-phase split --')
    tmp = Path(tempfile.mkdtemp(prefix='phase_cond_'))
    try:
        data = tmp / 'data'
        # 10 cases with both phases; 2 with venous only (mirrors the real set,
        # where one case has no arterial).
        for i in range(10):
            _make_case(data, f'case{i:02d}', i, ['venous', 'arterial'])
        for i in range(10, 12):
            _make_case(data, f'case{i:02d}', i, ['venous'])

        cfg = {'data_dir': str(data), 'file_tag': '_deeds', 'seg_suffix': '_seg_full',
               'val_split': 0.15, 'test_split': 0.15, 'data_seed': 42}

        tr1, va1, te1 = find_pairs_and_split({**cfg, 'target_phase': 'venous'})
        tr2, va2, te2 = find_pairs_and_split(
            {**cfg, 'target_phases': ['venous', 'arterial']})

        check(len(tr1) + len(va1) + len(te1) == 12,
              f'single-phase finds 12 venous pairs (got {len(tr1)+len(va1)+len(te1)})')
        check(len(tr2) + len(va2) + len(te2) == 22,
              f'multi-phase finds 22 pairs = 12 venous + 10 arterial '
              f'(got {len(tr2)+len(va2)+len(te2)})')

        def cases(ps):
            return {p['case_id'] for p in ps}

        # 4. the venous split must be untouched
        for name, a, b in [('train', tr1, tr2), ('val', va1, va2), ('test', te1, te2)]:
            ven = {p['case_id'] for p in b if p['phase'] == 'venous'}
            check(cases(a) == ven,
                  f'{name}: venous cases unchanged by adding a phase '
                  f'({len(cases(a))} cases)')

        # 3. no patient leakage
        ct, cv, cs = cases(tr2), cases(va2), cases(te2)
        check(not (ct & cv) and not (ct & cs) and not (cv & cs),
              'no case appears in two splits')
        by_case = {}
        for split, ps in [('train', tr2), ('val', va2), ('test', te2)]:
            for p in ps:
                by_case.setdefault(p['case_id'], set()).add(split)
        check(all(len(v) == 1 for v in by_case.values()),
              'every phase of a case lands in the SAME split')

        # phase ids are stable and match the configured order
        ids = {p['phase']: p['phase_id'] for p in tr2 + va2 + te2}
        check(ids.get('venous') == 0 and ids.get('arterial') == 1,
              f'phase_id follows target_phases order {ids}')

        # cache key must separate single- from multi-phase
        from dataset import CTPairDataset
        def digest(pairs, c):
            s = CTPairDataset.__new__(CTPairDataset)
            s.split_name = 'train'; s.ph = s.pw = 16; s.patch_depth = 1
            s.stride_h = s.stride_w = 8
            s.min_std, s.min_mean, s.min_max = 0.0, -1e9, -1e9
            s.hu_min, s.hu_max = -200, 400
            s.load_mask = False; s.mask_multilabel = False; s.organ_focus_frac = 0.0
            return CTPairDataset._cache_path(s, pairs, {**c, 'cache_dir': str(tmp)},
                                             100).name
        d1 = digest(tr1, {**cfg, 'target_phase': 'venous'})
        d2 = digest(tr2, {**cfg, 'target_phases': ['venous', 'arterial']})
        check(d1 != d2, f'multi-phase gets a different cache file ({d1} != {d2})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_model()
    test_forward_compat()
    test_data()
    print()
    if _failures:
        print(f'{len(_failures)} FAILED:')
        for f in _failures:
            print('  -', f)
        return 1
    print('phase conditioning PASSED.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
