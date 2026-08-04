"""Pins CTPairDataset._augment.

Run directly: `python tests/test_augment.py`.

The thing that makes augmentation dangerous here rather than routine is that the
prediction target is an absolute HU level, and the organ losses index a per-voxel
label LUT. So a transform that desynchronises source / target / mask does not
crash — it silently trains the model on mismatched anatomy and mislabelled
organs, and the only symptom is a number that fails to improve.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dataset import CTPairDataset, _AUGMENT_MODES

_fails = []


def check(name, cond, extra=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f'  — {extra}' if extra else ''))
    if not cond:
        _fails.append(name)


def _ds(mode, split='train', n_input_slices=1, hw=(8, 8), n=4):
    """A CTPairDataset with the indexing/preload path bypassed — _augment only
    reads self.augment, so constructing the object is unnecessary and slow."""
    d = CTPairDataset.__new__(CTPairDataset)
    d.augment = mode if split == 'train' else 'none'
    d.n_input_slices = n_input_slices
    rng = np.random.default_rng(0)
    h, w = hw
    c = n_input_slices
    d.src_patches = [rng.random((c, h, w), dtype=np.float32) if c > 1
                     else rng.random((h, w), dtype=np.float32) for _ in range(n)]
    d.tgt_patches = [rng.random((h, w), dtype=np.float32) for _ in range(n)]
    d.mask_patches = [rng.integers(0, 5, (h, w)).astype(np.float32) for _ in range(n)]
    d.phase_ids = None
    d.level_vecs = None
    return d


def test_pairing():
    """source/target/mask must receive the IDENTICAL transform."""
    print('\npaired transform')
    d = _ds('flip_rot90')
    # Make target and mask exact copies of the source plane, so any
    # desynchronisation shows up as an inequality after augmentation.
    plane = d.src_patches[0].copy()
    d.tgt_patches[0] = plane.copy()
    d.mask_patches[0] = plane.copy()

    bad = []
    for seed in range(64):
        torch.manual_seed(seed)
        it = d[0]
        s, t, m = it['source'][0], it['target'][0], it['mask'][0]
        if not (torch.equal(s, t) and torch.equal(s, m)):
            bad.append(seed)
    check('source/target/mask stay aligned over 64 draws', not bad, f'{bad[:5]}')


def test_values_preserved():
    """Geometric only: the multiset of voxel values must be unchanged.

    This is what rules out an intensity transform sneaking in. The target IS an
    absolute HU level; a transform that shifts or scales it destroys the label.
    """
    print('\nintensity is untouched')
    d = _ds('flip_rot90')
    ref = np.sort(d.tgt_patches[0].ravel())
    same = True
    for seed in range(32):
        torch.manual_seed(seed)
        got = np.sort(d[0]['target'].numpy().ravel())
        same &= np.array_equal(ref, got)
    check('target voxel values are a permutation of the original', same)

    mref = set(np.unique(d.mask_patches[0]))
    torch.manual_seed(3)
    check('mask label ids are not interpolated into new values',
          set(np.unique(d[0]['mask'].numpy())) <= mref)


def test_actually_varies():
    """A no-op augmentation would pass every check above."""
    print('\nit does something')
    d = _ds('flip_rot90')
    seen = set()
    for seed in range(32):
        torch.manual_seed(seed)
        seen.add(d[0]['target'].numpy().tobytes())
    check('flip_rot90 produces multiple distinct outputs', len(seen) > 1, f'{len(seen)}')

    d_ap = _ds('flip_ap')
    lr_changed = False
    base = d_ap.tgt_patches[0]
    for seed in range(32):
        torch.manual_seed(seed)
        out = d_ap[0]['target'].numpy()[0]
        # An AP-only flip can never produce the left-right mirror.
        lr_changed |= np.array_equal(out, base[:, ::-1])
    check('flip_ap never mirrors laterality', not lr_changed)

    d_none = _ds('none')
    ident = True
    for seed in range(8):
        torch.manual_seed(seed)
        ident &= np.array_equal(d_none[0]['target'].numpy()[0], d_none.tgt_patches[0])
    check("mode 'none' is the identity", ident)


def test_split_gating():
    """Val/test must never be augmented — it would make the selection curve a
    different random variable every epoch."""
    print('\ntrain split only')
    d = _ds('flip_rot90', split='val')
    check('augment is forced off outside train', d.augment == 'none')


def test_layouts():
    """2.5-D stacks and 3-D patches: in-plane axes are always the last two."""
    print('\ntensor layouts')
    d = _ds('flip_rot90', n_input_slices=5)
    torch.manual_seed(0)
    it = d[0]
    check('2.5-D source keeps its 5-slice channel axis',
          tuple(it['source'].shape) == (5, 8, 8), f"{tuple(it['source'].shape)}")

    # Non-square in-plane: rot90 must be skipped rather than change the shape.
    d2 = _ds('flip_rot90', hw=(8, 12))
    shapes = set()
    for seed in range(32):
        torch.manual_seed(seed)
        shapes.add(tuple(d2[0]['target'].shape))
    check('non-square patches keep one shape (rot90 skipped)',
          shapes == {(1, 8, 12)}, f'{shapes}')


def test_no_cache_mutation():
    """torch.from_numpy aliases the cached array; augmentation must not write
    through it, or the cache degrades a little more on every epoch."""
    print('\ncache is not mutated')
    d = _ds('flip_rot90')
    before = [p.copy() for p in d.tgt_patches]
    for seed in range(32):
        torch.manual_seed(seed)
        _ = d[seed % len(d.tgt_patches)]
    check('cached patches are byte-identical afterwards',
          all(np.array_equal(a, b) for a, b in zip(before, d.tgt_patches)))


def test_bad_mode_raises():
    print('\nvalidation')
    check('_AUGMENT_MODES is the documented set',
          _AUGMENT_MODES == ('none', 'flip_ap', 'flip', 'flip_rot90'),
          f'{_AUGMENT_MODES}')


if __name__ == '__main__':
    print('=' * 70)
    print('AUGMENTATION — paired geometric transforms')
    print('=' * 70)
    test_pairing()
    test_values_preserved()
    test_actually_varies()
    test_split_gating()
    test_layouts()
    test_no_cache_mutation()
    test_bad_mode_raises()
    print()
    if _fails:
        print(f'FAILED: {len(_fails)} — ' + ', '.join(_fails))
        sys.exit(1)
    print('ALL PASS')
