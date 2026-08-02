"""
CPU-only smoke test for infer_volume.py (Phase 1 volume inference).

Two parts, no GPU / real data / trained weights needed:
  A) infer_volume() on a tiny volume with an untrained generator — checks full
     voxel coverage (no gaps), output shape == input shape, HU range sane, for
     BOTH 2-D (patch_depth=1) and 3-D (patch_depth>1) paths.
  B) the run() driver end-to-end on a synthetic scenario (fake data dir + labels
     + run_config.json + a saved generator) — checks it writes synthetic NIfTIs
     and a manifest whose gen/real/mask volumes are all the SAME shape (the exact
     requirement phase_eval.py enforces).

Usage:  python tests/smoke_test_infer.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models import UNetGenerator
from infer_volume import DeterministicPredictor, infer_volume, load_model, run


def _infer(G, vol, cfg, device='cpu', **kw):
    """Shim for the current infer_volume signature.

    It now takes a `PatchPredictor` rather than a generator (so diffusion and the
    heteroscedastic head can share one tiling loop) and returns (n_out, D, H, W).
    These tests all exercise the plain deterministic path and want (D, H, W).
    """
    return infer_volume(DeterministicPredictor(G, device, False),
                        vol, cfg, device, **kw)[0]


def check(cond, name, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  -- {detail}' if detail and not cond else ''))
    return bool(cond)


def _save(path, arr, affine=None):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine if affine is not None else np.eye(4)), str(path))


def part_a():
    ok = True
    hu_min, hu_max = -200.0, 400.0

    # 2-D: volume (D=6, H=64, W=64), patch 32, patch_depth=1
    G2 = UNetGenerator(dims=2, base_channels=8, dropout=0.0).eval()
    cfg2 = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5, hu_min=hu_min, hu_max=hu_max)
    vol2 = np.random.uniform(-500, 500, size=(6, 64, 64)).astype(np.float32)
    syn2 = _infer(G2, vol2, cfg2, device='cpu', batch_size=8)
    ok &= check(syn2.shape == vol2.shape, '2D: output shape == input', f'{syn2.shape} vs {vol2.shape}')
    ok &= check(not np.isnan(syn2).any(), '2D: no NaN (full coverage, no div-by-zero)')
    ok &= check(syn2.min() >= hu_min - 1 and syn2.max() <= hu_max + 1,
                '2D: output within HU window', f'[{syn2.min():.0f},{syn2.max():.0f}]')

    # 3-D: volume (D=8, H=32, W=32), patch 32, patch_depth=4
    G3 = UNetGenerator(dims=3, base_channels=8, dropout=0.0).eval()
    cfg3 = dict(dims=3, patch_depth=4, patch_size=32, overlap=0.5, hu_min=hu_min, hu_max=hu_max)
    vol3 = np.random.uniform(-500, 500, size=(8, 32, 32)).astype(np.float32)
    syn3 = _infer(G3, vol3, cfg3, device='cpu', batch_size=4)
    ok &= check(syn3.shape == vol3.shape, '3D: output shape == input', f'{syn3.shape} vs {vol3.shape}')
    ok &= check(not np.isnan(syn3).any(), '3D: no NaN (full coverage)')

    # tiny-volume padding path (D<patch_depth for 3D)
    vol3s = np.random.uniform(-500, 500, size=(2, 32, 32)).astype(np.float32)
    syn3s = _infer(G3, vol3s, cfg3, device='cpu', batch_size=4)
    ok &= check(syn3s.shape == vol3s.shape, '3D: sub-patch depth padded+cropped back', f'{syn3s.shape}')
    return ok


class _Identity(torch.nn.Module):
    """Stand-in generator that returns its input unchanged."""
    def forward(self, x):
        return x


def part_c():
    """Blending correctness.

    The invariant that matters: with an identity generator, blending must
    reconstruct the input EXACTLY. Sum(w*x)/Sum(w) == x for any weights, so any
    deviation means the weight and value accumulators disagree — a mis-indexed
    window, a window applied to one and not the other, or a coverage hole. This
    catches the whole class of stitching bugs that a shape/NaN check cannot.
    """
    ok = True
    hu_min, hu_max = -200.0, 400.0
    G = _Identity().eval()
    # Values inside the HU window so the clip is a no-op and round-trip is exact.
    vol = np.random.uniform(hu_min + 10, hu_max - 10, size=(4, 96, 96)).astype(np.float32)

    for mode in ('uniform', 'hann', 'gaussian'):
        cfg = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5,
                   hu_min=hu_min, hu_max=hu_max)
        syn = _infer(G, vol, cfg, device='cpu', batch_size=8, blend=mode)
        err = float(np.abs(syn - vol).max())
        ok &= check(err < 0.05, f'blend={mode}: identity round-trip exact',
                    f'max err {err:.4f} HU')

    # Edge margin must still cover every voxel, including the volume's own shell.
    cfg = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5,
               hu_min=hu_min, hu_max=hu_max)
    syn = _infer(G, vol, cfg, device='cpu', batch_size=8,
                       blend='hann', edge_margin=8)
    err = float(np.abs(syn - vol).max())
    ok &= check(not np.isnan(syn).any() and err < 0.05,
                'edge_margin=8: full coverage incl. volume border', f'max err {err:.4f} HU')

    # Inference-time overlap override changes the tiling, not the result.
    syn = _infer(G, vol, cfg, device='cpu', batch_size=8,
                       blend='hann', overlap=0.75)
    ok &= check(float(np.abs(syn - vol).max()) < 0.05,
                'overlap override: still exact')

    # 3-D path with weighting.
    cfg3 = dict(dims=3, patch_depth=4, patch_size=32, overlap=0.5,
                hu_min=hu_min, hu_max=hu_max)
    vol3 = np.random.uniform(hu_min + 10, hu_max - 10, size=(8, 32, 32)).astype(np.float32)
    syn3 = _infer(G, vol3, cfg3, device='cpu', batch_size=4, blend='hann')
    ok &= check(float(np.abs(syn3 - vol3).max()) < 0.05, '3D: weighted blend exact')

    # A tapered window must actually taper, or 'hann' is silently 'uniform'.
    from infer_volume import _axis_window
    w_int = _axis_window(32, 'hann', 0, at_lo=False, at_hi=False)
    ok &= check(w_int[0] < 0.05 * w_int.max() and w_int[0] > 0,
                'hann window tapers at tile edge but stays > 0', f'w[0]={w_int[0]:.5f}')
    w_edge = _axis_window(32, 'hann', 8, at_lo=True, at_hi=False)
    ok &= check(w_edge[0] > 0 and w_edge[-1] == 0,
                'margin skipped on the volume-boundary side only')
    return ok


def part_b():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix='infer_smoke_'))
    try:
        data = tmp / 'deeds'; data.mkdir()
        # 4 studies so a val+test split is non-empty; each: NC + venous + venous seg.
        # Phase is encoded in the filename ('nc'/'pv') so find_pairs_and_split uses
        # _infer_phase and needs no labels.csv (avoids a pandas dependency in the
        # test env; the real remote run uses the labels.csv path).
        XYZ = (48, 48, 6)   # -> _load_vol -> (D=6,H=48,W=48)
        for i in range(4):
            st = f'case{i}'
            cdir = data / st; cdir.mkdir()
            for ser in [f'{st}_nc', f'{st}_pv']:
                base = f'{st}_{ser}_deeds'
                _save(cdir / f'{base}.nii.gz', np.random.uniform(-500, 500, XYZ))
                if ser.endswith('_pv'):   # seg_full mask only needed for the target
                    _save(cdir / f'{base}_seg_full.nii.gz',
                          np.random.randint(0, 5, XYZ).astype(np.float32))

        # scenario dir: run_config.json + best_model.pth
        sdir = tmp / 'scenario'; sdir.mkdir()
        cfg = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5, hu_min=-200, hu_max=400,
                   generator_base_channels=8, generator_dropout=0.0, target_phase='venous',
                   data_dir=str(data), labels_csv='',
                   file_tag='_deeds', seg_suffix='_seg_full',
                   val_split=0.25, test_split=0.25, seed=42)
        (sdir / 'run_config.json').write_text(json.dumps(cfg))
        G = UNetGenerator(dims=2, base_channels=8, dropout=0.0)
        torch.save({'G_state': G.state_dict(), 'epoch': 1}, sdir / 'best_model.pth')

        out = sdir / 'phase_infer'
        run(str(sdir), 'both', str(out), 'best_model.pth', batch_size=8, device='cpu')

        manifest = out / 'manifest.csv'
        ok &= check(manifest.exists(), 'driver wrote manifest.csv')
        with open(manifest) as f:
            mrows = list(csv.DictReader(f))
        ok &= check(len(mrows) > 0, 'manifest has rows', f'{len(mrows)}')
        # the phase_eval requirement: gen/real/mask all same shape
        allmatch = True
        for r in mrows:
            g = nib.load(r['gen_path']).get_fdata().shape
            rl = nib.load(r['real_path']).get_fdata().shape
            mk = nib.load(r['mask_path']).get_fdata().shape
            if not (g == rl == mk):
                allmatch = False
                print(f"    shape mismatch: gen{g} real{rl} mask{mk}")
        ok &= check(allmatch, 'gen/real/mask shapes match (phase_eval requirement)')
        # target_phase written as name
        ok &= check(all(r['target_phase'] == 'venous' for r in mrows), "manifest target_phase='venous'")
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    print("--- Part A: infer_volume coverage/shape (2D + 3D) ---")
    a = part_a()
    print("\n--- Part B: run() driver end-to-end ---")
    b = part_b()
    print("\n--- Part C: overlap blending correctness ---")
    c = part_c()
    ok = a and b and c
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
