"""
Fast, GPU-free structural smoke test for the deterministic Trainer.

Exercises every loss-flag combination in `config.py` against tiny synthetic
tensors (no real data, no GPU) for a couple of epochs each, to catch
structural bugs (key mismatches, shape errors, missing-arg errors) in
seconds instead of after a multi-hour remote data preload. Also runs two
representative 3-D scenarios (`3d_l1_only`, `3d_all_losses`) to exercise the
dims=3 code paths (3-D Sobel gradients, per-slice SSIM/FFT, etc).

FORK NOTE: the adversarial / perceptual / feature-matching / saliency / cycle /
seg-consistency scenarios are gone with their losses (see losses.py), and with
them the VGG16 download this test used to need — it now runs fully offline. The
heteroscedastic head is covered here; the diffusion trainer has its own smoke
test in tests/smoke_test_diffusion.py.

Usage:
    python tests/smoke_test.py
"""

import copy
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import train_config
from trainer import Trainer

logging.basicConfig(level=logging.WARNING)  # keep Trainer's per-scenario INFO noise down

PATCH  = 64  # divisible by 16 (UNetGenerator's 4 pooling stages) — see models.py
DEPTH3D = 8  # depth is never pooled (pool_stride=(1,2,2) for dims=3), so arbitrary


class _FakeDataset(Dataset):
    """Random tensors standing in for real NCCT/CECT patches. Mask is always
    present (harmless when a scenario doesn't use it — trainer.py gates on
    `'mask' in batch`). `dims=3` yields (1, D, H, W) volumes instead of the
    default (1, H, W) 2-D slices."""

    def __init__(self, n: int, dims: int = 2):
        self.n = n
        self.shape = (1, DEPTH3D, PATCH, PATCH) if dims == 3 else (1, PATCH, PATCH)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'source': torch.rand(*self.shape),
            'target': torch.rand(*self.shape),
            # multi-label mask (ids 0-3) so both organ-union (mask>0) and the
            # per-organ breakdown paths are exercised.
            'mask':   torch.randint(0, 4, self.shape).float(),
        }


def _fake_loaders(dims: int = 2):
    batch_size = 2 if dims == 2 else 1  # 3-D volumes are heavier per-sample on CPU
    train_loader = DataLoader(_FakeDataset(6, dims=dims), batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(_FakeDataset(4, dims=dims), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


ALL_EXTRA_FLAGS = ['ssim', 'gradient', 'frequency', 'organ', 'hu_profile']


def _scenario_configs() -> dict:
    scenarios = {'l1_only': {}}
    for f in ALL_EXTRA_FLAGS:
        scenarios[f'extra_{f}'] = {f'use_{f}': True}
    scenarios['all_losses'] = {f'use_{f}': True for f in ALL_EXTRA_FLAGS}
    # The heteroscedastic head replaces L1 with Gaussian NLL and emits a second
    # output channel, so it touches the trainer's shape handling everywhere the
    # deterministic path does — worth one scenario each way.
    scenarios['hetero'] = {'use_hetero': True}
    scenarios['hetero_organ'] = {'use_hetero': True, 'use_organ': True}
    return scenarios


def _scenario_configs_3d() -> dict:
    """Two representative 3-D scenarios (not the full 2-D matrix — 3-D convs are
    heavier and this is enough to exercise every loss's 3-D branch: the baseline
    sanity-checks the 3-D UNet forward pass, and 'all_losses' hits every
    3-D-aware code path at once — GradientLoss's 3-D Sobel, FrequencyLoss's
    per-slice FFT, SSIMLoss's per-slice loop, OrganWeightedLoss's rank-agnostic
    indexing."""
    return {
        '3d_l1_only':    {},
        '3d_all_losses': {f'use_{f}': True for f in ALL_EXTRA_FLAGS},
    }


def _run_scenario(name: str, overrides: dict, tmp_root: Path, dims: int = 2):
    cfg = dict(train_config)
    cfg.update(overrides)
    cfg['device']       = 'cpu'          # deliberately CPU-only: must work anywhere
    cfg['output_dir']   = tmp_root / name
    cfg['dims']         = dims
    cfg['patch_size']   = PATCH
    cfg['patch_depth']  = DEPTH3D if dims == 3 else 1
    cfg['save_samples_interval'] = 999   # skip sample-grid plotting overhead

    train_loader, val_loader = _fake_loaders(dims=dims)
    try:
        trainer = Trainer(cfg)
        trainer.train(train_loader, val_loader, epochs=2, start_epoch=0)
        return True, None
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix='litbaseline_smoketest_'))
    scenarios = list(_scenario_configs().items()) + \
                [(name, ov) for name, ov in _scenario_configs_3d().items()]
    dims_by_name = {name: 3 for name in _scenario_configs_3d()}
    results = {}
    try:
        for name, overrides in scenarios:
            dims = dims_by_name.get(name, 2)
            ok, err = _run_scenario(name, overrides, tmp_root, dims=dims)
            results[name] = (ok, err)
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f'  -- {err}' if err else ''))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    n_fail = sum(1 for ok, _ in results.values() if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} scenarios passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
