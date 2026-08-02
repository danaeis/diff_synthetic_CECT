"""
End-to-end smoke test: train and sample, on random tensors, on CPU, in seconds.

No data, no GPU, no checkpoints from a real run. What it proves is only that the
pieces fit together — DiffusionTrainer builds, its step runs, validation is
deterministic, checkpoints round-trip, the heteroscedastic path works, and
infer_volume's tiling loop reconstructs a volume through every predictor. Every
claim about quality comes from the server, not from here.

The one substantive check is `test_val_determinism`: the diffusion val loss must
be identical across epochs for an unchanged model. If it is not, checkpoint
selection is selecting on which timesteps happened to be drawn.

Run directly:
    python tests/smoke_test_diffusion.py
"""

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


class FakePatches(Dataset):
    """Minimal stand-in for CTPairDataset: the same keys, random content.

    A multi-label 'mask' is included so the organ / HU-profile losses take their
    real code path rather than the mask-is-None shortcut — those are the ones
    attached to the predicted x0, which is the part worth exercising.
    """

    def __init__(self, n=16, size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.src = torch.rand(n, 1, size, size, generator=g)
        self.tgt = torch.rand(n, 1, size, size, generator=g)
        # Labels 0..3; the LUT in OrganWeightedLoss is indexed by these.
        self.msk = torch.randint(0, 4, (n, 1, size, size), generator=g).float()
        self.case_ids = [f'case{i // 4}' for i in range(n)]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return {'source': self.src[i], 'target': self.tgt[i], 'mask': self.msk[i]}


def base_cfg(out):
    return dict(
        device='cpu', output_dir=str(out), dims=2, patch_depth=1, patch_size=32,
        overlap=0.5, generator_base_channels=8, generator_dropout=0.1,
        generator_norm='group', in_channels=1, n_input_slices=1,
        hu_min=-200, hu_max=400, batch_size=4, epochs=2, learning_rate=1e-3,
        use_mixed_precision=False, use_cosine_schedule=False,
        early_stop_patience=10, keep_last_n_checkpoints=1,
        keep_last_n_sample_epochs=1, save_samples_interval=1,
        report_per_organ_metrics=False, seed=42,
        target_phase='venous', target_phases=['venous'],
        diffusion_steps=100, diffusion_sample_every=1, diffusion_sample_steps=4,
        aux_max_t=70,
    )


# ---------------------------------------------------------------------------

def test_diffusion_trainer():
    print('\ndiffusion trainer')
    from trainer_diffusion import DiffusionTrainer
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        cfg = base_cfg(out)
        cfg.update(use_diffusion=True, parameterisation='v',
                   use_organ=True, use_hu_profile=True,
                   organ_weights={1: 3.0, 2: 0.0, 3: 1.0},
                   lambda_organ=20.0, lambda_hu_profile=50.0)
        torch.manual_seed(42)
        tr = DiffusionTrainer(cfg)
        ds = FakePatches()
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        tr.train(dl, dl, epochs=2)

        check('history recorded 2 epochs', tr.history['epoch'] == [1, 2])
        check('train loss finite',
              all(np.isfinite(v) for v in tr.history['train_gen_total']))
        check('organ loss is nonzero',
              any(v > 0 for v in tr.history['train_organ']),
              f"{tr.history['train_organ']}")
        check('hu_profile loss is nonzero',
              any(v > 0 for v in tr.history['train_hu_profile']))
        check('best_model.pth written', (out / 'best_model.pth').exists())
        check('history.json written', (out / 'history.json').exists())
        check('sample grid drawn from a DDIM sample',
              any(out.joinpath('samples').glob('ep*.png')))
        # FiLM gammas start at exactly 0 and must move, or the timestep
        # conditioning is receiving no gradient at all.
        check('FiLM(t) gammas moved off zero',
              max(tr.history['gamma_enc1'] + tr.history['gamma_dec1']) > 0,
              f"enc1 {tr.history['gamma_enc1']}")
        check('sampled val_org_ssim recorded',
              any(v != 0 for v in tr.history['val_org_ssim']))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_val_determinism():
    """The val loss must be identical for an unchanged model, across calls.

    This is the property the fixed (t, noise) grid exists for. Without it the
    epoch-to-epoch val curve is dominated by timestep sampling noise and
    `_selection_score` picks a checkpoint at random.
    """
    print('\ndeterministic validation')
    from trainer_diffusion import DiffusionTrainer
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        cfg = base_cfg(out)
        cfg.update(use_diffusion=True, diffusion_sample_every=10 ** 6)
        torch.manual_seed(42)
        tr = DiffusionTrainer(cfg)
        tr.current_epoch = 1
        dl = DataLoader(FakePatches(), batch_size=4, shuffle=False)
        a = tr._validate(dl)['val_loss']
        b = tr._validate(dl)['val_loss']
        check('val loss is reproducible', a == b, f'{a!r} vs {b!r}')
        check('val loss is finite and positive', np.isfinite(a) and a > 0, f'{a:.6f}')

        # A DIFFERENT model must give a different loss, or the check above is
        # passing because the function ignores the model entirely.
        torch.manual_seed(7)
        tr2 = DiffusionTrainer(cfg)
        tr2.current_epoch = 1
        check('val loss depends on the model', tr2._validate(dl)['val_loss'] != a)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_checkpoint_roundtrip():
    print('\ncheckpoint round-trip')
    from trainer_diffusion import DiffusionTrainer
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        cfg = base_cfg(out)
        cfg.update(use_diffusion=True, diffusion_sample_every=10 ** 6)
        torch.manual_seed(42)
        tr = DiffusionTrainer(cfg)
        dl = DataLoader(FakePatches(), batch_size=4, shuffle=False)
        tr.train(dl, dl, epochs=1)
        w_before = tr.G.out_conv.weight.detach().clone()

        torch.manual_seed(99)                       # different init
        tr2 = DiffusionTrainer(cfg)
        ep = tr2.load_checkpoint(str(out / 'best_model.pth'))
        check('resumed at the right epoch', ep == 1, f'{ep}')
        check('weights restored',
              torch.equal(tr2.G.out_conv.weight.detach(), w_before))
        check('history restored', tr2.history['epoch'] == [1])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_hetero_trainer():
    print('\nheteroscedastic trainer')
    from trainer import Trainer
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        cfg = base_cfg(out)
        cfg.update(use_hetero=True, use_organ=True,
                   organ_weights={1: 3.0, 2: 0.0, 3: 1.0}, lambda_organ=20.0,
                   selection_metric='val_org_ssim')
        torch.manual_seed(42)
        tr = Trainer(cfg)
        dl = DataLoader(FakePatches(), batch_size=4, shuffle=False)
        tr.train(dl, dl, epochs=2)
        check('NLL term is active', any(v != 0 for v in tr.history['train_nll']))
        check('L1 is logged but not optimised',
              all(v == 0 for v in tr.history['lambda_l1']))
        check('val metrics use the mu channel only',
              0.0 < tr.history['val_mae'][-1] < 1.0,
              f"{tr.history['val_mae'][-1]:.4f}")
        check('best_model.pth written', (out / 'best_model.pth').exists())
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_l1_trainer_still_works():
    """The deterministic path must survive the strip — it is the reference run."""
    print('\ndeterministic (L1) trainer')
    from trainer import Trainer
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        cfg = base_cfg(out)
        cfg.update(use_organ=True, organ_weights={1: 3.0, 2: 0.0, 3: 1.0},
                   lambda_organ=20.0, use_l1_decay=True, lambda_l1=100.0,
                   lambda_l1_floor=25.0, l1_decay_start_epoch=1,
                   l1_decay_end_epoch=2)
        torch.manual_seed(42)
        tr = Trainer(cfg)
        dl = DataLoader(FakePatches(), batch_size=4, shuffle=False)
        tr.train(dl, dl, epochs=2)
        check('L1 term active', all(v > 0 for v in tr.history['train_l1']))
        check('L1 decay curriculum ran',
              tr.history['lambda_l1'][-1] < tr.history['lambda_l1'][0],
              f"{tr.history['lambda_l1']}")
        check('NLL stays zero', all(v == 0 for v in tr.history['train_nll']))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_infer_volume_predictors():
    """The tiling loop must reconstruct a full volume through every predictor."""
    print('\ninfer_volume tiling')
    from infer_volume import (DeterministicPredictor, DiffusionPredictor,
                              HeteroPredictor, infer_volume)
    from models import UNetGenerator
    from models_hetero import HeteroGenerator
    from models_diffusion import DiffusionUNet, NoiseSchedule

    cfg = {'patch_depth': 1, 'patch_size': 32, 'overlap': 0.5,
           'hu_min': -200, 'hu_max': 400, 'n_input_slices': 1}
    vol = (np.random.default_rng(0).random((3, 64, 70)) * 600 - 200).astype(np.float32)

    n_tiles, per_slice = infer_volume(None, vol, cfg, 'cpu', count_only=True)
    check('dry run counts tiles', n_tiles == 3 * per_slice and per_slice > 1,
          f'{n_tiles} tiles, {per_slice}/slice')

    torch.manual_seed(0)
    G = UNetGenerator(dims=2, base_channels=8, norm='group'); G.eval()
    r = infer_volume(DeterministicPredictor(G, 'cpu', False), vol, cfg, 'cpu')
    check('deterministic: shape + HU range',
          r.shape == (1,) + vol.shape and -201 <= r.min() and r.max() <= 401,
          f'{r.shape}, [{r.min():.0f}, {r.max():.0f}]')

    H = HeteroGenerator(dims=2, base_channels=8, norm='group'); H.eval()
    r2 = infer_volume(HeteroPredictor(H, 'cpu', False), vol, cfg, 'cpu')
    check('hetero: two channels', r2.shape == (2,) + vol.shape, f'{r2.shape}')
    # The sd channel is a spread: scaled by the window width, never offset, so it
    # must be non-negative. Offsetting it by hu_min would make it about -200.
    check('hetero: sd channel is non-negative', r2[1].min() >= 0,
          f'min {r2[1].min():.3f}')

    M = DiffusionUNet(dims=2, base_channels=8, cond_channels=1); M.eval()
    p = DiffusionPredictor(M, NoiseSchedule(100), 'cpu', False, cfg,
                           ddim_steps=3, case_id='c1', sample_seed=1)
    a = infer_volume(p, vol, cfg, 'cpu')
    check('diffusion: shape', a.shape == (1,) + vol.shape, f'{a.shape}')
    check('diffusion: finite', bool(np.isfinite(a).all()))

    # Same (seed, case, k) must give the same volume; a different k must not.
    p2 = DiffusionPredictor(M, NoiseSchedule(100), 'cpu', False, cfg,
                            ddim_steps=3, case_id='c1', sample_seed=1)
    check('diffusion: sample is reproducible',
          np.array_equal(a, infer_volume(p2, vol, cfg, 'cpu')))
    p2.set_sample(1)
    b = infer_volume(p2, vol, cfg, 'cpu')
    check('diffusion: sample index changes the volume', not np.array_equal(a, b))
    # And the two samples must differ by more than float noise, or there is no
    # spread to calibrate.
    check('diffusion: samples differ substantially',
          float(np.abs(a - b).mean()) > 1e-3, f'mean |a-b| = {np.abs(a-b).mean():.4f} HU')


if __name__ == '__main__':
    print('=' * 70)
    print('SMOKE TEST — diffusion + heteroscedastic, CPU, random tensors')
    print('=' * 70)
    import logging
    logging.disable(logging.INFO)
    test_diffusion_trainer()
    test_val_determinism()
    test_checkpoint_roundtrip()
    test_hetero_trainer()
    test_l1_trainer_still_works()
    test_infer_volume_predictors()
    print('\n' + '=' * 70)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print('ALL PASS')
