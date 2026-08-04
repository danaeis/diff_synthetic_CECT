"""Pins the Step-3 diffusion changes: offset noise, min-SNR-gamma, EMA.

Run directly: `python tests/test_diffusion_core.py`.

Each of these is a change to what the model is trained on or which weights get
shipped, so each one can be wrong in a way that produces a plausible-looking
loss curve and a worse model. The offset-noise checks in particular exist
because it MUST be applied on both the training and the sampling side — applying
it on one side only is a train/test mismatch in exactly the channel this project
measures, and nothing else in the pipeline would notice.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from models_diffusion import NoiseSchedule, offset_noise
from trainer import WeightEMA

_fails = []


def check(name, cond, extra=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f'  — {extra}' if extra else ''))
    if not cond:
        _fails.append(name)


# ---------------------------------------------------------------------------
def test_offset_noise():
    print('\noffset noise')
    x = torch.zeros(4096, 1, 32, 32)

    torch.manual_seed(0)
    plain = offset_noise(x, 0.0)
    torch.manual_seed(0)
    check('c=0 is exactly randn_like', torch.equal(plain, torch.randn_like(x)))

    torch.manual_seed(1)
    off = offset_noise(x, 0.1)
    # Per-item DC: the spread of patch means must rise from 1/sqrt(HW) to ~c.
    dc_plain = plain.mean(dim=(1, 2, 3)).std().item()
    dc_off = off.mean(dim=(1, 2, 3)).std().item()
    expect_plain = 1.0 / np.sqrt(32 * 32)
    check('plain noise DC sd ~ 1/sqrt(HW)',
          abs(dc_plain - expect_plain) < 0.3 * expect_plain,
          f'{dc_plain:.5f} vs {expect_plain:.5f}')
    check('offset noise DC sd ~ c (the whole point)',
          abs(dc_off - 0.1) < 0.02, f'{dc_off:.5f}')
    check('offset raises the DC spread by ~c/(1/sqrt(HW))',
          dc_off / dc_plain > 2.0, f'{dc_off / dc_plain:.1f}x')

    # The offset is CONSTANT within an item — that is what makes it a level.
    per_item_spatial_std = (off[0, 0] - off[0, 0].mean()).std().item()
    check('within-item texture is untouched',
          abs(per_item_spatial_std - 1.0) < 0.1, f'{per_item_spatial_std:.4f}')

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    check('generator makes it reproducible',
          torch.equal(offset_noise(x[:8], 0.1, generator=g1),
                      offset_noise(x[:8], 0.1, generator=g2)))

    for ndim_shape in [(2, 1, 16, 16), (2, 1, 4, 16, 16), (2, 3, 16, 16)]:
        t = torch.zeros(*ndim_shape)
        check(f'shape {ndim_shape} round-trips',
              offset_noise(t, 0.1).shape == t.shape)


def test_offset_noise_reaches_inference():
    """The sampling half: one constant for the WHOLE volume, not per tile.

    Per-tile offsets would give every tile its own contrast level and average
    the case-level offset away across ~49 tiles per slice — the exact plumbing
    failure the volume-wide noise field exists to prevent.
    """
    print('\noffset noise, inference side')
    import infer_volume

    class _Cfg(dict):
        pass

    made = {}

    class _FakePred(infer_volume.DiffusionPredictor):
        def __init__(self, c):
            # Bypass the model/sampler construction; _build_noise needs none of it.
            self.sample_seed, self.case_id, self.k = 0, 'case-a', 0
            self.offset_noise = c
            self._shape = (8, 32, 32)
            self._noise = None

    a = _FakePred(0.0); a._build_noise()
    b = _FakePred(0.1); b._build_noise()
    made['plain'], made['off'] = a._noise, b._noise

    # Same seed, so the fields differ only by the added constant.
    delta = (made['off'] - made['plain'])
    check('inference offset is a single constant over the whole volume',
          float(delta.std()) < 1e-6, f'std {float(delta.std()):.3e}')
    check('inference offset is non-zero', abs(float(delta.mean())) > 1e-4,
          f'{float(delta.mean()):.4f}')

    c1 = _FakePred(0.1); c1._build_noise()
    check('same (seed, case, k) still reproduces the field',
          torch.equal(c1._noise, made['off']))

    d = _FakePred(0.1); d.case_id = 'case-b'; d._build_noise()
    check('a different case gets a different offset',
          not torch.equal(d._noise, made['off']))


# ---------------------------------------------------------------------------
def test_snr_loss_weight():
    """The property that matters is the EFFECTIVE weight on the x0 error.

    A weight tested in its own parameterisation's units proves nothing: plain
    v-MSE looks like a uniform weight of 1.0 and is in fact a 24,000x tilt
    toward t=0. Everything below is converted to the common currency first.
    """
    print('\nSNR loss weighting')
    s = NoiseSchedule(n_steps=1000)
    t = torch.tensor([0, 250, 500, 700, 900, 999])
    snr = s.snr(t)
    check('SNR is monotonically decreasing in t',
          bool((snr[1:] < snr[:-1]).all()), f'{snr.tolist()}')

    implicit = {'x0': torch.ones_like(snr), 'v': snr + 1.0}

    # The premise of the whole change: plain v-MSE is NOT flat in x0 space.
    check('plain v-MSE over-weights t=0 by >1000x vs t=999',
          float(implicit['v'][0] / implicit['v'][-1]) > 1e3,
          f"{float(implicit['v'][0] / implicit['v'][-1]):.3g}x")

    eff = {}
    for param in ('v', 'x0'):
        w = s.snr_loss_weight(t, 5.0, param)
        check(f'{param}: weights are finite and positive',
              bool(torch.isfinite(w).all() and (w > 0).all()), f'{w.tolist()}')
        eff[param] = w * implicit[param]

    check('both parameterisations get the SAME effective x0 weighting',
          bool(torch.allclose(eff['v'], eff['x0'], rtol=1e-4)),
          f"{eff['v'].tolist()} vs {eff['x0'].tolist()}")

    e = eff['x0']
    # This is the deviation from the published min-SNR-gamma target, and it is
    # the point: the level lives at high t, so high t must keep FULL weight.
    high = snr <= 5.0
    check('high-t (SNR <= gamma) keeps weight 1.0',
          bool(torch.allclose(e[high], torch.ones_like(e[high]), atol=1e-5)),
          f'{e[high].tolist()}')
    check('the published min(SNR,gamma) target would zero t=999; this does not',
          float(e[-1]) > 0.99, f'{float(e[-1]):.4g}')
    check('low-t domination is removed', float(e[0]) < 1e-2, f'{float(e[0]):.3g}')
    check('effective weight is non-decreasing in t',
          bool((e[1:] >= e[:-1] - 1e-6).all()), f'{e.tolist()}')

    # gamma=inf must reduce to "no reweighting at all".
    w_inf = s.snr_loss_weight(t, float('inf'), 'x0')
    check('gamma=inf is the identity weighting',
          bool(torch.allclose(w_inf, torch.ones_like(w_inf))), f'{w_inf.tolist()}')

    try:
        s.snr_loss_weight(t, 5.0, 'eps')
        check('unknown parameterisation raises', False)
    except ValueError:
        check('unknown parameterisation raises', True)


# ---------------------------------------------------------------------------
def test_ema():
    print('\nweight EMA')
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Conv2d(4, 1, 1))
    ema = WeightEMA(m, decay=0.9)
    start = {k: v.clone() for k, v in m.state_dict().items()}

    for _ in range(200):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        ema.update(m)

    avg = ema.averaged_state(m)
    live = m.state_dict()
    check('EMA differs from the live weights',
          not all(torch.equal(avg[k], live[k]) for k in avg))
    check('EMA differs from the initial weights',
          not all(torch.equal(avg[k], start[k]) for k in avg))
    check('EMA keeps dtypes',
          all(avg[k].dtype == live[k].dtype for k in avg))
    check('EMA state has every float key',
          set(avg) == set(live), f'{set(live) - set(avg)}')

    # The average of a random walk must be less extreme than the endpoint.
    w = '0.weight'
    check('EMA is smoother than the live iterate',
          float((avg[w] - start[w]).abs().mean()) < float((live[w] - start[w]).abs().mean()),
          f"{float((avg[w] - start[w]).abs().mean()):.4f} vs "
          f"{float((live[w] - start[w]).abs().mean()):.4f}")

    # swapped() must restore EXACTLY, or training silently continues from the
    # average after every validation pass.
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with ema.swapped(m):
        inside = {k: v.clone() for k, v in m.state_dict().items()}
    after = m.state_dict()
    check('swapped() installs the averaged weights',
          all(torch.equal(inside[k], avg[k]) for k in avg))
    check('swapped() restores the live weights exactly',
          all(torch.equal(after[k], before[k]) for k in before))

    st = ema.state_dict()
    ema2 = WeightEMA(m, decay=0.9)
    ema2.load_state_dict(st)
    check('EMA state round-trips',
          all(torch.equal(ema2.shadow[k], ema.shadow[k]) for k in ema.shadow)
          and ema2.step == ema.step)

    # Warmup: a fresh EMA must track the model closely rather than lag ~1/(1-d).
    torch.manual_seed(1)
    m2 = nn.Conv2d(1, 2, 3)
    e2 = WeightEMA(m2, decay=0.999)
    with torch.no_grad():
        m2.weight.mul_(0.0).add_(5.0)
    e2.update(m2)
    check('warmup ramp moves the average on step 1',
          float(e2.shadow['weight'].mean()) > 0.3,
          f"{float(e2.shadow['weight'].mean()):.3f}")


if __name__ == '__main__':
    print('=' * 70)
    print('DIFFUSION CORE — offset noise, min-SNR-gamma, EMA')
    print('=' * 70)
    test_offset_noise()
    test_offset_noise_reaches_inference()
    test_snr_loss_weight()
    test_ema()
    print()
    if _fails:
        print(f'FAILED: {len(_fails)} — ' + ', '.join(_fails))
        sys.exit(1)
    print('ALL PASS')
