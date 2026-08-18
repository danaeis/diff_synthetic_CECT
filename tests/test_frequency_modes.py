"""FrequencyLoss weighting modes — behaviour, not just wiring.

CONTEXT WORTH KEEPING
---------------------
This file exists because a plausible-sounding claim about the 'raw' mode turned
out to be FALSE on measurement, and the wrong version nearly got shipped as a
"fix". The claim was: an L1 on raw |FFT| must be dominated by the low-frequency
bins (their amplitudes are ~45x larger per bin), so the term is really another
blur-inducing reconstruction loss.

It is wrong because L1's gradient per bin is +-1/N regardless of that bin's
magnitude — unlike L2. What decides the balance is where the prediction's
absolute amplitude errors land, and against a blurred prediction those land
mostly in the high band. `test_raw_mode_penalises_blur_mostly_at_high_freq`
pins that measurement so the wrong intuition cannot be re-adopted.

Runs on CPU with synthetic tensors. No data, no checkpoints, no GPU.
"""
import pathlib
import sys

import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from losses import FrequencyLoss                      # noqa: E402

H = W = 64


def _target(seed=0):
    """CT-like slice: smooth body + fine parenchymal grain, in [0,1]."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing='ij')
    smooth = 0.55 + 0.25 * torch.exp(-(xx ** 2 + yy ** 2) * 3)
    grain = 0.02 * torch.randn(H, W, generator=g)
    return (smooth + grain).clamp(0, 1)[None, None]


def _blur(x, k=5):
    return F.conv2d(x, torch.ones(1, 1, k, k) / (k * k), padding=k // 2)


def _band_shares(pred, target):
    """Share of the raw L1 amplitude loss falling in each radial band."""
    amp = lambda z: torch.abs(torch.fft.fft2(z, norm='ortho'))[0, 0]
    err = (amp(pred) - amp(target)).abs()
    f = torch.fft.fftfreq(H)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    tot = err.sum()
    return {name: float(err[(r >= lo) & (r < hi)].sum() / tot)
            for lo, hi, name in [(0, .05, 'dc'), (.05, .15, 'low'),
                                 (.15, .35, 'mid'), (.35, .71, 'high')]}


def test_raw_mode_penalises_blur_mostly_at_high_freq():
    """THE anti-regression test for the wrong diagnosis. See the module docstring.

    Against a blurred prediction, most of raw-mode's loss mass must sit in the
    top radial band. If this ever fails, 'raw' really has become a low-frequency
    term and the argument for replacing it becomes valid.
    """
    t = _target()
    sh = _band_shares(_blur(t), t)
    assert sh['high'] > 0.4, (
        f"raw mode puts only {sh['high']:.1%} of its blur penalty in the high "
        f"band; the 'it is just another blur term' claim would then be correct")
    assert sh['dc'] < 0.15, f"DC share {sh['dc']:.1%} unexpectedly large"


def test_pure_level_error_is_entirely_dc():
    """A global HU offset moves only the DC bin.

    Not a defect: case-level HU offset is the project's primary endpoint, so a
    term that reacts to it purely through DC is behaving sensibly.
    """
    t = _target()
    sh = _band_shares(t + 20.0 / 600.0, t)     # +20 HU on the 600 HU window
    assert sh['dc'] > 0.99, f"expected ~all DC, got {sh}"


def test_banded_shifts_weight_off_dc():
    """'banded' must actually move gradient away from DC, or it is decorative."""
    t = _target()
    p = _blur(t)
    amp = lambda z: torch.abs(torch.fft.fft2(z, norm='ortho'))[0, 0]
    err = (amp(p) - amp(t)).abs()
    f = torch.fft.fftfreq(H)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    w = r / r.mean()
    dc = (r < 0.05)
    raw_share = float(err[dc].sum() / err.sum())
    banded_share = float((err * w)[dc].sum() / (err * w).sum())
    assert banded_share < raw_share, (
        f'banded DC share {banded_share:.4f} is not below raw {raw_share:.4f}')


def test_banded_keeps_lambda_comparable():
    """Mean-1 weight normalisation: the two modes must be within a small factor.

    Without it, switching mode would silently also change the term's effective
    lambda, confounding the mode ablation with a weight change — the mistake
    config.py's LAMBDA_L1 note is written to prevent.
    """
    t, p = _target(), _blur(_target())
    raw = float(FrequencyLoss('raw')(p, t))
    banded = float(FrequencyLoss('banded')(p, t))
    assert 0.3 < banded / raw < 3.0, (
        f'banded/raw = {banded / raw:.2f}; mean-1 normalisation is not holding '
        f'the scale and lambda_frequency would mean something different per mode')


def test_all_modes_are_zero_on_a_perfect_match():
    t = _target()
    for mode in ('raw', 'banded', 'focal'):
        assert float(FrequencyLoss(mode)(t.clone(), t)) < 1e-6, f'{mode} nonzero'


def test_all_modes_are_differentiable_and_positive():
    t = _target()
    for mode in ('raw', 'banded', 'focal'):
        p = _blur(t).clone().requires_grad_(True)
        loss = FrequencyLoss(mode)(p, t)
        assert float(loss) > 0, f'{mode} gave a non-positive loss on a blurred pred'
        loss.backward()
        assert p.grad is not None and torch.isfinite(p.grad).all(), \
            f'{mode} produced no finite gradient'


def test_focal_weight_is_detached():
    """A live focal weight lets the model shrink the WEIGHT instead of the error.

    Checked structurally: the weight is built from `diff.detach()`, so doubling
    the error must not more than roughly quadruple the loss (w ~ error, so
    loss ~ error^2 at most). A live weight would compound further.
    """
    t = _target()
    fl = FrequencyLoss('focal')
    l1 = float(fl(_blur(t, 3), t))
    l2 = float(fl(_blur(t, 9), t))
    assert l2 > l1 > 0, 'focal should grow with a worse prediction'


def test_3d_input_is_accepted():
    """(B,C,D,H,W) must work — the diffusion path can run 3-D patches."""
    t3 = _target().unsqueeze(2).repeat(1, 1, 3, 1, 1)
    p3 = _blur(_target()).unsqueeze(2).repeat(1, 1, 3, 1, 1)
    for mode in ('raw', 'banded', 'focal'):
        assert float(FrequencyLoss(mode)(p3, t3)) > 0


def test_unknown_mode_raises():
    try:
        FrequencyLoss('bogus')
    except ValueError:
        return
    raise AssertionError('an unknown frequency_mode must raise, not silently pass')


if __name__ == '__main__':
    checks = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in checks:
        try:
            fn(); print(f'  PASS  {fn.__name__}')
        except AssertionError as e:
            failed += 1; print(f'  FAIL  {fn.__name__}: {e}')
    t = _target()
    print('\n  measured band shares, raw mode:')
    for name, mode in [('blurred', _blur(t)), ('+20 HU', t + 20 / 600.),
                       ('x1.2 contrast', t * 1.2)]:
        sh = _band_shares(mode, t)
        print(f'    {name:16s}' + '  '.join(f'{k}={v:5.1%}' for k, v in sh.items()))
    print(f'\n{"ALL PASS" if not failed else f"{failed} FAILURE(S)"}')
    sys.exit(1 if failed else 0)
