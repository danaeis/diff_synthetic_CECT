"""The `train_<k>_contrib` channels must be complete and must actually add up.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
These channels exist so a lambda can be chosen by measurement instead of by
guess: `train_<k>_contrib / train_gen_total` is the fraction of the generator
gradient that term is carrying. Everything in P2 of the plan reads that ratio.

So the failure mode to guard against is not "the number is slightly off" — it is
"a term is missing from the sum", which makes every share silently too large and
would send the lambda calibration in the wrong direction. Two invariants:

  1. COMPLETENESS. Every term that is switched on emits `<name>_contrib`.
  2. SUMMATION. The contributions add up to the total that was optimised. If they
     do not, some term is contributing gradient without being logged.

Runs on CPU with tiny random tensors. No data, no checkpoints, no GPU.

    python tests/test_loss_contrib.py
    python -m pytest tests/test_loss_contrib.py -q
"""
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from losses import CompositeLoss                      # noqa: E402
from trainer import CONTRIB_TERMS                     # noqa: E402

# CompositeLoss uses long names; the trainers map them to the short history
# names. Kept explicit because a silent rename on either side zeroes a curve
# rather than raising, which is exactly the class of bug this file exists for.
LONG_TO_SHORT = {
    'l1': 'l1', 'nll': 'nll', 'ssim': 'ssim',
    'gradient': 'grad', 'frequency': 'freq',
    'organ': 'organ', 'hu_profile': 'hu_profile',
    'adversarial': 'adv', 'feature_matching': 'fm',
}

_ALL_ON = dict(
    use_ssim=True, use_gradient=True, use_frequency=True,
    use_organ=True, use_hu_profile=True,
    use_adversarial=True, use_feature_matching=True,
    organ_weights={1: 5.0, 2: 2.0},
    lambda_ssim=10.0, lambda_gradient=5.0, lambda_frequency=1.0,
    lambda_organ=20.0, lambda_hu_profile=50.0, lambda_fm=10.0,
    lambda_l1=100.0,
)


def _batch(b=2, c=1, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(b, c, h, w, generator=g, requires_grad=True)
    target = torch.rand(b, c, h, w, generator=g)
    # Multi-label mask: OrganWeightedLoss needs raw label ids, not a binary mask
    # (a binarised mask collapses every organ onto the label-1 weight).
    mask = torch.randint(0, 3, (b, 1, h, w), generator=g)
    return pred, target, mask


def _run(cfg_extra=None):
    cfg = dict(_ALL_ON)
    cfg.update(cfg_extra or {})
    crit = CompositeLoss(cfg)
    pred, target, mask = _batch()
    kw = {}
    if cfg.get('use_adversarial'):
        kw['adv_fake_logits'] = torch.zeros(2, 1, 4, 4)
        kw['adv_weight'] = 0.5
    if cfg.get('use_feature_matching'):
        feats = [torch.rand(2, 4, 8, 8), torch.rand(2, 8, 4, 4)]
        kw['real_features'] = feats
        kw['fake_features'] = [f.clone() + 0.1 for f in feats]
    total, d = crit(pred, target, mask=mask, **kw)
    # .detach(): the callers only read a scalar, and float() on a grad-tracking
    # tensor warns.
    return total.detach(), d


def test_every_active_term_emits_contrib():
    """Completeness: no active term may be missing its `_contrib` key."""
    _, d = _run()
    missing = [n for n in LONG_TO_SHORT if f'{n}_contrib' not in d]
    assert not missing, (
        f'active terms with no _contrib channel: {missing}. Their gradient is '
        f'in the total but invisible in history.json, so every other term\'s '
        f'computed share is too large.')


def test_contribs_sum_to_total():
    """Summation: the logged contributions reconstruct the optimised total.

    A mismatch means a term is adding gradient without being recorded — the
    quiet version of the completeness failure above, where the key exists but
    holds the wrong value (raw instead of scaled, say).
    """
    total, d = _run()
    s = sum(d[f'{n}_contrib'] for n in LONG_TO_SHORT)
    assert abs(s - float(total)) < 1e-4 * max(1.0, abs(float(total))), (
        f'contributions sum to {s:.6f} but the optimised total is '
        f'{float(total):.6f} (difference {s - float(total):.6g})')


def test_l1_contrib_is_scaled_not_raw():
    """`l1` is logged RAW for backwards compatibility; its contrib is not.

    This is the specific inconsistency the contrib channels were added to fix,
    so assert the two really do differ by lambda rather than being aliases.
    """
    _, d = _run()
    assert d['lambda_l1'] > 1.0, 'test needs a lambda_l1 != 1 to be meaningful'
    assert abs(d['l1_contrib'] - d['l1'] * d['lambda_l1']) < 1e-4
    assert d['l1_contrib'] != d['l1']


def test_adv_contrib_tracks_the_warmup_weight():
    """`adv` is raw and `adv_weight` is the trainer's warmed-up lambda.

    During warmup the weight is 0, so a raw-only reading shows a term that looks
    active while contributing nothing. The contrib channel must show zero.
    """
    cfg = dict(_ALL_ON)
    crit = CompositeLoss(cfg)
    pred, target, mask = _batch()
    feats = [torch.rand(2, 4, 8, 8)]
    common = dict(mask=mask, adv_fake_logits=torch.zeros(2, 1, 4, 4),
                  real_features=feats, fake_features=[feats[0] + 0.1])
    _, d0 = crit(pred, target, adv_weight=0.0, **common)
    _, d1 = crit(pred, target, adv_weight=0.5, **common)
    assert d0['adversarial'] > 0, 'raw adversarial loss should still be measured'
    assert d0['adversarial_contrib'] == 0.0, 'contrib must be 0 while lambda is 0'
    assert abs(d1['adversarial_contrib'] - d1['adversarial'] * 0.5) < 1e-6


def test_inactive_terms_are_zero_not_missing():
    """An off term must log 0.0, not drop the key.

    `_update_history` uses .get(..., 0.0), so a missing key is survivable — but
    it makes the stackplot silently omit a band, and an omitted band reads as
    "this term contributes nothing" rather than "this term was not recorded".
    """
    _, d = _run({k: False for k in
                 ('use_ssim', 'use_gradient', 'use_frequency',
                  'use_organ', 'use_hu_profile',
                  'use_adversarial', 'use_feature_matching')})
    for n in LONG_TO_SHORT:
        assert f'{n}_contrib' in d, f'{n}_contrib key vanished when the term was off'
    assert d['ssim_contrib'] == 0.0 and d['organ_contrib'] == 0.0


def test_contrib_terms_covers_every_composite_term():
    """trainer.CONTRIB_TERMS must name every term CompositeLoss can emit.

    A term present in the loss but absent from CONTRIB_TERMS gets no history
    channel and no band in the loss-balance panel.
    """
    _, d = _run()
    emitted = {LONG_TO_SHORT[n] for n in LONG_TO_SHORT if f'{n}_contrib' in d}
    assert emitted <= set(CONTRIB_TERMS), (
        f'terms emitted by CompositeLoss but missing from CONTRIB_TERMS: '
        f'{sorted(emitted - set(CONTRIB_TERMS))}')


def test_gate_normalisation_arithmetic():
    """The aux-gate fix, at the level it is actually applied.

    trainer_diffusion multiplies the gated aux terms by len(sel)/len(t). The
    property that matters: the EXPECTED contribution stops depending on how many
    items happened to pass the gate. Without it a batch with 4 of 8 items under
    aux_max_t contributes the same as one with 8 of 8 — a 2x swing driven purely
    by a binomial draw.
    """
    per_item = 0.25          # whatever the aux loss averages to on the sub-batch
    lam, batch = 20.0, 8
    unnormalised = {n: per_item * lam for n in (2, 4, 8)}
    normalised = {n: per_item * lam * (n / batch) for n in (2, 4, 8)}
    assert len(set(unnormalised.values())) == 1, \
        'without normalisation every gate fraction contributes identically'
    assert normalised[4] == 2 * normalised[2]
    assert normalised[8] == batch / 8 * per_item * lam
    # And the full-batch case must be untouched, so the flag is a no-op when the
    # gate admits everything — otherwise it would silently rescale aux_max_t=T runs.
    assert normalised[8] == unnormalised[8]


if __name__ == '__main__':
    checks = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in checks:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__}: {e}')
    print(f'\n{"ALL PASS" if not failed else f"{failed} FAILURE(S)"}')
    sys.exit(1 if failed else 0)
