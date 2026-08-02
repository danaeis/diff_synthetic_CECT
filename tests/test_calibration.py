"""
The calibration metrics, checked against closed forms and against distributions
whose true calibration is known by construction.

These matter more than usual: the headline claim of this repo is a NUMBER these
functions produce ("variance ratio moved from 0.176 toward 1.0"), so a
sign error or an off-by-one in a quantile would not look like a bug, it would
look like a result.

Plain functions, no pytest dependency (the same style as tests/test_metrics.py —
pytest is not installed on the training host):
    python tests/test_calibration.py
"""

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import norm

from metrics import (achieved_level, calibration_metrics, crps_ensemble,
                     interval_coverage, min_samples_for_level, pit_values,
                     variance_ratio)

RNG = np.random.default_rng(20260801)


def _crps_gaussian(mu, sd, y):
    """Closed-form CRPS of N(mu, sd) against a scalar observation.

        CRPS = sd * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sd
    """
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def test_crps_matches_closed_form():
    """The energy form must reproduce the Gaussian closed form.

    The last case is the one that matters: mu/sd/y at the scale of a real aortic
    HU level and its measured 24.4 HU across-case spread.
    """
    for mu, sd, y in [(0.0, 1.0, 0.0), (0.3, 0.8, 1.1),
                      (-2.0, 3.0, 4.0), (100.0, 24.4, 137.0)]:
        emp = crps_ensemble(RNG.normal(mu, sd, 200_000), y)
        exact = _crps_gaussian(mu, sd, y)
        assert abs(emp - exact) < 0.01 * max(1.0, sd), f'{emp:.4f} vs {exact:.4f}'


def test_crps_of_point_predictor_is_mae():
    """A zero-spread ensemble must score CRPS = |mu - y|.

    This is what makes CRPS directly comparable to the deterministic model's MAE:
    a probabilistic model that cannot beat it has not earned its sampling cost.
    """
    y = 3.0
    for mu in (3.0, 0.0, 7.5):
        assert abs(crps_ensemble(np.full(64, mu), y) - abs(mu - y)) < 1e-9


def test_crps_penalises_both_failure_modes():
    """Over-confident AND over-dispersed ensembles must both score worse than a
    correctly-spread one centred on the same place."""
    y = 0.0
    good = crps_ensemble(RNG.normal(0, 1, 40_000), y)
    narrow = crps_ensemble(RNG.normal(0, 0.05, 40_000), y)
    wide = crps_ensemble(RNG.normal(0, 8.0, 40_000), y)
    # A too-narrow ensemble centred exactly on the truth is NOT penalised — it is
    # right. The failure shows up when the truth is off-centre, which is the case
    # the deterministic model is actually in.
    y_off = 1.5
    good_off = crps_ensemble(RNG.normal(0, 1, 40_000), y_off)
    narrow_off = crps_ensemble(RNG.normal(0, 0.05, 40_000), y_off)
    assert wide > good, 'over-dispersion must cost'
    assert narrow_off > good_off, 'over-confidence must cost when the truth is off-centre'
    assert narrow < good, 'a spike on the exact truth is genuinely better'


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_of_calibrated_ensemble():
    """Truths drawn from the same distribution as the ensemble must land inside
    the band at the rate `achieved_level` predicts — at EVERY ensemble size.

    This is the test that caught the original implementation. Interpolated
    empirical quantiles are biased downward for small n (0.726 at n=8, 0.813 at
    n=20, still only 0.891 at n=200), so a perfectly calibrated model would have
    read as over-confident and the bias would have been larger than any real
    miscalibration this project is likely to measure.
    """
    for n in (19, 20, 25, 50, 200):
        cov = np.mean([interval_coverage(RNG.normal(0, 1, n), t)
                       for t in RNG.normal(0, 1, 4000)])
        tgt = achieved_level(n, 0.90)
        assert abs(cov - tgt) < 0.02, f'n={n}: measured {cov:.3f} vs target {tgt:.3f}'


def test_coverage_detects_the_measured_underdispersion():
    """An ensemble at the deterministic model's var ratio 0.176 (sd ratio 0.42)
    must show badly low coverage — this is the failure the whole repo exists to
    measure, so the metric has to see it."""
    cov = np.mean([interval_coverage(RNG.normal(0, np.sqrt(0.176), 200), t)
                   for t in RNG.normal(0, 1, 3000)])
    assert cov < 0.60, cov


def test_coverage_needs_enough_samples():
    """Below `min_samples_for_level` the interval is UNREACHABLE, not just noisy,
    and the function must say NaN rather than return a number that reads as
    miscalibration but is a sample-size artifact.

    n samples make n+1 gaps and the widest expressible interval [min, max] holds
    n-1 of them, so the ceiling is (n-1)/(n+1): 19 samples for a 90% interval,
    3 for a 50% one. The planned N=8 exploratory run therefore CANNOT produce a
    90% coverage number, which is why calibration_eval.py reports both levels.
    """
    assert min_samples_for_level(0.90) == 19
    assert min_samples_for_level(0.50) == 3
    assert np.isnan(interval_coverage(RNG.normal(0, 1, 8), 0.0, 0.90))
    assert np.isnan(interval_coverage(RNG.normal(0, 1, 18), 0.0, 0.90))
    assert not np.isnan(interval_coverage(RNG.normal(0, 1, 19), 0.0, 0.90))
    # The 50% interval is what an N=8 run can still report.
    assert not np.isnan(interval_coverage(RNG.normal(0, 1, 8), 0.0, 0.50))
    assert np.isnan(interval_coverage(RNG.normal(0, 1, 2), 0.0, 0.50))


def test_coverage_scales_with_level():
    ts = RNG.normal(0, 1, 3000)
    c50 = np.mean([interval_coverage(RNG.normal(0, 1, 200), t, 0.50) for t in ts])
    c90 = np.mean([interval_coverage(RNG.normal(0, 1, 200), t, 0.90) for t in ts])
    assert c50 < c90
    assert abs(c50 - 0.50) < 0.03 and abs(c90 - 0.90) < 0.03


# ---------------------------------------------------------------------------
# Variance ratio — the headline
# ---------------------------------------------------------------------------

def test_variance_ratio_matched_and_underdispersed():
    """1.0 for matched spread; 0.176 for the measured deterministic failure.

    The second number is the one recorded in analysis/BASELINE_REFERENCE.md, so
    this pins the metric against the value the whole comparison is anchored to.
    """
    # Aortic level sd 24.4 HU across cases, from the real data.
    truths = RNG.normal(120, 24.4, 2000)
    assert abs(variance_ratio(truths + RNG.normal(0, 0.5, 2000), truths) - 1.0) < 0.05
    assert abs(variance_ratio(np.sqrt(0.176) * truths, truths) - 0.176) < 0.02


def test_variance_ratio_predictive_reading():
    """With pred_vars given it reads the CLAIMED uncertainty, not the output spread.

    The two readings can disagree — a model can spread its outputs correctly and
    still misstate its confidence — which is why both are reported.
    """
    truths = RNG.normal(0, 10.0, 2000)
    means = truths + RNG.normal(0, 1, 2000)      # outputs spread correctly
    tiny = np.full(2000, 0.01)                   # but claims near-zero uncertainty
    assert abs(variance_ratio(means, truths) - 1.0) < 0.1
    assert variance_ratio(means, truths, pred_vars=tiny) < 0.01


def test_variance_ratio_ignores_nan_cases():
    """An organ absent in some cases must not poison the ratio."""
    truths = RNG.normal(0, 5.0, 200)
    means = truths.copy()
    truths[::10] = np.nan
    assert abs(variance_ratio(means, truths) - 1.0) < 0.15
    assert np.isnan(variance_ratio(np.full(2, 1.0), np.full(2, 1.0)))


# ---------------------------------------------------------------------------
# PIT
# ---------------------------------------------------------------------------

def test_pit_uniform_when_calibrated():
    p = np.array([pit_values(RNG.normal(0, 1, 100), t) for t in RNG.normal(0, 1, 4000)])
    assert abs(p.mean() - 0.5) < 0.02
    # A flat histogram: every decile should hold ~10% of the mass.
    hist, _ = np.histogram(p, bins=10, range=(0, 1))
    assert hist.max() / hist.min() < 1.6, hist


def test_pit_u_shaped_when_overconfident():
    """Over-confident → the truth falls outside the ensemble → mass at 0 and 1."""
    p = np.array([pit_values(RNG.normal(0, 0.2, 100), t) for t in RNG.normal(0, 1, 4000)])
    hist, _ = np.histogram(p, bins=10, range=(0, 1))
    edges = hist[0] + hist[-1]
    middle = hist[4] + hist[5]
    assert edges > 3 * middle, f'expected U shape, got {hist}'


def test_pit_ties_do_not_pile_at_the_edges():
    """A discrete ensemble with repeated values must not read as over-confident
    purely because of ties."""
    s = np.zeros(50)
    assert abs(float(pit_values(s, 0.0)) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def test_calibration_metrics_bundle():
    s = RNG.normal(5.0, 2.0, 500)
    d = calibration_metrics(s, 5.4)
    assert set(d) == {'crps', 'coverage', 'coverage_target', 'coverage50',
                      'coverage50_target', 'ens_sd', 'abs_err'}
    assert abs(d['coverage_target'] - 0.90) < 0.02
    assert abs(d['coverage50_target'] - 0.50) < 0.03
    assert abs(d['ens_sd'] - 2.0) < 0.2
    assert abs(d['abs_err'] - 0.4) < 0.2
    assert d['coverage'] in (0.0, 1.0)          # one truth: inside or outside
    assert abs(d['crps'] - _crps_gaussian(5.0, 2.0, 5.4)) < 0.05


def test_field_shaped_inputs():
    """The functions must work on (n_samples, ...) fields, not just scalars —
    that is the per-voxel path."""
    # Calibrated: the truth is one more draw from the same distribution as the
    # ensemble, NOT the centre it was built around.
    mu = RNG.normal(0, 5, (16, 16))
    samples = mu[None] + RNG.normal(0, 1, (64, 16, 16))
    truth = mu + RNG.normal(0, 1, (16, 16))
    assert np.isfinite(crps_ensemble(samples, truth))
    cov = interval_coverage(samples, truth)
    assert abs(cov - achieved_level(64, 0.90)) < 0.06, cov
    assert pit_values(samples, truth).shape == (16, 16)

    # An ensemble centred exactly on the truth covers it essentially always —
    # the degenerate case, kept so the check above cannot pass by accident.
    centred = truth[None] + RNG.normal(0, 1, (64, 16, 16))
    assert interval_coverage(centred, truth) > 0.95


if __name__ == '__main__':
    import traceback
    fails = []
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    print('=' * 70); print('CALIBRATION METRICS'); print('=' * 70)
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            fails.append(name)
    print('=' * 70)
    if fails:
        print(f"FAILED ({len(fails)}): {', '.join(fails)}")
        sys.exit(1)
    print('ALL PASS')
