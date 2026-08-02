"""
CPU-only smoke test for the ROI-guided phase feature/eval LOGIC.

Covers the pure-numpy parts that don't need xgboost / TotalSegmentator / real
CT data:
  1. features_from_mask computes correct per-organ aggregates (median/mean) and
     returns NaN for organs absent from the mask.
  2. load_organ_label_map validates an explicit map.
  3. PhaseEvaluator.score_case wiring: de-normalisation, same-mask feature
     comparison, and the classification-agreement flags — exercised with a fake
     ensemble (duck-typed predict_proba) so no xgboost is needed.

Usage:  python smoke_test_xgb.py
"""

import sys

import numpy as np

from organ_features import ORGANS, features_from_mask, load_organ_label_map
from phase_eval import PhaseEvaluator, aggregate


def check(cond, name, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  -- {detail}' if detail and not cond else ''))
    return bool(cond)


# Small explicit map: assign each organ a distinct label id 1..16.
ORGAN_MAP = {o: i + 1 for i, o in enumerate(ORGANS)}


class _FakeXGB:
    """Duck-typed stand-in for an xgboost model: predicts phase from the AORTA
    feature — high aorta HU → arterial(1), mid → venous(2), low → non-contrast(0).
    Enough structure to exercise the eval wiring deterministically."""
    def predict_proba(self, X):
        aorta = X[0][ORGANS.index('aorta')]
        aorta = 0.0 if np.isnan(aorta) else aorta
        if aorta > 250:      p = [0.05, 0.85, 0.10, 0.0]
        elif aorta > 90:     p = [0.10, 0.15, 0.75, 0.0]
        else:                p = [0.90, 0.05, 0.05, 0.0]
        return np.array([p])


def _build_volume(organ_hu: dict, shape=(8, 16, 16)):
    """Return (ct, mask): each organ occupies a distinct slab with a set HU."""
    ct = np.full(shape, -1000.0, np.float32)          # air background
    mask = np.zeros(shape, np.int32)
    z = 0
    for organ, hu in organ_hu.items():
        lid = ORGAN_MAP[organ]
        ct[z] = hu
        mask[z] = lid
        z += 1
    return ct, mask


def main():
    ok = True

    # 1. feature extraction correctness
    ct, mask = _build_volume({'liver': 110.0, 'aorta': 320.0, 'pancreas': 80.0})
    feats = features_from_mask(ct, mask, ORGAN_MAP, aggregation='median')
    ok &= check(abs(feats[ORGANS.index('liver')] - 110.0) < 1e-6, 'liver median HU correct')
    ok &= check(abs(feats[ORGANS.index('aorta')] - 320.0) < 1e-6, 'aorta median HU correct')
    ok &= check(np.isnan(feats[ORGANS.index('brain')]), 'absent organ (brain) → NaN')

    # 2. explicit organ map validation
    ok &= check(load_organ_label_map(ORGAN_MAP) == ORGAN_MAP, 'explicit organ map accepted')
    try:
        load_organ_label_map({'liver': 5})  # missing others
        ok &= check(False, 'incomplete organ map should raise')
    except KeyError:
        ok &= check(True, 'incomplete organ map raises KeyError')

    # 3. evaluator wiring with a fake ensemble
    ev = PhaseEvaluator(models=[{'fold': 0, 'model': _FakeXGB()}], organ_label_map=ORGAN_MAP)

    # Real CECT = arterial (bright aorta). Generated matches it well.
    real_ct, m = _build_volume({'liver': 110.0, 'aorta': 320.0, 'pancreas': 85.0})
    # Generated is in [0,1] over window [-200,400]; encode aorta≈300, liver≈115.
    def to_norm(hu): return (hu - (-200.0)) / (400.0 - (-200.0))
    gen_ct = np.full(real_ct.shape, to_norm(-1000.0), np.float32)
    gen_ct[m == ORGAN_MAP['liver']] = to_norm(115.0)
    gen_ct[m == ORGAN_MAP['aorta']] = to_norm(300.0)
    gen_ct[m == ORGAN_MAP['pancreas']] = to_norm(80.0)

    r = ev.score_case(gen_ct, real_ct, m, target_phase=1, hu_min=-200.0, hu_max=400.0)
    ok &= check(r['pred_real'] == 1, 'real arterial CECT classified arterial', f"pred_real={r['pred_real']}")
    ok &= check(r['pred_gen'] == 1, 'well-matched generated classified arterial', f"pred_gen={r['pred_gen']}")
    ok &= check(r['gen_matches_target'] and r['gen_matches_real'], 'agreement flags set')
    # aorta feature error: gen 300 (clipped ≤400 ok) vs real clipped 320 → ~20 HU
    ao_err = r['per_organ_abs_err_hu']['aorta']
    ok &= check(ao_err is not None and abs(ao_err - 20.0) < 5.0, 'aorta feature error ~20 HU',
                f'got {ao_err}')

    # A generated that FAILS to enhance the aorta (looks non-contrast) → mismatch.
    gen_bad = gen_ct.copy()
    gen_bad[m == ORGAN_MAP['aorta']] = to_norm(40.0)   # unenhanced
    r_bad = ev.score_case(gen_bad, real_ct, m, target_phase=1, hu_min=-200.0, hu_max=400.0)
    ok &= check(not r_bad['gen_matches_target'], 'unenhanced generated fails target-phase check',
                f"pred_gen={r_bad['pred_gen']}")
    ok &= check(r_bad['feature_l1_hu'] > r['feature_l1_hu'], 'worse image → larger feature L1')

    # 4. aggregation
    agg = aggregate([r, r_bad])
    ok &= check(agg['n_cases'] == 2 and 0.0 <= agg['gen_phase_accuracy_vs_target'] <= 1.0,
                'aggregate summary well-formed')

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
