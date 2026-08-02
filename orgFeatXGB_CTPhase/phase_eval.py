"""
Phase-fidelity evaluation of a synthesised CECT against the real CECT.

This is the "3a" metric: use the trained ROI-guided XGBoost phase model to score
how well a generator reproduces the contrast-phase signal, in two complementary
ways per case:

  A) FEATURE level — extract the 16 per-organ HU features from BOTH the generated
     and the real CECT using the SAME co-registered organ mask, and compare them
     (per-organ absolute HU error + mean). This directly measures whether the
     generator got the organ/vessel enhancement levels right — the physical
     substance of contrast phase — independent of the classifier.

  B) CLASSIFICATION level — run the XGBoost ensemble on the generated features and
     ask: does it classify as the intended TARGET phase? does it agree with the
     real CECT's own prediction? with what confidence? This is the published
     phase-classification model used as a synthesis quality metric.

Normalisation: the generator emits images in [0,1] over a clipped HU window
(config HU_MIN/HU_MAX). Features must be in HU for the model, so generated volumes
are de-normalised: hu = norm * (hu_max - hu_min) + hu_min. NOTE this clips organ
HU to [hu_min, hu_max]; if hu_max is below bright arterial-aorta HU (~350-450),
that enhancement is truncated — fine for venous/liver targets, a known limitation
for arterial. Pass --no_denorm if your generated volumes are already in HU.

For a fair per-organ FEATURE comparison, the real CECT is clipped to the same
window by default (so gen and real live in the same domain); the CLASSIFICATION
of the real CECT uses its true (unclipped) HU as a ceiling/sanity check.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from organ_features import ORGANS, features_from_mask, load_organ_label_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

PHASE_NAMES = {0: 'non-contrast', 1: 'arterial', 2: 'venous', 3: 'delayed'}
PHASE_IDS = {v: k for k, v in PHASE_NAMES.items()}


class PhaseEvaluator:
    def __init__(self, weights_path: Optional[str] = None,
                 organ_label_map: Optional[Dict[str, int]] = None,
                 models: Optional[list] = None):
        # `models` lets callers/tests inject an already-loaded ensemble; otherwise
        # load from disk. Either way it's a list of {"fold","model"} dicts.
        if models is not None:
            self.models = models
        else:
            with open(weights_path, 'rb') as f:
                self.models = pickle.load(f)
        if not isinstance(self.models, list) or 'model' not in self.models[0]:
            raise ValueError("unexpected weights format (expected list of {'fold','model'})")
        # An explicit organ map is required when TotalSegmentator isn't importable.
        self.organ_map = organ_label_map or load_organ_label_map()

    def predict(self, features: np.ndarray):
        """Ensemble mean predict_proba across folds → (probs (C,), pred_id)."""
        probs = np.mean([m['model'].predict_proba([features])[0] for m in self.models], axis=0)
        return probs, int(np.argmax(probs))

    @staticmethod
    def _denorm(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
        return vol * (hu_max - hu_min) + hu_min

    def score_case(
        self, gen_vol: np.ndarray, real_vol: np.ndarray, mask_vol: np.ndarray,
        target_phase: int,
        hu_min: Optional[float] = -200.0, hu_max: Optional[float] = 400.0,
        clip_real_to_window: bool = True,
        gen_in_hu: bool = False,
    ) -> Dict:
        # gen_in_hu=True: the generated volume is ALREADY in HU (e.g. saved by
        # infer_volume.py) — don't de-normalise it, but still use the window to
        # clip the real CECT so the per-organ feature comparison stays fair.
        window_known = hu_min is not None and hu_max is not None
        gen_hu = (gen_vol.astype(np.float64) if (gen_in_hu or not window_known)
                  else self._denorm(gen_vol, hu_min, hu_max))
        real_hu = real_vol.astype(np.float64)

        # --- A) feature comparison (same domain for both) ---
        real_for_feat = np.clip(real_hu, hu_min, hu_max) if (window_known and clip_real_to_window) else real_hu
        f_gen = features_from_mask(gen_hu, mask_vol, self.organ_map)
        f_real_win = features_from_mask(real_for_feat, mask_vol, self.organ_map)
        per_organ_err = np.abs(f_gen - f_real_win)
        feature_l1 = float(np.nanmean(per_organ_err))

        # --- B) classification ---
        probs_gen, pred_gen = self.predict(f_gen)
        f_real_true = features_from_mask(real_hu, mask_vol, self.organ_map)   # unclipped ceiling
        probs_real, pred_real = self.predict(f_real_true)

        return {
            'target_phase': int(target_phase),
            'pred_gen': pred_gen, 'pred_real': pred_real,
            'gen_matches_target': bool(pred_gen == target_phase),
            'gen_matches_real': bool(pred_gen == pred_real),
            'real_matches_target': bool(pred_real == target_phase),
            'gen_target_prob': float(probs_gen[target_phase]),
            'real_target_prob': float(probs_real[target_phase]),
            'feature_l1_hu': feature_l1,
            'per_organ_abs_err_hu': {o: (None if np.isnan(e) else float(e))
                                     for o, e in zip(ORGANS, per_organ_err)},
            'gen_probs': probs_gen.tolist(), 'real_probs': probs_real.tolist(),
        }

    def score_case_paths(self, gen_path, real_path, mask_path, target_phase, **kw) -> Dict:
        import nibabel as nib
        gen = nib.load(gen_path).get_fdata().astype(np.float32)
        real = nib.load(real_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata()
        if not (gen.shape == real.shape == mask.shape):
            raise ValueError(f"shape mismatch gen{gen.shape}/real{real.shape}/mask{mask.shape} "
                             f"for {Path(gen_path).name} — all must be on the same grid")
        return self.score_case(gen, real, mask, target_phase, **kw)


def aggregate(case_results: List[Dict]) -> Dict:
    """Dataset-level summary: the headline synthesis phase-fidelity numbers."""
    n = len(case_results)
    if n == 0:
        return {}
    def frac(key): return float(np.mean([c[key] for c in case_results]))
    per_organ = {o: [] for o in ORGANS}
    for c in case_results:
        for o, e in c['per_organ_abs_err_hu'].items():
            if e is not None:
                per_organ[o].append(e)
    return {
        'n_cases': n,
        'gen_phase_accuracy_vs_target': frac('gen_matches_target'),
        'gen_agreement_with_real': frac('gen_matches_real'),
        'real_phase_accuracy_vs_target': frac('real_matches_target'),   # ceiling sanity check
        'mean_gen_target_prob': frac('gen_target_prob'),
        'mean_feature_l1_hu': float(np.mean([c['feature_l1_hu'] for c in case_results])),
        'per_organ_mean_abs_err_hu': {o: (float(np.mean(v)) if v else None)
                                      for o, v in per_organ.items()},
    }


def main():
    ap = argparse.ArgumentParser(description='Phase-fidelity eval of synthesised vs real CECT')
    ap.add_argument('--weights', required=True, help='trained XGBoost ensemble .pkl')
    ap.add_argument('--manifest', required=True,
                    help='CSV with columns: gen_path,real_path,mask_path,target_phase '
                         '(target_phase as id 0-3 or name)')
    ap.add_argument('--out_json', default='phase_eval_report.json')
    ap.add_argument('--hu_min', type=float, default=-200.0)
    ap.add_argument('--hu_max', type=float, default=400.0)
    ap.add_argument('--no_denorm', action='store_true',
                    help='drop the HU window entirely (gen already HU AND skip real clipping)')
    ap.add_argument('--gen_in_hu', action='store_true',
                    help="generated volumes are already in HU (e.g. from infer_volume.py) — "
                         "keep the window to clip the real CECT for a fair feature comparison")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.manifest)
    ev = PhaseEvaluator(args.weights)

    hu_min = None if args.no_denorm else args.hu_min
    hu_max = None if args.no_denorm else args.hu_max

    results = []
    for _, row in df.iterrows():
        tp = row['target_phase']
        tp = PHASE_IDS[str(tp).lower()] if not str(tp).isdigit() else int(tp)
        try:
            r = ev.score_case_paths(row['gen_path'], row['real_path'], row['mask_path'], tp,
                                    hu_min=hu_min, hu_max=hu_max, gen_in_hu=args.gen_in_hu)
            r['gen_path'] = row['gen_path']
            results.append(r)
        except Exception as e:
            log.warning(f"skip {row.get('gen_path')}: {e}")

    summary = aggregate(results)
    log.info("=== Phase-fidelity summary ===")
    for k, v in summary.items():
        if not isinstance(v, dict):
            log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    Path(args.out_json).write_text(json.dumps({'summary': summary, 'cases': results}, indent=2))
    log.info(f"Saved → {args.out_json}")


if __name__ == '__main__':
    main()


__all__ = ['PhaseEvaluator', 'aggregate']
