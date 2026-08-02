#!/usr/bin/env python
"""
Does the model TRACK each case's contrast level, or predict the population mean?

Motivation. Four measurements have ruled out overfitting, registration error,
voxel-scale inconsistency and model capacity as the cause of the ~8 HU per-organ
floor (../synthetic_CECT/PROJECT_PLAN.md 1.3-1.5, 1.9). The remaining hypothesis is aleatoric:
absolute enhancement depends on injection dose, bolus timing and cardiac output,
none of which is visible in a non-contrast scan. A regressor trained with L1
cannot invent that information, so it should fall back on the conditional
median — the average enhancement over the training set.

That hypothesis makes three sharp, falsifiable predictions, and this script tests
all three. It needs NO acquisition metadata (there is none: vindr_nifti_metadata
carries only geometry), because the bolus position is readable from the real CECT
itself.

  1. LEVEL-TRACKING SLOPE. Regress the generated per-organ median HU on the real
     one, across cases:
         beta = 1  -> the model tracks each case's true level (hypothesis FALSE)
         beta = 0  -> the model emits the same level regardless of case, i.e.
                      pure mean-prediction (hypothesis TRUE)
     Anything in between says how much of the case-to-case level variation the
     model actually recovers.

  2. UNDER-DISPERSION. var(generated) / var(real) across cases. A mean-predictor
     collapses the spread; the ratio goes to 0. This is the textbook signature of
     L1/L2 regression under irreducible uncertainty.

  3. ERROR TRACKS ENHANCEMENT. If the model predicts the average, it must
     UNDERSHOOT strongly-enhancing cases and OVERSHOOT weak ones, so the signed
     error correlates negatively with the true level (slope = beta - 1).

Medians, not means, throughout: the XGBoost phase model reads per-organ median HU
and nothing else, so that is the quantity the whole pipeline is judged on.

Aorta and portal vein double as a bolus-position surrogate — aorta hot / PV cool
is early arterial, the reverse is late venous — so their ratio locates each case
on the enhancement curve without any DICOM tag.

Usage:
    python scripts/audit_enhancement.py \
      --manifest ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
      --out analysis/enhancement.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    raise SystemExit("audit_enhancement.py needs nibabel (pip install nibabel)")

# Organs worth reading individually. The first four carry the contrast signal;
# liver/spleen are parenchymal references; bowel is deliberately included as a
# NEGATIVE control (it is zero-weighted in training and barely enhances, so it
# should show a different pattern from the vessels if the effect is real).
KEY_ORGANS = ['aorta', 'portal_vein_and_splenic_vein', 'inferior_vena_cava',
              'heart', 'liver', 'pancreas', 'gallbladder', 'colon']

MIN_VOXELS = 64          # below this an organ median is too noisy to regress on


def _load(p: str) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def _fit(x: np.ndarray, y: np.ndarray):
    """Least-squares slope/intercept/r for y ~ x, with n and r^2."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.std(x) == 0:
        return dict(n=n, slope=float('nan'), intercept=float('nan'),
                    r=float('nan'), r2=float('nan'))
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(n=n, slope=float(slope), intercept=float(intercept),
                r=r, r2=r * r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--label_map', type=Path,
                    default=Path(__file__).resolve().parent.parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--report', type=Path, default=None,
                    help='phase_eval_report.json — adds per-case featHU correlations')
    ap.add_argument('--organs', nargs='*', default=KEY_ORGANS)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    if not a.label_map.exists():
        raise SystemExit(f"label map not found: {a.label_map}")
    name_to_id = json.loads(a.label_map.read_text())
    organs = [o for o in a.organs if o in name_to_id]
    missing = [o for o in a.organs if o not in name_to_id]
    if missing:
        print(f"note: not in label map, skipped: {missing}")

    rows = list(csv.DictReader(open(a.manifest)))
    print(f"Reading {len(rows)} cases from {a.manifest.parent.parent.name}\n")

    # per-organ, per-case medians
    real_hu: Dict[str, List[float]] = {o: [] for o in organs}
    gen_hu: Dict[str, List[float]] = {o: [] for o in organs}
    case_ids: List[str] = []

    for r in rows:
        try:
            gen = _load(r['gen_path'])
            real = _load(r['real_path'])
            mask = _load(r['mask_path'])
        except Exception as e:                               # noqa: BLE001
            print(f"  skip {Path(r['gen_path']).name}: {e}")
            continue
        if not (gen.shape == real.shape == mask.shape):
            print(f"  skip {Path(r['gen_path']).name}: shape mismatch")
            continue
        case_ids.append(Path(r['gen_path']).name.replace('_syn.nii.gz', ''))
        for o in organs:
            sel = mask == name_to_id[o]
            if sel.sum() < MIN_VOXELS:
                real_hu[o].append(np.nan); gen_hu[o].append(np.nan)
                continue
            real_hu[o].append(float(np.median(real[sel])))
            gen_hu[o].append(float(np.median(gen[sel])))

    if not case_ids:
        raise SystemExit("no readable cases — is this the training host?")
    n = len(case_ids)
    R = {o: np.array(real_hu[o]) for o in organs}
    G = {o: np.array(gen_hu[o]) for o in organs}

    # ── 1 + 2 + 3 ────────────────────────────────────────────────────────────
    print("=" * 78)
    print("LEVEL TRACKING — generated organ median HU regressed on the real one")
    print("=" * 78)
    print("  beta=1 tracks the case's true level | beta=0 emits a constant\n")
    print(f"  {'organ':<30}{'n':>4}{'beta':>8}{'r2':>7}"
          f"{'real sd':>9}{'gen sd':>8}{'var ratio':>11}{'bias':>8}")

    out = {'n_cases': n, 'case_ids': case_ids, 'organs': {}}
    for o in organs:
        f = _fit(R[o], G[o])
        ok = np.isfinite(R[o]) & np.isfinite(G[o])
        sd_r = float(np.std(R[o][ok])) if ok.sum() > 1 else float('nan')
        sd_g = float(np.std(G[o][ok])) if ok.sum() > 1 else float('nan')
        vr = (sd_g ** 2 / sd_r ** 2) if sd_r > 0 else float('nan')
        bias = float(np.mean(G[o][ok] - R[o][ok])) if ok.sum() else float('nan')
        err_fit = _fit(R[o], G[o] - R[o])
        out['organs'][o] = {
            'beta': f['slope'], 'r2': f['r2'], 'n': f['n'],
            'real_sd': sd_r, 'gen_sd': sd_g, 'var_ratio': vr, 'bias': bias,
            'err_slope': err_fit['slope'], 'err_r': err_fit['r'],
            'real_median_hu': float(np.nanmedian(R[o])),
            'gen_median_hu': float(np.nanmedian(G[o])),
        }
        print(f"  {o:<30}{f['n']:>4}{f['slope']:>8.3f}{f['r2']:>7.2f}"
              f"{sd_r:>9.1f}{sd_g:>8.1f}{vr:>11.2f}{bias:>+8.1f}")

    # ── bolus-position surrogate ─────────────────────────────────────────────
    ao, pv = 'aorta', 'portal_vein_and_splenic_vein'
    if ao in organs and pv in organs:
        ok = np.isfinite(R[ao]) & np.isfinite(R[pv])
        ratio = np.full(n, np.nan)
        ratio[ok] = R[ao][ok] / np.maximum(R[pv][ok], 1.0)
        out['bolus_ratio_real'] = ratio.tolist()
        print(f"\n  bolus surrogate  aorta/PV median HU ratio: "
              f"median {np.nanmedian(ratio):.2f}, "
              f"range {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}")
        print(f"  real aorta HU spans {np.nanmin(R[ao]):.0f}-{np.nanmax(R[ao]):.0f} "
              f"(sd {np.nanstd(R[ao]):.0f}) across the {n} cases")

    # ── featHU correlations, if a report was given ───────────────────────────
    if a.report and a.report.exists():
        rep = json.loads(a.report.read_text())
        per = rep.get('per_case') or rep.get('cases') or []
        feat = np.array([c.get('feature_l1_hu', np.nan) for c in per], dtype=float)
        if len(feat) == n:
            print(f"\n{'=' * 78}\nPER-CASE ERROR vs ENHANCEMENT LEVEL\n{'=' * 78}")
            for o in organs:
                f = _fit(R[o], feat)
                flag = '  <-- significant' if abs(f['r']) > 0.44 else ''
                print(f"  featHU vs real {o:<30} r={f['r']:+.3f}{flag}")
            out['feat_vs_level'] = {o: _fit(R[o], feat) for o in organs}
        else:
            print(f"\nnote: report has {len(feat)} cases, manifest {n} — skipping "
                  f"featHU correlation")

    # ── verdict ──────────────────────────────────────────────────────────────
    betas = [out['organs'][o]['beta'] for o in organs
             if np.isfinite(out['organs'][o]['beta'])]
    vrs = [out['organs'][o]['var_ratio'] for o in organs
           if np.isfinite(out['organs'][o]['var_ratio'])]
    contrast = [o for o in (ao, pv, 'inferior_vena_cava', 'heart') if o in organs]
    cb = [out['organs'][o]['beta'] for o in contrast
          if np.isfinite(out['organs'][o]['beta'])]

    mb = float(np.median(betas)) if betas else float('nan')
    mv = float(np.median(vrs)) if vrs else float('nan')
    mcb = float(np.median(cb)) if cb else float('nan')
    out['median_beta'] = mb
    out['median_var_ratio'] = mv
    out['median_beta_contrast_organs'] = mcb

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  median beta (all organs)          {mb:.3f}")
    print(f"  median beta (contrast organs)     {mcb:.3f}")
    print(f"  median var(gen)/var(real)         {mv:.3f}")

    if mcb < 0.35:
        print(f"\n  ALEATORIC HYPOTHESIS SUPPORTED. The model recovers only "
              f"{100 * mcb:.0f}% of the\n"
              f"  case-to-case variation in contrast level and its output is "
              f"{1 / max(mv, 1e-6):.1f}x\n"
              f"  under-dispersed. It is largely emitting the training-set average "
              f"enhancement.\n"
              f"  No loss or architecture change can fix this — the information is "
              f"not in the\n"
              f"  input. Remedies: condition on phase (multi-phase training), "
              f"condition on a\n"
              f"  user-specified target level, or predict a distribution instead of "
              f"a point.")
    elif mcb > 0.75:
        print(f"\n  ALEATORIC HYPOTHESIS NOT SUPPORTED. The model tracks "
              f"{100 * mcb:.0f}% of the true\n"
              f"  level variation, so contrast level IS largely predictable from the "
              f"NCCT.\n"
              f"  The residual is then ordinary model error — spatial/texture work "
              f"and\n"
              f"  capacity remain on the table.")
    else:
        print(f"\n  PARTIAL. The model recovers {100 * mcb:.0f}% of the level "
              f"variation: some of the\n"
              f"  contrast level is predictable, some is not. Conditioning should "
              f"recover a\n"
              f"  measurable but bounded amount — size the expected gain from "
              f"(1 - beta).")

    print(f"\n  n = {n} cases. |r| < 0.44 is not significant at p<0.05 with this n.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {**out, 'real_hu': {o: R[o].tolist() for o in organs},
             'gen_hu': {o: G[o].tolist() for o in organs}}, indent=2))
        print(f"\n[written] {a.out}")


if __name__ == '__main__':
    main()
