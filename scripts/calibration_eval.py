#!/usr/bin/env python
"""
Is the predicted spread the TRUE conditional spread?

This is the question DIFFUSION_PLAN.md §3 identifies as the actual contribution.
A point metric cannot answer it: §3 shows one honest draw from the true p(y|x) is
expected to be ~1.41x WORSE on featHU than the conditional mean, so a model that
gets the distribution exactly right LOSES on featHU. This script scores the
distribution instead.

TWO LEVELS, and they are not interchangeable
--------------------------------------------
  per-ORGAN (headline)  the predictive distribution of an organ's MEDIAN HU for
      one case, scored against that case's real median. This is the quantity the
      XGBoost phase model reads (featHU IS per-organ median-HU error), the
      quantity audit_enhancement.py regresses, and the level at which the
      residual was measured to be aleatoric: dose and bolus timing set a case's
      whole enhancement level, not individual voxels.
      Targets: variance ratio 1.0 (currently 0.176), 90% coverage 0.90.

      SAMPLE SIZE. A 90% interval needs at least 19 samples to be EXPRESSIBLE
      (metrics.min_samples_for_level); below that it is reported as NaN rather
      than as a low score. A 50% interval needs 3, so it is the coverage number
      an N=8 exploratory run can actually produce. CRPS and the variance ratio
      have no such floor and are usable at any N.

  per-VOXEL             the predictive distribution of one voxel, over the organ
      mask. Reported because it is the only level at which a heteroscedastic NLL
      head can compete, and because the GAP between the two levels for that model
      is itself the result — per-voxel independent sigmas average away to nothing
      at organ scale (sigma/sqrt(n) over ~1e5 voxels), so it scores well here and
      near zero above, by construction. See models_hetero.py.

TWO INPUT MODES
---------------
  --mode ensemble   a diffusion run. Reads `organ_medians.json` written by
      infer_volume.py (per-organ median HU for each of the N samples, per case).
      Per-voxel needs the individual sample volumes, i.e. `--save_samples` at
      inference time; without them only the per-organ block is produced.

  --mode gaussian   a heteroscedastic run. Reads `{case}_syn.nii.gz` (mu) and
      `{case}_std.nii.gz` (sigma) and MONTE-CARLOs the organ-median distribution
      from the organ's own voxels — exact, rather than the sqrt(pi/2)*sigma/sqrt(n)
      asymptotic, and cheap because only masked voxels are drawn. Per-voxel CRPS
      uses the closed-form Gaussian expression, no sampling at all.

Both modes emit the same JSON so the two models sit in one table.

Usage:
    python scripts/calibration_eval.py --mode ensemble \
        --dir ../out_synthesis_train/literature_baseline_diff_v/phase_infer \
        --out analysis/calibration_diff_v.json

    python scripts/calibration_eval.py --mode gaussian \
        --dir ../out_synthesis_train/literature_baseline_hetero_nll/phase_infer \
        --out analysis/calibration_hetero.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from metrics import (achieved_level, crps_ensemble, interval_coverage,
                     min_samples_for_level, pit_values, variance_ratio)

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    raise SystemExit("calibration_eval.py needs nibabel (pip install nibabel)")

# The 16 organs the XGBoost phase model reads, in its trained order. Imported
# rather than re-listed so a version bump cannot silently score a different set
# from the one featHU is computed over.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                       'orgFeatXGB_CTPhase'))
from organ_features import ORGANS                            # noqa: E402

# Below this an organ median is too noisy to be a calibration target — the same
# threshold audit_enhancement.py uses, so the two scripts score the same organs.
MIN_VOXELS = 64

# Monte-Carlo replicates of the organ median in --mode gaussian. 512 gives a
# coverage estimate stable to ~+/-0.01, which is well inside the 0.90 target's
# useful resolution, and costs nothing next to loading the volumes.
N_MC = 512


def _load(p: str) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-organ (headline)
# ---------------------------------------------------------------------------

def score_per_organ(real: Dict[str, List[float]],
                    samples: Dict[str, List[List[float]]],
                    level: float = 0.90) -> Dict:
    """Score each organ's median-HU predictive distribution across cases.

    `real[organ]`    = [truth per case]
    `samples[organ]` = [[sample_1 .. sample_N] per case]

    Cases where an organ is absent carry NaN and are dropped PER ORGAN, so one
    missing gallbladder does not remove a case from every other organ's score.
    """
    out: Dict[str, Dict] = {}
    for o in ORGANS:
        y = np.asarray(real.get(o, []), np.float64)
        S = np.asarray(samples.get(o, []), np.float64)      # (n_cases, n_samples)
        if y.size == 0 or S.size == 0:
            continue
        ok = np.isfinite(y) & np.all(np.isfinite(S), axis=1)
        if ok.sum() < 3:
            continue
        y, S = y[ok], S[ok]

        # CRPS and coverage are per case, then averaged / counted over cases.
        crps = float(np.mean([crps_ensemble(S[i], y[i]) for i in range(len(y))]))
        # Coverage at two levels. The 90% one needs n >= 19 samples to be
        # EXPRESSIBLE at all (metrics.min_samples_for_level) — below that it is
        # NaN, not a low score — while the 50% one needs only 3, so an N=8
        # exploratory run still yields a real calibration number.
        cov = float(np.mean([interval_coverage(S[i], y[i], level)
                             for i in range(len(y))]))
        cov50 = float(np.mean([interval_coverage(S[i], y[i], 0.50)
                               for i in range(len(y))]))
        pit = np.asarray([float(pit_values(S[i], y[i])) for i in range(len(y))])

        means = S.mean(axis=1)
        # Two var-ratio readings — see metrics.variance_ratio. The first is the
        # one comparable to analysis/BASELINE_REFERENCE.md's 0.176; the second
        # says whether the model's CLAIMED uncertainty is the right size.
        vr_out = variance_ratio(means, y)
        vr_pred = variance_ratio(means, y, pred_vars=S.var(axis=1, ddof=1))

        out[o] = {
            'n_cases':        int(ok.sum()),
            'crps_hu':        crps,
            'coverage':       cov,
            'coverage50':     cov50,
            'var_ratio_out':  vr_out,
            'var_ratio_pred': vr_pred,
            'real_sd_hu':     float(np.std(y)),
            'pred_mean_sd_hu': float(np.std(means)),
            'ens_sd_hu':      float(np.mean(np.std(S, axis=1, ddof=1))),
            'abs_err_hu':     float(np.mean(np.abs(means - y))),
            'pit':            pit.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Per-voxel
# ---------------------------------------------------------------------------

def _gaussian_crps(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution.

        CRPS = sd * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sd

    Used instead of sampling in --mode gaussian: at ~1e7 masked voxels, drawing
    an ensemble per voxel to then estimate a quantity that has an exact
    expression is pure waste, and the closed form has no Monte-Carlo noise to
    confound a comparison against the ensemble model.
    """
    from scipy.special import erf
    sd = np.maximum(sd, 1e-6)
    z = (y - mu) / sd
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    return float(np.mean(sd * (z * (2.0 * Phi - 1.0) + 2.0 * phi
                               - 1.0 / np.sqrt(np.pi))))


def score_per_voxel_ensemble(sample_vols: np.ndarray, real: np.ndarray,
                             sel: np.ndarray, level: float = 0.90,
                             max_voxels: int = 2_000_000,
                             rng: Optional[np.random.Generator] = None) -> Dict:
    """Per-voxel CRPS/coverage over the organ mask from N sample volumes.

    Subsampled to `max_voxels` because the estimate is a mean over voxels and a
    uniform subsample of 2e6 of them is already tighter than any difference worth
    reporting, while the full set makes the sort inside crps_ensemble the
    dominant cost of the whole script.
    """
    rng = rng or np.random.default_rng(0)
    idx = np.flatnonzero(sel.ravel())
    if idx.size == 0:
        return {}
    if idx.size > max_voxels:
        idx = rng.choice(idx, max_voxels, replace=False)
    S = sample_vols.reshape(sample_vols.shape[0], -1)[:, idx].astype(np.float64)
    y = real.ravel()[idx].astype(np.float64)
    return {
        'crps_hu':  crps_ensemble(S, y),
        'coverage': interval_coverage(S, y, level),
        'ens_sd_hu': float(np.mean(np.std(S, axis=0, ddof=1))),
        'abs_err_hu': float(np.mean(np.abs(S.mean(axis=0) - y))),
        'n_voxels': int(idx.size),
    }


def score_per_voxel_gaussian(mu: np.ndarray, sd: np.ndarray, real: np.ndarray,
                             sel: np.ndarray, level: float = 0.90) -> Dict:
    """Per-voxel CRPS/coverage over the organ mask for a (mu, sigma) field."""
    from scipy.stats import norm
    m, s, y = mu[sel].astype(np.float64), sd[sel].astype(np.float64), real[sel].astype(np.float64)
    if m.size == 0:
        return {}
    z = norm.ppf(0.5 + level / 2.0)
    return {
        'crps_hu':  _gaussian_crps(m, s, y),
        'coverage': float(np.mean(np.abs(y - m) <= z * np.maximum(s, 1e-6))),
        'ens_sd_hu': float(np.mean(s)),
        'abs_err_hu': float(np.mean(np.abs(m - y))),
        'n_voxels': int(m.size),
    }


# ---------------------------------------------------------------------------
# Mode: gaussian (heteroscedastic run) — build the organ-median distribution
# ---------------------------------------------------------------------------

def organ_medians_from_gaussian(mu: np.ndarray, sd: np.ndarray,
                                mask: np.ndarray, label_map: Dict[str, int],
                                n_mc: int, rng: np.random.Generator):
    """Monte-Carlo the predictive distribution of each organ's MEDIAN HU.

    Draws n_mc realisations of the organ's voxels from N(mu, sd) and takes the
    median of each. Exact for the model as specified — no asymptotic — and it
    makes the structural point of models_hetero.py measurable rather than
    asserted: with independent per-voxel sigmas the resulting spread is
    ~sigma/sqrt(n_voxels), so these draws collapse onto a single value and the
    organ-level coverage that follows is near zero.

    Voxels are capped per organ; the median's sampling distribution is what is
    wanted and it is already degenerate well before 50k voxels, so drawing more
    changes nothing but the runtime.
    """
    out: Dict[str, List[float]] = {}
    for o in ORGANS:
        lid = label_map.get(o)
        if lid is None:
            continue
        sel = mask == lid
        n = int(sel.sum())
        if n < MIN_VOXELS:
            out[o] = [float('nan')] * n_mc
            continue
        m, s = mu[sel], sd[sel]
        if n > 50_000:
            keep = rng.choice(n, 50_000, replace=False)
            m, s = m[keep], s[keep]
        draws = m[None, :] + s[None, :] * rng.standard_normal((n_mc, m.size))
        out[o] = np.median(draws, axis=1).astype(np.float64).tolist()
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_ensemble(d: Path, level: float, per_voxel: bool, rng) -> Dict:
    om_path = d / 'organ_medians.json'
    if not om_path.exists():
        raise SystemExit(
            f"{om_path} not found. It is written by infer_volume.py when "
            f"--n_samples > 1; a run made before that flag existed has to be "
            f"re-sampled.")
    om = json.loads(om_path.read_text())
    cases = om['cases']
    real = {o: [] for o in ORGANS}
    samp = {o: [] for o in ORGANS}
    for cid in sorted(cases):
        c = cases[cid]
        for i, o in enumerate(om['organs']):
            if o not in real:
                continue
            real[o].append(c['real'][i])
            samp[o].append([s[i] for s in c['samples']])

    res = {'mode': 'ensemble', 'n_cases': len(cases),
           'n_samples': int(om['n_samples']),
           'level': level,
           'per_organ': score_per_organ(real, samp, level)}

    if per_voxel:
        rows = list(csv.DictReader(open(d / 'manifest.csv')))
        vox = []
        for r in rows:
            cid = Path(r['gen_path']).name.replace('_syn.nii.gz', '')
            files = sorted(d.glob(f'{cid}_s*.nii.gz'))
            if not files:
                continue
            S = np.stack([_load(str(f)) for f in files])
            v = score_per_voxel_ensemble(S, _load(r['real_path']),
                                         _load(r['mask_path']) > 0, level, rng=rng)
            if v:
                vox.append(v)
        res['per_voxel'] = _mean_dicts(vox) if vox else None
        if not vox:
            print("note: no per-sample volumes found — re-run inference with "
                  "--save_samples for the per-voxel block")
    return res


def run_gaussian(d: Path, level: float, per_voxel: bool, label_map: Dict[str, int],
                 n_mc: int, rng) -> Dict:
    rows = list(csv.DictReader(open(d / 'manifest.csv')))
    real = {o: [] for o in ORGANS}
    samp = {o: [] for o in ORGANS}
    vox = []
    for r in rows:
        gen_p = Path(r['gen_path'])
        sd_p = gen_p.with_name(gen_p.name.replace('_syn.nii.gz', '_std.nii.gz'))
        if not sd_p.exists():
            raise SystemExit(
                f"{sd_p} not found. --mode gaussian needs the per-voxel sigma "
                f"volume that infer_volume.py writes for a heteroscedastic run.")
        mu, sd = _load(str(gen_p)), _load(str(sd_p))
        rv, mask = _load(r['real_path']), _load(r['mask_path'])
        if not (mu.shape == sd.shape == rv.shape == mask.shape):
            print(f"  skip {gen_p.name}: shape mismatch")
            continue
        mask_i = np.round(mask).astype(np.int32)
        draws = organ_medians_from_gaussian(mu, sd, mask_i, label_map, n_mc, rng)
        for o in ORGANS:
            lid = label_map.get(o)
            sel = mask_i == lid if lid is not None else None
            real[o].append(float(np.median(rv[sel]))
                           if sel is not None and sel.sum() >= MIN_VOXELS
                           else float('nan'))
            samp[o].append(draws.get(o, [float('nan')] * n_mc))
        if per_voxel:
            v = score_per_voxel_gaussian(mu, sd, rv, mask_i > 0, level)
            if v:
                vox.append(v)

    res = {'mode': 'gaussian', 'n_cases': len(rows), 'n_samples': n_mc,
           'level': level,
           'per_organ': score_per_organ(real, samp, level)}
    if per_voxel:
        res['per_voxel'] = _mean_dicts(vox) if vox else None
    return res


def _mean_dicts(ds: List[Dict]) -> Dict:
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


# ---------------------------------------------------------------------------

def _report(res: Dict):
    po = res['per_organ']
    print('=' * 92)
    print(f"CALIBRATION — {res['mode']} | {res['n_cases']} cases | "
          f"N={res['n_samples']} | {int(res['level'] * 100)}% interval")
    print('=' * 92)
    n_s = int(res['n_samples'])
    tgt90, tgt50 = achieved_level(n_s, res['level']), achieved_level(n_s, 0.50)
    print(f"  {'organ':<30}{'n':>4}{'CRPS':>8}{'|err|':>8}{'cov90':>8}{'cov50':>8}"
          f"{'ens sd':>9}{'real sd':>9}{'varR out':>10}{'varR pred':>11}")
    for o, v in po.items():
        print(f"  {o:<30}{v['n_cases']:>4}{v['crps_hu']:>8.2f}{v['abs_err_hu']:>8.2f}"
              f"{v['coverage']:>8.2f}{v['coverage50']:>8.2f}"
              f"{v['ens_sd_hu']:>9.1f}{v['real_sd_hu']:>9.1f}"
              f"{v['var_ratio_out']:>10.3f}{v['var_ratio_pred']:>11.3f}")
    if np.isnan(tgt90):
        print(f"\n  ** cov90 is NaN: {n_s} samples cannot express a "
              f"{res['level']:.0%} interval (needs >= "
              f"{min_samples_for_level(res['level'])}). Read cov50 instead, or "
              f"re-sample with --n_samples 20. This is a sample-size limit, NOT "
              f"a calibration failure. **")
    else:
        print(f"\n  coverage targets at n={n_s}: cov90 -> {tgt90:.3f}, "
              f"cov50 -> {tgt50:.3f}  (order-statistic intervals are discrete, so "
              f"these are what a perfectly calibrated model scores here)")

    # Contrast organs are the ones §2's argument is about; the GI tract is a
    # negative control and averaging it in would flatter or flatten the headline.
    contrast = ['aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein',
                'heart', 'liver']
    have = [o for o in contrast if o in po]
    if have:
        mvr = float(np.median([po[o]['var_ratio_pred'] for o in have]))
        mcov = float(np.median([po[o]['coverage'] for o in have]))
        mcov50 = float(np.median([po[o]['coverage50'] for o in have]))
        mcrps = float(np.mean([po[o]['crps_hu'] for o in have]))
        print(f"\n  HEADLINE (contrast organs: {', '.join(have)})")
        print(f"    median var ratio (predicted)   {mvr:.3f}   "
              f"[target 1.000 | deterministic baseline 0.176]")
        print(f"    median 90% coverage            {mcov:.3f}   "
              f"[target {tgt90:.3f}]")
        print(f"    median 50% coverage            {mcov50:.3f}   "
              f"[target {tgt50:.3f}]")
        print(f"    mean CRPS                      {mcrps:.2f} HU")
        res['headline'] = {'organs': have, 'median_var_ratio_pred': mvr,
                           'median_coverage': mcov, 'median_coverage50': mcov50,
                           'coverage_target': tgt90, 'coverage50_target': tgt50,
                           'mean_crps_hu': mcrps}

    pv = res.get('per_voxel')
    if pv:
        print(f"\n  PER-VOXEL (organ mask): CRPS {pv['crps_hu']:.2f} HU | "
              f"coverage {pv['coverage']:.3f} | mean sd {pv['ens_sd_hu']:.1f} HU | "
              f"|err| {pv['abs_err_hu']:.2f} HU")
    print("\n  Read together: a model can score well per-voxel and near zero "
          "per-organ.\n  That gap is the finding, not a bug — see models_hetero.py.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', type=Path, required=True,
                    help='an inference output dir (has manifest.csv)')
    ap.add_argument('--mode', choices=['ensemble', 'gaussian'], required=True)
    ap.add_argument('--level', type=float, default=0.90)
    ap.add_argument('--per_voxel', action='store_true', default=True)
    ap.add_argument('--no_per_voxel', dest='per_voxel', action='store_false')
    ap.add_argument('--n_mc', type=int, default=N_MC,
                    help='--mode gaussian: organ-median Monte-Carlo replicates')
    ap.add_argument('--label_map', type=Path,
                    default=Path(__file__).resolve().parent.parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.mode == 'ensemble':
        res = run_ensemble(a.dir, a.level, a.per_voxel, rng)
    else:
        if not a.label_map.exists():
            raise SystemExit(f"label map not found: {a.label_map}")
        name_to_id = {k: int(v) for k, v in
                      json.loads(a.label_map.read_text()).items()}
        res = run_gaussian(a.dir, a.level, a.per_voxel,
                           {o: name_to_id[o] for o in ORGANS if o in name_to_id},
                           a.n_mc, rng)

    _report(res)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
        print(f"\n[written] {a.out}")


if __name__ == '__main__':
    main()
