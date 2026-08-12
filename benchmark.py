#!/usr/bin/env python
"""
Master benchmark table for NCCT→CECT models — one metric suite, one test set.

Every model is scored the same way, from a manifest CSV
`(gen_path, real_path, mask_path, target_phase)` of NIfTI volumes — the same
contract phase_eval.py already uses. So any method, however it was trained (2D or
3D, GAN or diffusion, this repo or an external one), plugs in by emitting synthetic
CECT volumes on the source grid and a manifest pointing at them.

Per case it computes, on the shared HU[-200,400]→[0,1] domain:
  * global pixel : PSNR, SSIM, MAE, MSE, PCC        (metrics.py)
  * organ-region : PSNR, SSIM, MAE, MSE, PCC        (metrics.py, mask>0)
  * phase fidelity: acc-vs-target, agreement-w/-real, gen target-prob,
                    feature-L1 (HU)                  (phase_eval.PhaseEvaluator)

Each NIfTI triple is loaded once and shared between the pixel metrics and the
phase evaluator — no double read of large volumes.

Significance is by PAIRED per-case test vs a chosen baseline (the same 20 test
cases scored by two models), plus per-model per-case std, because between-case
variance dominates between-model differences on this data. There is no
epoch-to-epoch "noise floor" here — that only existed inside a training run.

Usage:
    # auto-discover the existing runs (each has phase_infer/manifest.csv):
    python benchmark.py --runs_dir ../out_synthesis_train \
        --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl --out analysis/benchmark

    # or explicit models:
    python benchmark.py --weights xgb.pkl \
        --manifest ours=.../manifest.csv --manifest resvit=.../manifest.csv
"""

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# append, NOT insert(0, ...): orgFeatXGB_CTPhase has its own train.py, and putting
# it first on sys.path shadows this repo's train.py for anything imported later in
# the same process.
sys.path.append(str(Path(__file__).resolve().parent / 'orgFeatXGB_CTPhase'))
import metrics as M
from phase_eval import PhaseEvaluator, aggregate            # reused verbatim
from organ_features import load_organ_label_map             # noqa: F401 (PhaseEvaluator uses it)

PHASE_IDS = {'non-contrast': 0, 'arterial': 1, 'venous': 2, 'delayed': 3}


def _load(p: str) -> np.ndarray:
    import nibabel as nib
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def _phase_id(v) -> int:
    return PHASE_IDS[str(v).lower()] if not str(v).isdigit() else int(v)


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def read_tiling(manifest: Path) -> Dict:
    """Patch geometry from the run's own run_config.json, for the seam metric.

    `<run>/phase_infer/manifest.csv` → `<run>/run_config.json`, and
    `<run>/phase_infer/<phase>/manifest.csv` → the same file one level further up.
    The config is located by walking UP rather than by a fixed `parent.parent`,
    which was correct only for the flat layout: for a multi-phase run it resolved
    to `<run>/phase_infer/`, found nothing, and returned {} — so seam came out NaN
    for exactly those runs and read as "no tiling geometry" rather than as a bug.

    Returns {} for models that genuinely were not produced by this pipeline
    (external baselines, whole-slice models) — the seam metric is then reported as
    NaN rather than as a wrong number computed against a tiling that never
    happened.

    Anchored on the `phase_infer/` directory rather than on a depth count, because
    the two layouts sit at different depths and no single count is right for both.
    A plain "walk up until a run_config.json turns up" is worse still: for a model
    that genuinely has none it would keep climbing into the runs_dir and score the
    seam against some unrelated run's patch geometry — a wrong number where NaN is
    the correct answer.

    Falls back to the historical `parent.parent` for an output directory not named
    `phase_infer` (infer_volume.py's --out_dir is free-form).
    """
    anchor = next((q for q in manifest.parents if q.name == 'phase_infer'), None)
    cfg_path = ((anchor.parent if anchor else manifest.parent.parent)
                / 'run_config.json')
    if not cfg_path.exists():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return {}
    if 'patch_size' not in cfg:
        return {}
    return {'patch_size': cfg['patch_size'], 'overlap': cfg.get('overlap', 0.5)}


def score_model(name: str, manifest: Path, ev: PhaseEvaluator,
                hu_min: float, hu_max: float, gen_in_hu: bool,
                tiling: Optional[Dict] = None) -> List[Dict]:
    """Full per-case metric rows for one model."""
    tiling = tiling or {}
    rows = []
    with manifest.open() as f:
        cases = list(csv.DictReader(f))
    for c in cases:
        try:
            gen = _load(c['gen_path'])       # HU if gen_in_hu, else model's own scale
            real = _load(c['real_path'])     # HU
            mask = _load(c['mask_path'])
        except Exception as e:
            print(f"  [{name}] skip {Path(c.get('gen_path','?')).name}: load failed ({e})")
            continue
        if not (gen.shape == real.shape == mask.shape):
            print(f"  [{name}] skip {Path(c['gen_path']).name}: shape "
                  f"{gen.shape}/{real.shape}/{mask.shape}")
            continue
        tp = _phase_id(c['target_phase'])

        # Pixel metrics: both to the shared [0,1] domain. If the gen is not already
        # HU, to_unit still maps its window consistently with the real volume.
        g01 = M.to_unit(gen, hu_min, hu_max)
        r01 = M.to_unit(real, hu_min, hu_max)
        vm = M.volume_metrics(g01, r01)
        om = M.masked_metrics(g01, r01, mask)
        # Body-region: same pixel metrics, but excluding voxels the HU window
        # saturates to 0/1 in both volumes. Built from `real` so it is identical
        # across models.
        bmask = M.body_mask(real)
        bm = M.masked_metrics(g01, r01, bmask)

        # Texture + reconstruction consistency. Neither is expressible in the
        # metrics above: PSNR/SSIM/per-organ HU all reward blur, and nothing
        # scores tile seams or through-plane flicker at all.
        tx = M.texture_metrics(g01, r01, mask)
        cs = M.consistency_metrics(g01, r01, tiling.get('patch_size'),
                                   tiling.get('overlap'))

        # Phase fidelity: the evaluator handles HU internally.
        ph = ev.score_case(gen, real, mask, tp, hu_min=hu_min, hu_max=hu_max,
                            gen_in_hu=gen_in_hu)

        rows.append({
            'model': name,
            # Join key across models = the full REAL CECT path: identical for a
            # given case across every model's manifest (they all point at the same
            # ground-truth file), and unique per case. Neither the gen basename
            # (models may reuse gen.nii.gz per case dir) nor the real basename
            # (per-case dirs can repeat it) is safe as the key.
            '_key': c['real_path'],
            'case': Path(c['real_path']).name,
            **{f'{k}': vm[k] for k in ('psnr', 'ssim', 'mae', 'mse', 'pcc')},
            **{f'org_{k}': om[k] for k in ('psnr', 'ssim', 'mae', 'mse', 'pcc')},
            **{f'body_{k}': bm[k] for k in ('psnr', 'ssim', 'mae', 'mse', 'pcc')},
            'body_frac': float(bmask.mean()),
            **tx,          # raps_hf, grad_w1, org_grad_w1
            **cs,          # seam, zflicker
            'phase_match': int(ph['gen_matches_target']),
            'agree_real': int(ph['gen_matches_real']),
            'gen_prob': ph['gen_target_prob'],
            'feature_l1_hu': ph['feature_l1_hu'],
            '_phase_case': ph,       # kept for the phase aggregate()
        })
    return rows


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float('nan')


def _nanstd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def summarise(name: str, rows: List[Dict]) -> Dict:
    pixel = ['psnr', 'ssim', 'mae', 'mse', 'pcc',
             'org_psnr', 'org_ssim', 'org_mae', 'org_mse', 'org_pcc',
             'body_psnr', 'body_ssim', 'body_mae', 'body_frac',
             'raps_hf', 'grad_w1', 'org_grad_w1', 'seam', 'zflicker', 'zaniso']
    out = {'model': name, 'n': len(rows)}
    for k in pixel:
        out[k] = _nanmean([r.get(k) for r in rows])
        out[k + '_std'] = _nanstd([r.get(k) for r in rows])
    ph = aggregate([r['_phase_case'] for r in rows])         # reuse phase_eval
    out.update({
        'phase_acc': ph.get('gen_phase_accuracy_vs_target'),
        'agree_real': ph.get('gen_agreement_with_real'),
        'gen_prob': ph.get('mean_gen_target_prob'),
        'feature_l1_hu': ph.get('mean_feature_l1_hu'),
    })
    return out


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def paired_t(deltas: List[float]):
    d = [x for x in deltas if not (isinstance(x, float) and math.isnan(x))]
    if len(d) < 2:
        return (0.0, 0.0, 0)
    m, s = st.mean(d), st.stdev(d)
    t = m / (s / math.sqrt(len(d))) if s > 0 else 0.0
    return (m, t, sum(1 for x in d if x < 0))


def sig(t):
    a = abs(t)
    return '***' if a > 3.88 else '**' if a > 2.86 else '*' if a > 2.09 else 'ns'


def _dev1(x):
    """Distance from 1.0 — the quality direction for ratio metrics (raps_hf,
    zflicker, seam), where both 0.7 and 1.3 are failures in opposite directions
    and the raw value cannot be ranked."""
    return abs(x - 1.0) if x is not None and not math.isnan(x) else float('nan')


# (row key, better_is_low, transform, display label)
PAIRED_METRICS = [
    ('feature_l1_hu', True,  None,  'feature_l1_hu'),
    ('org_mae',       True,  None,  'org_mae'),
    ('org_ssim',      False, None,  'org_ssim'),
    ('psnr',          False, None,  'psnr'),
    ('body_mae',      True,  None,  'body_mae'),
    # Texture: the axis the four above cannot see.
    ('raps_hf',       True,  _dev1, '|raps_hf-1|'),
    ('grad_w1',       True,  None,  'grad_w1'),
    ('org_grad_w1',   True,  None,  'org_grad_w1'),
    # Reconstruction artifacts.
    ('seam',          True,  _dev1, '|seam-1|'),
    ('zflicker',      True,  _dev1, '|zflicker-1|'),
    ('zaniso',        True,  _dev1, '|zaniso-1|'),
]


def paired_block(all_rows: Dict[str, List[Dict]], base: str, out: List[str]):
    out.append(f'\n## Paired per-case tests vs "{base}"  (negative = better)')
    br = {r['_key']: r for r in all_rows[base]}
    # A fixed 26-char model column silently ran into the metric column for the
    # `<run>/<phase>` names ('multiphase_film_adv_slices11/venous' is 35), which
    # printed as 'multiphase_film_adv/venousfeature_l1_hu'. Size it to the data.
    mw = max(26, max(len(n) for n in all_rows) + 2)
    out.append(f"\n{'model':{mw}s}{'metric':16s}{'delta':>10}{'t':>8}{'sig':>5}{'better':>9}")
    for name, rows in all_rows.items():
        if name == base:
            continue
        rr = {r['_key']: r for r in rows}
        common = [c for c in br if c in rr]
        if not common:
            # No shared cases — e.g. an arterial arm against a venous baseline,
            # where `_key` is the real CECT path and the two never coincide.
            # paired_t([]) returns (0.0, 0.0, 0), so without this the block would
            # print a full set of '+0.000 ns' rows that read as "identical to the
            # baseline" when nothing was actually compared.
            out.append(f'{name:{mw}s}— no cases in common with the baseline; '
                       f'not comparable')
            out.append('')
            continue
        first = True
        for metric, better_is_low, tf, label in PAIRED_METRICS:
            d = []
            for c in common:
                a, b = br[c].get(metric), rr[c].get(metric)
                if a is None or b is None:
                    continue
                if tf is not None:
                    a, b = tf(a), tf(b)
                d.append((b - a) if better_is_low else (a - b))   # sign: neg = better
            m, t, w = paired_t(d)
            out.append(f"{(name if first else ''):{mw}s}{label:16s}"
                       f"{m:+10.3f}  {t:+8.2f}  {sig(t):>4}{w:>5}/{len(d)}")
            first = False
        out.append('')
    out.append('  sign convention: negative delta = the model beats the baseline.')
    out.append('  sig: * p<.05  ** p<.01  *** p<.001 (paired, df~n-1)')


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def master_table(summaries: List[Dict], out: List[str]):
    out.append('# NCCT→CECT benchmark — master table\n')
    out.append('All metrics on the shared HU[-200,400]→[0,1] domain, same test cases.\n')
    cols = [('n', 'n', '{:d}'), ('psnr', 'PSNR', '{:.2f}'), ('ssim', 'SSIM', '{:.4f}'),
            ('mae', 'MAE', '{:.4f}'), ('mse', 'MSE', '{:.5f}'), ('pcc', 'PCC', '{:.4f}'),
            ('org_psnr', 'oPSNR', '{:.2f}'), ('org_ssim', 'oSSIM', '{:.4f}'),
            ('org_mae', 'oMAE', '{:.4f}'),
            ('body_psnr', 'bPSNR', '{:.2f}'), ('body_mae', 'bMAE', '{:.4f}'),
            ('body_frac', 'body%', '{:.2f}'),
            ('phase_acc', 'phase', '{:.2f}'), ('gen_prob', 'prob', '{:.4f}'),
            ('feature_l1_hu', 'featHU', '{:.2f}'),
            ('raps_hf', 'RAPS', '{:.3f}'), ('grad_w1', 'gradW1', '{:.4f}'),
            ('org_grad_w1', 'oGradW1', '{:.4f}'),
            ('seam', 'seam', '{:.3f}'), ('zflicker', 'zflick', '{:.3f}'),
            ('zaniso', 'zaniso', '{:.3f}')]
    out.append('| model | ' + ' | '.join(h for _, h, _ in cols) + ' |')
    out.append('|' + '---|' * (len(cols) + 1))
    for s in summaries:
        cells = []
        for key, _, fmt in cols:
            v = s.get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) and not
                         (isinstance(v, float) and math.isnan(v)) else '—')
        out.append(f"| {s['model']} | " + ' | '.join(cells) + ' |')
    out.append('\noPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. '
               'Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.')
    out.append('\n**Texture and consistency** (`metrics.py`): RAPS = high-frequency '
               'spectral energy vs real; gradW1/oGradW1 = W1 distance between '
               'gradient-magnitude distributions, global and organ-region; seam = '
               'tile-boundary over interior gradient; zflick = inter-slice difference '
               'vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and '
               'both directions are failures** — rank them by |value - 1|, never as '
               '"higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated '
               'texture. seam > 1 is a visible tile boundary; zflick > 1 is '
               'slice-to-slice flicker. gradW1 is a distance: lower is better. '
               'seam is NaN for models with no known tiling geometry.')
    out.append('\n**Caveat:** external models retrained on this data at this scale do not '
               'reproduce their papers\' reported numbers — this is a controlled same-data, '
               'same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see '
               'PROJECT_PLAN.md); read organ-region + phase fidelity as primary.')


# ---------------------------------------------------------------------------

def discover(runs_dir: Path) -> Dict[str, Path]:
    """Trained runs that also have a manifest, i.e. have been through infer_volume.py.

    A run with a checkpoint but no manifest is reported rather than silently
    dropped: that is a missing inference step, not an absent model, and it is
    invisible in the output table otherwise.
    """
    found, pending = {}, []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.replace('literature_baseline_', '')
        pi = d / 'phase_infer'
        if (pi / 'manifest.csv').exists():
            found[name] = pi / 'manifest.csv'
            continue
        # MULTI-PHASE runs write one manifest per phase SUBDIRECTORY
        # (infer_volume.py: a multi-phase model emits several volumes per case,
        # which would collide on '{case_id}_syn.nii.gz' in a single directory).
        # Looking only for the flat manifest silently skipped those runs
        # entirely — they were absent from the table rather than reported.
        #
        # Each phase becomes its OWN model row, `<run>/<phase>`. They must not be
        # pooled: an arterial and a venous volume are different targets, and
        # averaging their featHU would make the multi-phase arms incomparable to
        # every single-phase run, which is the whole point of the M1/M2/M3 design.
        per_phase = sorted(pi.glob('*/manifest.csv')) if pi.is_dir() else []
        if per_phase:
            for m in per_phase:
                found[f'{name}/{m.parent.name}'] = m
            continue
        if (d / 'best_model.pth').exists():
            pending.append(d.name)
    if pending:
        print(f'  [discover] {len(pending)} trained run(s) have no '
              f'phase_infer/manifest.csv and are NOT scored — run infer_volume.py '
              f'on them first: {", ".join(pending)}')
    return found


def resolve_baseline(names, requested: str | None):
    """Map a `--baseline` spelling onto a model name, or None if it matches none.

    discover() strips the `literature_baseline_` prefix, so a baseline given as a
    run DIRECTORY name would silently never match. Accept either spelling. With
    no request, fall back to the L1-only reference arm.
    """
    names = list(names)
    if not requested:
        # 'l1_only' is the reference arm. Matching any *only* name sorted first
        # picked `l1_huprofile_only` — an ablation, and the weakest level model in
        # the table — as the silent default for a whole paired report.
        return next((n for n in names if n in ('l1_only', 'literature_baseline_l1_only')),
                    next((n for n in names if 'only' in n), next(iter(names), None)))
    for cand in (requested, requested.replace('literature_baseline_', ''),
                 f'literature_baseline_{requested}'):
        if cand in names:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='XGBoost phase-model .pkl')
    ap.add_argument('--organ_map', type=Path,
                    default=Path(__file__).resolve().parent / 'orgFeatXGB_CTPhase' /
                            'retrain_out_full' / 'ts_label_map_total.json',
                    help='explicit {name:id} map so eval does not need TotalSegmentator')
    ap.add_argument('--manifest', action='append', default=[],
                    help='name=path.csv (repeatable)')
    ap.add_argument('--runs_dir', type=Path, default=None,
                    help='auto-discover <run>/phase_infer/manifest.csv under here')
    ap.add_argument('--baseline', default=None, help='model name for paired tests')
    ap.add_argument('--hu_min', type=float, default=-200.0)
    ap.add_argument('--hu_max', type=float, default=400.0)
    ap.add_argument('--gen_not_hu', action='store_true',
                    help='generated volumes are NOT already in HU (default: they are)')
    ap.add_argument('--out', type=Path, default=Path('analysis/benchmark'))
    args = ap.parse_args()

    manifests: Dict[str, Path] = {}
    if args.runs_dir:
        manifests.update(discover(args.runs_dir))
    for spec in args.manifest:
        name, _, path = spec.partition('=')
        manifests[name] = Path(path)
    if not manifests:
        raise SystemExit('no manifests — pass --runs_dir or --manifest name=path.csv')

    # Check the baseline name BEFORE scoring: it costs seconds to catch a typo
    # here and ~an hour of volume scoring to catch it after the loop, where the
    # results were then discarded unwritten.
    if args.baseline and resolve_baseline(manifests, args.baseline) is None:
        raise SystemExit(f'--baseline {args.baseline!r} is not one of the models to be '
                         f'scored: ' + ', '.join(manifests) +
                         '\n(pass its manifest too: --manifest '
                         f'{args.baseline}=<path>/manifest.csv, or --runs_dir)')

    organ_map = None
    if args.organ_map and args.organ_map.exists():
        # The full 117-class TS map contains the 16 organs the model needs; keep
        # only those (load_organ_label_map validates the exact 16-organ contract).
        full = json.loads(args.organ_map.read_text())
        from organ_features import ORGANS
        organ_map = {o: int(full[o]) for o in ORGANS if o in full}
        if len(organ_map) != len(ORGANS):
            organ_map = None    # incomplete → let PhaseEvaluator fall back to TS
    ev = PhaseEvaluator(args.weights, organ_label_map=organ_map)
    gen_in_hu = not args.gen_not_hu
    print(f'Scoring {len(manifests)} models: {", ".join(manifests)}')

    all_rows, summaries = {}, []
    for name, mpath in manifests.items():
        if not mpath.exists():
            print(f'  [{name}] manifest missing: {mpath} — skipped')
            continue
        rows = score_model(name, mpath, ev, args.hu_min, args.hu_max, gen_in_hu,
                           tiling=read_tiling(mpath))
        if not rows:
            print(f'  [{name}] no scored cases — skipped')
            continue
        all_rows[name] = rows
        summaries.append(summarise(name, rows))
        s = summaries[-1]
        print(f'  [{name}] n={s["n"]} PSNR {s["psnr"]:.2f} oSSIM {s["org_ssim"]:.4f} '
              f'featHU {s["feature_l1_hu"]:.2f}')

    if not all_rows:
        raise SystemExit('no model produced scored cases — nothing to report')

    lines: List[str] = []
    master_table(summaries, lines)
    # A missing paired block is a note in the report, never a hard exit: the
    # per-model scoring is the expensive part and stays worth writing out.
    base = resolve_baseline(all_rows, args.baseline)
    if len(all_rows) < 2:
        lines.append(f'\n_Only one model scored ({next(iter(all_rows))}) — paired tests '
                     'need at least two, so none were run._')
    elif base is None:
        lines.append(f'\n_Baseline {args.baseline!r} scored no cases — paired tests '
                     'skipped._')
    else:
        paired_block(all_rows, base, lines)
    report = '\n'.join(lines)
    print('\n' + report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'master_table.md').write_text(report)
    with (args.out / 'master_table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in summaries[0] if k != 'model'] and
                           ['model'] + [k for k in summaries[0] if k != 'model'])
        w.writeheader()
        w.writerows(summaries)
    # per-case rows for downstream analysis
    with (args.out / 'per_case.csv').open('w', newline='') as f:
        flat = [{k: v for k, v in r.items() if k != '_phase_case'}
                for rows in all_rows.values() for r in rows]
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(f'\n[written] {args.out}/master_table.md, master_table.csv, per_case.csv')


if __name__ == '__main__':
    main()
