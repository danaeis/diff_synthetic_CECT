"""
Re-train the CT contrast-phase XGBoost model — cleanly, on the FULL dataset,
saving fresh weights AND the complete feature table.

Improvements over the original train.py:
  - Builds the 16-organ feature table from EITHER the TS statistics .pkl
    (`stats_path`) OR, if that's missing, directly from the (CT, TS mask) pair
    (`organ_features.features_from_mask`) — so a missing/incomplete stats dump
    can't silently drop cases.
  - Reports feature-source counts and drops any all-NaN / unlabelled rows with a
    clear log (rather than crashing mid-fold).
  - StratifiedGroupKFold by StudyInstanceUID (patient-grouped, phase-stratified)
    — same honest protocol; also aggregates an out-of-fold (OOF) prediction for
    every case, giving one leakage-free confusion matrix over the whole dataset.
  - Saves: fresh ensemble weights (.pkl, same {"fold","model"} format the
    inference script expects), the full feature table (.npz + .csv), and a JSON
    metrics summary.

Usage (remote, in the CTPhase-XGBoost dir with its env):
    python retrain_xgb.py \
        --metadata_csv vindr_nifti_metadata.csv \
        --out_weights  xgb_vindr_retrained.pkl \
        --out_dir      retrain_out
"""

from __future__ import annotations   # lazy annotations: allows `pd.DataFrame` hints without a module-level pandas import

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
# pandas is imported lazily where used (labels/metadata CSV) so this module can
# be imported — and the directory-discovery path tested — without it.

from organ_features import ORGANS, features_from_stats_pkl, features_from_mask, load_organ_label_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

PHASE_NAMES = {0: 'non-contrast', 1: 'arterial', 2: 'venous', 3: 'delayed'}
PHASE_TO_ID = {v: k for k, v in PHASE_NAMES.items()}

# Filename-keyword phase fallback + label-string normalisation, mirroring the
# synthesis dataset.py (same keyword sets / aliases) so this reads the identical
# deeds-aligned data + labels.csv the synthesis pipeline uses.
_PHASE_KW = {
    'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native', 'nc', 'nce', 'noncon'],
    'arterial':     ['arterial', 'art', 'early', 'phase1', 'p1'],
    'venous':       ['venous', 'portal', 'pv', 'phase2', 'p2', 'late'],
    'delayed':      ['delayed', 'delay', 'equilibrium', 'phase3', 'p3'],
}
_ALIASES = {
    'nc': 'non-contrast', 'noncontrast': 'non-contrast', 'pre': 'non-contrast',
    'precontrast': 'non-contrast', 'non contrast': 'non-contrast',
    'art': 'arterial', 'aterial': 'arterial', 'a': 'arterial',
    'pv': 'venous', 'portal': 'venous', 'portal venous': 'venous',
    'portal-venous': 'venous', 'portal_venous': 'venous', 'v': 'venous',
    'delay': 'delayed', 'd': 'delayed',
}


def _normalize_phase(label):
    s = str(label).strip().lower()
    if s in PHASE_TO_ID:
        return s
    return _ALIASES.get(s)


def _infer_phase(name: str):
    n = name.lower()
    for phase, kws in _PHASE_KW.items():
        if any(k in n for k in kws):
            return phase
    return None


def build_feature_table_from_dir(data_dir, labels_csv, file_tag, seg_suffix, organ_map=None):
    """
    Discover labelled volumes in the synthesis deeds-aligned layout and extract
    the 16 per-organ median-HU features from each volume's co-registered mask.

    Mirrors synthesis dataset.py: per-case subdirs, `*{file_tag}.nii.gz` volumes
    (skip `_seg`/`_dvf`), phase from labels.csv (StudyInstanceUID/SeriesInstanceUID
    /Label) else inferred from filename; mask = volume with `{file_tag}.nii.gz` →
    `{file_tag}{seg_suffix}.nii.gz`. ONE labelled sample per unique volume
    (no generation pairs), grouped by case (patient).
    """
    from organ_features import features_from_paths
    data_dir = Path(data_dir)

    phase_map = {}
    if labels_csv and Path(labels_csv).exists():
        import pandas as pd
        ldf = pd.read_csv(labels_csv)
        for _, r in ldf.iterrows():
            phase_map.setdefault(str(r['StudyInstanceUID']), {})[str(r['SeriesInstanceUID'])] = str(r['Label']).lower()
        log.info(f"Loaded labels CSV: {len(phase_map)} studies")

    # --- Pass 1 (fast): collect candidate (volume, mask, phase) tuples without
    # loading any NIfTI, so we can show a progress bar over the slow extraction. ---
    log.info("Scanning data_dir for labelled volumes with masks …")
    candidates = []
    n_no_mask = n_no_label = 0
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        for f in sorted(case_dir.glob(f'*{file_tag}.nii.gz')):
            if '_seg' in f.name or '_dvf' in f.name:
                continue
            stem = f.name.replace(f'{file_tag}.nii.gz', '').replace('.nii', '')
            parts = stem.split('_')
            series_id = parts[1] if len(parts) >= 2 else stem
            raw = phase_map.get(case_id, {}).get(series_id)
            phase = _normalize_phase(raw) if raw is not None else _infer_phase(f.name)
            if phase is None or phase not in PHASE_TO_ID:
                n_no_label += 1
                continue
            mask_path = str(f).replace(f'{file_tag}.nii.gz', f'{file_tag}{seg_suffix}.nii.gz')
            if not Path(mask_path).exists():
                n_no_mask += 1
                continue
            candidates.append((case_id, series_id, str(f), mask_path, phase))
    log.info(f"Found {len(candidates)} candidate volumes "
             f"(skipped no_label={n_no_label}, no_mask={n_no_mask})")

    # Resolving the organ label map imports TotalSegmentator (heavy — can take
    # 10-30s the first time); log it so a slow import doesn't look like a hang.
    log.info("Resolving organ→label-id map from TotalSegmentator (importing TS may take a moment) …")
    organ_map = organ_map or load_organ_label_map()
    log.info(f"Resolved {len(organ_map)} organ labels; extracting per-organ features …")

    # --- Pass 2 (slow): load each (CT, mask) and extract features, with a bar. ---
    X, y, groups, series, sources = [], [], [], [], []
    n_fail = 0
    for case_id, series_id, vol_path, mask_path, phase in tqdm(
            candidates, desc='Extracting organ features', unit='vol'):
        try:
            feats = features_from_paths(vol_path, mask_path, organ_map)
        except Exception as e:
            log.warning(f"  feature extract failed for {Path(vol_path).name}: {e}")
            n_fail += 1
            continue
        if np.all(np.isnan(feats)):
            n_fail += 1
            continue
        X.append(feats); y.append(PHASE_TO_ID[phase])
        groups.append(case_id); series.append(series_id); sources.append('mask')

    log.info(f"Feature table (dir mode): {len(X)} volumes  "
             f"(no_label={n_no_label}, no_mask={n_no_mask}, failed={n_fail})")
    if X:
        dist = {PHASE_NAMES[i]: int(np.sum(np.array(y) == i)) for i in sorted(set(y))}
        log.info(f"  phase distribution: {dist}  | {len(set(groups))} studies")
    return (np.array(X, dtype=np.float64), np.array(y), np.array(groups),
            np.array(series), np.array(sources))


def _resolve(path: str, data_root: str) -> str:
    p = Path(path)
    if data_root and not p.is_absolute():
        return str(Path(data_root) / p)
    return path


def build_feature_table(df: pd.DataFrame, data_root: str, use_mask_fallback: bool):
    """Return (X, y, groups, series, sources) with one 16-vector per row."""
    organ_map = None
    X, y, groups, series, sources = [], [], [], [], []
    n_pkl = n_mask = n_drop = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building features', unit='case'):
        feats = None
        # 1) preferred: TS statistics pickle
        sp = row.get('stats_path')
        if isinstance(sp, str) and Path(_resolve(sp, data_root)).exists():
            try:
                feats = features_from_stats_pkl(_resolve(sp, data_root))
                src = 'pkl'; n_pkl += 1
            except Exception as e:
                log.warning(f"  stats pkl failed for {row.get('SeriesInstanceUID')}: {e}")

        # 2) fallback: extract from (CT, mask)
        if feats is None and use_mask_fallback:
            ct = _resolve(row.get('orig_volume_path', ''), data_root)
            mk = _resolve(row.get('VolumePath', ''), data_root)
            if Path(ct).exists() and Path(mk).exists():
                if organ_map is None:
                    organ_map = load_organ_label_map()
                try:
                    import nibabel as nib
                    ctv = nib.load(ct).get_fdata().astype(np.float32)
                    mkv = nib.load(mk).get_fdata()
                    feats = features_from_mask(ctv, mkv, organ_map)
                    src = 'mask'; n_mask += 1
                except Exception as e:
                    log.warning(f"  mask extract failed for {row.get('SeriesInstanceUID')}: {e}")

        if feats is None or np.all(np.isnan(feats)):
            n_drop += 1
            continue

        X.append(feats)
        y.append(int(row['ct_phase']))
        groups.append(str(row['StudyInstanceUID']))
        series.append(str(row['SeriesInstanceUID']))
        sources.append(src)

    log.info(f"Feature table: {len(X)} cases  (from pkl={n_pkl}, from mask={n_mask}, dropped={n_drop})")
    return (np.array(X, dtype=np.float64), np.array(y), np.array(groups),
            np.array(series), np.array(sources))


def main():
    ap = argparse.ArgumentParser(description='Re-train CT phase XGBoost with fresh weights + features')
    # --- mode A: synthesis deeds-aligned directory (recommended for this repo) ---
    ap.add_argument('--data_dir', default='',
                    help='deeds-aligned data dir (per-case subdirs with *_deeds.nii.gz '
                         '+ *_deeds_seg_reg.nii.gz). If set, uses directory discovery '
                         'instead of the metadata CSV.')
    ap.add_argument('--labels_csv', default='', help='labels.csv (StudyInstanceUID/SeriesInstanceUID/Label)')
    ap.add_argument('--file_tag', default='_deeds')
    ap.add_argument('--seg_suffix', default='_seg_reg')
    ap.add_argument('--inspect_only', action='store_true',
                    help='just print the label ids present in the first mask found '
                         '(verify the multi-label organ scheme before a full run)')
    # --- mode B: original metadata CSV ---
    ap.add_argument('--metadata_csv', default='vindr_nifti_metadata.csv')
    ap.add_argument('--data_root', default='', help='prefix for relative paths in the metadata CSV')
    ap.add_argument('--no_mask_fallback', action='store_true',
                    help='use only stats .pkl features (no mask-based extraction)')
    # --- shared ---
    ap.add_argument('--out_weights', default='xgb_vindr_retrained.pkl')
    ap.add_argument('--out_dir', default='retrain_out')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # Verify the mask label scheme on one real mask, then exit.
    if args.inspect_only:
        from organ_features import inspect_mask_labels, load_organ_label_map
        one = next(Path(args.data_dir).rglob(f'*{args.file_tag}{args.seg_suffix}.nii.gz'), None)
        if one is None:
            raise FileNotFoundError(f"no *{args.file_tag}{args.seg_suffix}.nii.gz under {args.data_dir}")
        log.info(f"Inspecting {one}")
        inspect_mask_labels(str(one), load_organ_label_map())
        return

    import xgboost as xgb
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Dump the resolved organ→label-id map — the single source of truth the
    # synthesis per-organ metrics reuse (so their organ NAMES match this model's,
    # with no hardcoded / drift-prone ids). Best-effort: needs TS importable.
    try:
        (out / 'organ_label_map.json').write_text(json.dumps(load_organ_label_map(), indent=2))
        log.info(f"Wrote organ_label_map.json → {out}/")
    except Exception as e:
        log.warning(f"could not dump organ_label_map.json ({e}); synthesis per-organ "
                    f"metrics will fall back to raw label ids")

    if args.data_dir:
        X, y, groups, series, sources = build_feature_table_from_dir(
            args.data_dir, args.labels_csv, args.file_tag, args.seg_suffix)
    else:
        import pandas as pd
        df = pd.read_csv(args.metadata_csv)
        log.info(f"Loaded metadata: {len(df)} rows, {df['StudyInstanceUID'].nunique()} studies")
        X, y, groups, series, sources = build_feature_table(
            df, args.data_root, use_mask_fallback=not args.no_mask_fallback)
    if len(X) == 0:
        raise RuntimeError("No usable feature rows — check --data_dir/--labels_csv "
                           "(dir mode) or stats_path/mask paths/--data_root (csv mode)")

    # Persist the complete feature table.
    import pandas as pd
    np.savez(out / 'features.npz', X=X, y=y, groups=groups, series=series, sources=sources,
             organs=np.array(ORGANS))
    pd.DataFrame(X, columns=ORGANS).assign(
        ct_phase=y, StudyInstanceUID=groups, SeriesInstanceUID=series, source=sources
    ).to_csv(out / 'features.csv', index=False)

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_pred = np.full(len(y), -1)
    all_models, fold_reports = [], []

    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        model = xgb.XGBClassifier(
            learning_rate=0.05, max_depth=4, n_estimators=200, n_jobs=-1,
            eval_metric='mlogloss' if len(np.unique(y[tr])) > 2 else 'logloss',
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        oof_pred[te] = pred
        acc = accuracy_score(y[te], pred)
        log.info(f"fold {fold+1}/{args.n_splits}: test acc={acc:.4f} ({len(te)} cases)")
        fold_reports.append(classification_report(y[te], pred, output_dict=True, zero_division=0))
        all_models.append({'fold': fold, 'model': model})

    # Leakage-free overall metrics from out-of-fold predictions.
    labels = sorted(np.unique(y).tolist())
    names = [PHASE_NAMES.get(i, str(i)) for i in labels]
    oof_acc = accuracy_score(y, oof_pred)
    oof_conf = confusion_matrix(y, oof_pred, labels=labels).tolist()
    oof_report = classification_report(y, oof_pred, labels=labels, target_names=names,
                                       zero_division=0, output_dict=True)
    log.info(f"Out-of-fold overall accuracy: {oof_acc:.4f}")
    log.info(f"OOF confusion ({names}):")
    for r in oof_conf:
        log.info(f"    {r}")

    # --- Per-organ report (with organ names) ---
    # (a) Which organs drive the classifier: mean feature importance across folds,
    #     normalised to sum 1, mapped to organ names.
    imp = np.mean([m['model'].feature_importances_ for m in all_models], axis=0)
    imp = imp / (imp.sum() + 1e-12)
    organ_importance = {o: float(v) for o, v in zip(ORGANS, imp)}
    # (b) Descriptive per-organ median HU per phase (nan-aware) — shows the actual
    #     enhancement signal, e.g. aorta bright in arterial vs venous.
    import warnings
    organ_hu_by_phase = {}
    for pid in labels:
        rows = X[y == pid]
        with warnings.catch_warnings():          # nanmedian warns on all-NaN organ cols
            warnings.simplefilter('ignore', category=RuntimeWarning)
            med = np.nanmedian(rows, axis=0) if len(rows) else np.full(len(ORGANS), np.nan)
        organ_hu_by_phase[PHASE_NAMES.get(pid, str(pid))] = {
            o: (None if np.isnan(v) else float(v)) for o, v in zip(ORGANS, med)}
    # (c) Per-organ OOF coverage: how many volumes actually had each organ
    #     segmented (a low count means that organ's feature is mostly NaN).
    organ_coverage = {o: int(np.sum(~np.isnan(X[:, j]))) for j, o in enumerate(ORGANS)}

    log.info("Per-organ importance (mean gain across folds, top→bottom):")
    for o, v in sorted(organ_importance.items(), key=lambda kv: -kv[1]):
        hu = "  ".join(f"{ph[:3]}={organ_hu_by_phase[ph][o]:.0f}"
                       if organ_hu_by_phase[ph][o] is not None else f"{ph[:3]}=NA"
                       for ph in organ_hu_by_phase)
        log.info(f"    {o:32s} imp={v:.3f}  cov={organ_coverage[o]:3d}/{len(y)}  medianHU[{hu}]")

    with open(args.out_weights, 'wb') as f:
        pickle.dump(all_models, f)
    (out / 'metrics.json').write_text(json.dumps({
        'n_cases': int(len(y)), 'n_studies': int(len(set(groups.tolist()))),
        'feature_sources': {s: int((sources == s).sum()) for s in set(sources.tolist())},
        'oof_accuracy': float(oof_acc), 'oof_confusion': oof_conf,
        'oof_report': oof_report, 'fold_reports': fold_reports,
        'organs': ORGANS,
        'organ_importance': organ_importance,
        'organ_hu_by_phase': organ_hu_by_phase,
        'organ_coverage': organ_coverage,
    }, indent=2))
    # A compact standalone per-organ CSV report for quick reading.
    import pandas as pd
    rep = pd.DataFrame({'organ': ORGANS,
                        'importance': [organ_importance[o] for o in ORGANS],
                        'coverage': [organ_coverage[o] for o in ORGANS]})
    for ph in organ_hu_by_phase:
        rep[f'medianHU_{ph}'] = [organ_hu_by_phase[ph][o] for o in ORGANS]
    rep.sort_values('importance', ascending=False).to_csv(out / 'organ_report.csv', index=False)
    log.info(f"Saved weights → {args.out_weights}; features + metrics + organ_report.csv → {out}/")


if __name__ == '__main__':
    main()
