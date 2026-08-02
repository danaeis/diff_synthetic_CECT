# Re-training the phase model + using it as a synthesis phase-fidelity metric

New files added to this repo (all self-contained, no changes to the original
`train.py` / `totalseg_get_phase.py`):

| file | purpose |
|---|---|
| `organ_features.py` | Extract the 16 per-organ median-HU features from a TS stats `.pkl` **or** directly from a (CT, multi-label mask) pair. Organ→label-id map is read from the installed TotalSegmentator at runtime (no hardcoded ids). |
| `retrain_xgb.py` | Re-train the XGBoost ensemble on the **full** dataset; saves fresh weights + the complete feature table + a leakage-free out-of-fold confusion matrix. |
| `phase_eval.py` | Score a synthesised CECT vs the real CECT — feature-level (per-organ HU error) and classification-level (does it classify as the target phase?). |
| `smoke_test_xgb.py` | CPU-only logic test (no xgboost/TS/data needed). |

---

## Step 0 — verify the mask label scheme (once)

The mask-based paths assume your `_seg*.nii.gz` masks are standard multi-label
TotalSegmentator output. Confirm the organ ids resolve on one real mask:

```python
python -c "
from organ_features import load_organ_label_map, inspect_mask_labels
m = load_organ_label_map()                      # from installed TotalSegmentator
inspect_mask_labels('../../ncct_cect/vindr_ds/segmentation_masks/<one>_seg.nii.gz', m)
"
```
It prints which organ labels are present. If it errors that TotalSegmentator
isn't importable, install it (`pip install TotalSegmentator`) or pass a verified
explicit `{organ: id}` map.

---

## Step 1 — re-train the model (fresh weights + complete features)

### Mode A (recommended here): the synthesis deeds-aligned directory

Retrains directly on the SAME data the synthesis pipeline uses — per-case subdirs
with `*_deeds.nii.gz` volumes + co-registered `*_deeds_seg_reg.nii.gz` masks —
discovering one labelled sample per unique volume (no generation pairs, no path
dependence on the old `vindr_nifti_metadata.csv`). This is the fix for the
"0 cases … dropped=334" error (that error means the old CSV's paths don't exist
in your layout).

First verify the mask label scheme resolves (Step 0 above, or):
```bash
python retrain_xgb.py --data_dir /abs/path/to/B2_deeds__aligned --inspect_only
```
Then train:
```bash
python retrain_xgb.py \
    --data_dir   /abs/path/to/.../vindr_ds/all_baseline_algorithms/B2_deeds__aligned \
    --labels_csv /abs/path/to/.../vindr_ds/labels.csv \
    --file_tag   _deeds \
    --seg_suffix _seg_reg \
    --out_weights xgb_vindr_retrained.pkl \
    --out_dir     retrain_out
```
(Phase comes from `labels.csv` by StudyInstanceUID/SeriesInstanceUID/Label, with a
filename-keyword fallback — same resolution as the synthesis `dataset.py`.)

### Mode B: the original metadata CSV

Uses `vindr_nifti_metadata.csv` (`stats_path`, `orig_volume_path`, `VolumePath`
mask, `ct_phase`, `StudyInstanceUID`). Prefers the TS stats `.pkl` per case; falls
back to (CT, mask). `--data_root` prefixes the CSV's relative paths:

```bash
python retrain_xgb.py \
    --metadata_csv vindr_nifti_metadata.csv \
    --data_root    /media/disk1/saeedeh_danaei/ncct_cect/vindr_ds/ \
    --out_weights  xgb_vindr_retrained.pkl \
    --out_dir      retrain_out
```

Outputs:
- `xgb_vindr_retrained.pkl` — fresh ensemble (same `[{"fold","model"}]` format the
  inference/eval code expects).
- `retrain_out/features.{npz,csv}` — the complete 16-organ feature table (X, y,
  StudyInstanceUID, SeriesInstanceUID, feature source per case).
- `retrain_out/metrics.json` — per-fold reports **and** an out-of-fold overall
  accuracy + confusion matrix (the honest, leakage-free headline number to trust
  over the old `xgb_vindr.pkl`).

The log prints how many cases came from stats-pkl vs mask extraction and how many
were dropped — check that the case count matches your expected dataset size
(that's the "completeness" concern about the old weights).

---

## Step 2 — use it as a synthesis phase-fidelity metric (the "3a" loss/metric)

For a trained generator, produce full synthesised CECT volumes for your test
cases, then build a manifest CSV:

```csv
gen_path,real_path,mask_path,target_phase
/out/case001_gen.nii.gz,/data/case001_venous.nii.gz,/data/case001_seg_reg.nii.gz,venous
...
```
- `gen_path`: the generated CECT. If it's in the generator's `[0,1]` window space,
  leave de-norm on (default `--hu_min -200 --hu_max 400`, matching `config.py`).
  If you already reconstructed it in HU, add `--no_denorm`.
- `real_path`: the real CECT (original HU).
- `mask_path`: the co-registered multi-label organ mask (same grid as gen & real).
- `target_phase`: the phase the generator was asked to produce (id `0-3` or name).

```bash
python phase_eval.py \
    --weights  xgb_vindr_retrained.pkl \
    --manifest manifest.csv \
    --out_json phase_eval_report.json
```

Headline numbers in the summary:
- **`gen_phase_accuracy_vs_target`** — fraction of generated CECTs the model
  classifies as the intended phase. This is the key "did it make the RIGHT phase?"
  score PSNR/SSIM can't give you — add it as a column to
  `scenario_results_overview.md`.
- **`gen_agreement_with_real`** — fraction where generated & real get the same
  predicted phase.
- **`real_phase_accuracy_vs_target`** — sanity/ceiling: the *real* CECTs should
  score near 1.0; if not, the mask/window/label-map setup needs checking before
  trusting the generated numbers.
- **`mean_feature_l1_hu`** + **`per_organ_mean_abs_err_hu`** — per-organ HU error
  between generated and real (which organs the generator under/over-enhances).

---

## Notes / limitations

- **HU clipping (arterial):** generated volumes live in `[hu_min, hu_max]`. If
  `hu_max=400` is below bright arterial-aorta HU (~350–450), that enhancement is
  truncated, so the classification metric is most reliable for venous/non-contrast
  targets and slightly pessimistic for arterial. The per-organ feature error clips
  the real CECT to the same window so that comparison stays fair.
- **Differentiable training loss (the "3b" option)** is intentionally NOT built
  here — XGBoost isn't differentiable. This module is the evaluation metric. The
  differentiable per-organ enhancement-matching loss remains a separate future step.
- Run `python smoke_test_xgb.py` (in this repo's env) any time to confirm the
  feature/eval logic still passes after edits — it needs no weights, TS, or data.
