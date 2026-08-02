"""
Save a FULL multi-label TotalSegmentator mask for one CT volume.

The existing `*_seg_reg.nii.gz` masks in the synthesis data are missing the
aorta / heart / IVC (see retrain_out coverage analysis) — exactly the vessels
that separate arterial from venous. This regenerates a COMPLETE `total`-task
multi-label mask (all ~117 structures incl. aorta=52) on the volume's own grid,
for use by both the XGBoost phase model and the synthesis organ losses/metrics.

Same TotalSegmentator python-API call as `ts_get_stats.py` (proven in this env),
but saves the segmentation image instead of the per-organ stats:
    totalsegmentator(ct, None, ml=True, fast=..., roi_subset=None) -> seg image

Usage:
    python ts_get_mask.py <input_ct.nii.gz> <output_mask.nii.gz> [--full]
      --full : full-resolution TS (slower, better vessels). Default = fast (3mm),
               matching the published model's `fast=True`.
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


def get_mask(input_path: str, output_path: str, fast: bool = True):
    ct_img = nib.load(input_path)
    seg = totalsegmentator(
        ct_img, None, ml=True, fast=fast, roi_subset=None, quiet=True,
    )
    # statistics=False → returns just the seg image; be defensive if a tuple.
    if isinstance(seg, (tuple, list)):
        seg = seg[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(seg, output_path)


def main():
    ap = argparse.ArgumentParser(description="Save a full multi-label TotalSegmentator mask")
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    ap.add_argument("--full", action="store_true",
                    help="full-resolution TS (default: fast/3mm, matches the model)")
    args = ap.parse_args()

    if Path(args.output_path).exists():
        print(f"skip (exists): {args.output_path}")
        return
    try:
        get_mask(args.input_path, args.output_path, fast=not args.full)
        print(f"ok: {args.output_path}")
    except Exception as e:
        print(f"FAILED {args.input_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
