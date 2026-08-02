#!/usr/bin/env python
"""
Dump the FULL TotalSegmentator `total` label map as {organ_name: label_id} JSON.

Why this exists separately from `organ_features.load_organ_label_map()`: that
function deliberately returns only the 16 `ORGANS` the XGBoost phase model was
trained on, in the exact feature order the model expects — it must not grow.
The synthesis side needs *all* 117 classes, both to name per-organ metrics
(otherwise they print as `label_<id>`) and to build the per-organ loss weight
LUT in config.ORGAN_WEIGHTS.

Usage:
    python orgFeatXGB_CTPhase/dump_ts_label_map.py \
        --out orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json
"""

import argparse
import json
from pathlib import Path

_DEFAULT_OUT = Path(__file__).resolve().parent / 'retrain_out_full' / 'ts_label_map_total.json'


def dump(out_path: Path) -> dict:
    """Write {name: id} for TotalSegmentator's `total` task. Returns the map."""
    try:
        from totalsegmentator.map_to_binary import class_map
    except Exception as e:                                   # pragma: no cover
        raise RuntimeError(
            f"Could not import TotalSegmentator to resolve label ids ({e}). "
            "Run this on a machine with totalsegmentator installed — the JSON "
            "it writes is portable, so it only has to be generated once."
        )
    name_to_id = {str(name): int(i) for i, name in class_map['total'].items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(name_to_id, indent=2, sort_keys=True))
    return name_to_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    m = dump(args.out)
    print(f"wrote {len(m)} labels -> {args.out}")

    # Sanity-print the ids the synthesis weighting actually keys on, so a
    # TS-version mismatch is caught here rather than silently mis-weighting.
    id_to_name = {v: k for k, v in m.items()}
    for group, ids in {
        'phase-critical vessels': [52, 63, 64, 65, 66, 67, 68, 53],
        'heart / solid organs':   [51, 5, 7, 1, 2, 3, 4],
        'zero-weighted GI':       [6, 18, 19, 20, 15],
    }.items():
        print(f"  {group}:")
        for i in ids:
            print(f"    {i:4d}  {id_to_name.get(i, '<<< MISSING >>>')}")


if __name__ == '__main__':
    main()
