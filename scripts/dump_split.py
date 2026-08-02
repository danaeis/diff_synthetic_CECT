#!/usr/bin/env python
"""
Dump the canonical case-level train/val/test split to splits/split.json.

This is THE shared split for the benchmark: every external model's data adapter
reads it so all models train and test on identical cases (seed 42). Run once, on
the server where the data lives; the JSON is portable.

Reuses `find_pairs_and_split` (dataset.py) with the project's own config so the
split is bit-for-bit the same one our own runs used.

Usage:
    python scripts/dump_split.py                       # uses config.train_config
    python scripts/dump_split.py --seg_suffix _seg_full --out splits/split.json
"""

import argparse
import json
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import train_config
from dataset import find_pairs_and_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('splits/split.json'))
    ap.add_argument('--seg_suffix', default=None, help='override cfg seg_suffix')
    args = ap.parse_args()

    cfg = dict(train_config)
    if args.seg_suffix:
        cfg['seg_suffix'] = args.seg_suffix

    train, val, test = find_pairs_and_split(cfg)

    def rows(pairs):
        # Emit everything a downstream adapter needs to locate each case's files.
        return [{'case_id': p['case_id'],
                 'ncct': p['source_path'],
                 'cect': p['target_path'],
                 'seg':  p.get('seg_path')} for p in pairs]

    split = {
        'seed': cfg.get('seed', 42),
        'target_phase': cfg.get('target_phase', 'venous'),
        'seg_suffix': cfg.get('seg_suffix'),
        'hu_min': cfg.get('hu_min', -200.0), 'hu_max': cfg.get('hu_max', 400.0),
        'data_dir': str(cfg.get('data_dir')),
        'counts': {'train': len(train), 'val': len(val), 'test': len(test)},
        'train': rows(train), 'val': rows(val), 'test': rows(test),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2))
    print(f"[written] {args.out}  "
          f"(train={len(train)} val={len(val)} test={len(test)}, seed={split['seed']})")
    print("Every external model's adapter must read THIS file for its split.")


if __name__ == '__main__':
    main()
