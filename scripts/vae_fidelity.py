#!/usr/bin/env python
"""
Gate D1 — the autoencoder ceiling for any latent-diffusion route (DIFFUSION_PLAN.md §4).

Stable Diffusion operates in a VAE latent space: 8x spatial downsampling, 4 latent
channels, trained on 8-bit RGB natural images. This round-trips the REAL CECTs
through that VAE — no diffusion, no training, no model of ours at all — and scores
the reconstruction with the normal benchmark suite.

WHY IT COMES FIRST. It is a CEILING measurement: no latent-diffusion method can
beat its own autoencoder. If the VAE alone already loses more HU fidelity than the
deterministic baseline's whole error, every SD / ControlNet / latent-diffusion
route is dead before any of them is tried, and the pixel-space recommendation in
§5 stops being a preference and becomes a measurement.

WHAT TO EXPECT, so the result is interpretable either way (§4):
  featHU, org_mae   MAY SURVIVE. An organ's MEDIAN HU is a low-frequency
                    statistic and VAEs preserve low frequencies well.
  raps_hf, grad_w1  LIKELY FAIL. Fine parenchymal texture at 8x downsampling, in
                    a domain the VAE has never seen.
Those two can diverge, so score the WHOLE suite, not just featHU.

DECISION RULE
  VAE featHU > 13.5 HU (the deterministic best)
      -> every SD/ControlNet route is dead on arrival. Go pixel-space.
  VAE featHU < ~8 HU but the texture metrics are ruined
      -> latent diffusion is viable ONLY if optimising HU fidelity over texture.
  both good
      -> SD fine-tuning becomes worth considering and §5's recommendation changes.

Report the number either way. "We measured the autoencoder ceiling before
committing" is a methods paragraph; not measuring it is a reviewer's question.

OFFLINE OPERATION
The training host has no internet, so nothing here touches the HuggingFace hub.
`--vae_path` is a LOCAL directory containing the sd-vae-ft-mse config and weights,
staged separately (see vendor/README.md) and rsynced across. HF_HUB_OFFLINE is set
before diffusers is imported so an accidental network call fails loudly instead of
hanging on a socket timeout.

Usage:
    python scripts/vae_fidelity.py \
        --vae_path vendor/sd-vae-ft-mse \
        --split test \
        --out_dir ../out_synthesis_train/vae_ceiling

    python orgFeatXGB_CTPhase/phase_eval.py \
        --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
        --manifest ../out_synthesis_train/vae_ceiling/manifest.csv --gen_in_hu \
        --out_json ../out_synthesis_train/vae_ceiling/phase_eval_report.json
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np

# Set BEFORE diffusers/huggingface_hub is imported — they read it at import time.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import train_config
from dataset import _load_vol, find_pairs_and_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def load_vae(vae_path: str, device: str):
    """Load `AutoencoderKL` from a LOCAL directory.

    Fails with an actionable message rather than a stack trace: on the intended
    (offline) host the two realistic failures are "diffusers is not installed"
    and "the weights were never rsynced", and both are fixed by the staging
    instructions rather than by anything in this file.
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        raise SystemExit(
            "vae_fidelity.py needs `diffusers`, which is not installed.\n"
            "This host has no internet, so install it from the staged wheels:\n"
            "    pip install --no-index --find-links vendor/wheels diffusers\n"
            "See vendor/README.md. Nothing else in this repo needs diffusers — "
            "Gate D1 is the only consumer.")
    p = Path(vae_path)
    if not p.exists():
        raise SystemExit(
            f"VAE weights not found at {p}.\n"
            f"They are staged on a machine with internet and rsynced here; see "
            f"vendor/README.md. Do not pass a hub id — this host is offline.")
    import torch
    vae = AutoencoderKL.from_pretrained(str(p), local_files_only=True).to(device)
    vae.eval()
    n = sum(x.numel() for x in vae.parameters()) / 1e6
    log.info(f"Loaded VAE from {p} ({n:.1f}M params, scale factor "
             f"{2 ** (len(vae.config.block_out_channels) - 1)}x)")
    return vae


def roundtrip_volume(vae, vol_dhw: np.ndarray, hu_min: float, hu_max: float,
                     device: str, batch_size: int = 8) -> np.ndarray:
    """Encode/decode every axial slice and reassemble the volume, in HU.

    Three conversions, each forced by what the VAE expects:
      * HU -> [-1,1] via the SAME window the models train on, so the ceiling is
        measured in the domain everything else is scored in.
      * 1 channel -> 3, by replication. The VAE's first conv takes RGB; feeding a
        grey image as three identical channels is the standard adaptation and is
        what any latent-diffusion route here would have to do too.
      * 3 channels -> 1 on the way out, by MEAN rather than by taking the red
        channel. The decoder does not reproduce the replication exactly, and the
        mean is the minimum-variance estimator of the value that was encoded.

    `latent_dist.mode()` rather than `.sample()`: this measures the ceiling of the
    autoencoder, and adding the encoder's own sampling noise would measure
    something strictly worse than what a latent diffusion model would use.

    Slices are padded to a multiple of 8 and cropped back, because the 8x
    downsampling requires it and silently resizing would change the geometry the
    organ masks are defined on.
    """
    import torch
    D, H, W = vol_dhw.shape
    span = hu_max - hu_min
    x = np.clip(vol_dhw, hu_min, hu_max)
    x = ((x - hu_min) / span) * 2.0 - 1.0                    # -> [-1,1]

    ph = (-H) % 8
    pw = (-W) % 8
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode='edge')

    out = np.empty_like(x)
    with torch.no_grad():
        for i in range(0, D, batch_size):
            sl = torch.from_numpy(x[i:i + batch_size]).float().to(device)
            sl = sl.unsqueeze(1).repeat(1, 3, 1, 1)          # (B,3,H,W)
            lat = vae.encode(sl).latent_dist.mode()
            rec = vae.decode(lat).sample
            out[i:i + batch_size] = rec.mean(dim=1).float().cpu().numpy()

    out = out[:, :H, :W]
    return ((np.clip(out, -1.0, 1.0) + 1.0) / 2.0) * span + hu_min


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vae_path', default='vendor/sd-vae-ft-mse',
                    help='LOCAL directory with the VAE config + weights. Never a '
                         'hub id: the training host is offline.')
    ap.add_argument('--split', default='test', choices=['val', 'test', 'both'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    import torch
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = dict(train_config)
    hu_min, hu_max = float(cfg['hu_min']), float(cfg['hu_max'])
    _, val_pairs, test_pairs = find_pairs_and_split(cfg)
    pairs = {'val': val_pairs, 'test': test_pairs,
             'both': val_pairs + test_pairs}[a.split]

    vae = load_vae(a.vae_path, device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    rows: List[dict] = []
    for pair in pairs:
        case_id = pair['case_id']
        seg = pair.get('seg_path')
        if not seg:
            log.warning(f"  {case_id}: no mask — skipping (can't phase-score)")
            continue
        # The REAL CECT is the input here, not the NCCT: this measures what the
        # autoencoder loses, with no synthesis involved at all.
        real_nii = nib.load(pair['target_path'])
        real_dhw = _load_vol(pair['target_path'])
        rec = roundtrip_volume(vae, real_dhw, hu_min, hu_max, device, a.batch_size)
        gen_path = out / f'{case_id}_syn.nii.gz'
        nib.save(nib.Nifti1Image(np.transpose(rec, (2, 1, 0)).astype(np.float32),
                                 real_nii.affine, real_nii.header), str(gen_path))
        rows.append({'gen_path': str(gen_path), 'real_path': pair['target_path'],
                     'mask_path': seg,
                     'target_phase': pair.get('phase', cfg['target_phase'])})
        log.info(f"  {case_id}: round-tripped {real_dhw.shape}")

    manifest = out / 'manifest.csv'
    with open(manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path', 'mask_path',
                                          'target_phase'])
        w.writeheader(); w.writerows(rows)

    # The manifest has no tiling geometry, so benchmark.py will report seam=NaN.
    # That is correct: nothing here was tiled, and a seam number would be
    # meaningless. Recorded so the NaN is not read as a bug.
    (out / 'run_config.json').write_text(json.dumps({
        'model': 'sd-vae-ft-mse round-trip (Gate D1)',
        'vae_path': str(a.vae_path), 'split': a.split,
        'hu_min': hu_min, 'hu_max': hu_max,
        'patch_size': None, 'overlap': None,
        'note': 'ceiling measurement, not a synthesis model — input is the REAL '
                'CECT. seam is NaN because nothing was tiled.',
    }, indent=2))

    log.info(f"Wrote {len(rows)} rows → {manifest}")
    log.info("Next, and read BOTH:")
    log.info(f"  python orgFeatXGB_CTPhase/phase_eval.py --weights "
             f"orgFeatXGB_CTPhase/xgb_vindr_full.pkl --manifest {manifest} "
             f"--gen_in_hu --hu_min {hu_min} --hu_max {hu_max} "
             f"--out_json {out/'phase_eval_report.json'}")
    log.info(f"  python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl "
             f"--manifest {manifest} --out analysis/vae_ceiling")
    log.info("Decision rule: featHU > 13.5 HU kills every latent-diffusion route "
             "(DIFFUSION_PLAN.md §4). Report the number either way.")


if __name__ == '__main__':
    main()
