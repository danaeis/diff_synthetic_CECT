"""
Entry point for the literature-baseline NCCT → CECT training run.

Usage:
    cd train/literature_baseline

    # Baseline run (2-D, pix2pixHD losses)
    python train.py

    # Switch to 3-D patches (edit config.py: PATCH_DEPTH=8, DIMS=3, PATCH_SIZE=96)
    # or override from CLI:
    python train.py --patch_depth 8 --dims 3 --patch_size 96

    # Ablation: add extra losses
    python train.py --use_ssim --use_gradient

    # Heteroscedastic calibration baseline (DIFFUSION_PLAN.md §6 step 2)
    python train.py --use_hetero --use_organ --use_per_organ_weights

    # Conditional diffusion (§6 steps 4-5)
    python train.py --use_diffusion --parameterisation v --generator_norm group

    # Resume from latest checkpoint
    python train.py --resume
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch

import random
import numpy as np

from config import train_config, resolve_organ_weights
from dataset import build_loaders
from trainer import Trainer
from trainer_diffusion import DiffusionTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
log = logging.getLogger(__name__)

def set_seed(s: int, deterministic: bool = True):
    """Seed every RNG that touches training.

    Note what is NOT seeded elsewhere: dataset.py seeds only numpy, via
    `data_seed`, and that covers the split and patch selection. Weight init,
    dropout and DataLoader shuffle all draw from torch's global RNG, which was
    previously unseeded — so no two runs of the same scenario were comparable.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"Seed       : {s} (deterministic={deterministic})")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description='Literature baseline NCCT→CECT')
    # Resume is now the default: if the output dir already holds a checkpoint we
    # continue from it. --resume is kept for backwards compatibility (no-op),
    # and --fresh forces a clean start that ignores any existing checkpoint.
    p.add_argument('--resume',      action='store_true',
                   help='(default) continue from the latest checkpoint if one exists')
    p.add_argument('--fresh',       action='store_true',
                   help='ignore any existing checkpoint and start training from scratch')
    p.add_argument('--device',      type=str,   default=None,
                   help="override device, e.g. 'cuda', 'cuda:0', 'cpu'")
    
    p.add_argument('--seed', type=int, default=None,
                   help='weight init / dropout / batch order. Vary this for seed replicates.')
    p.add_argument('--data_seed', type=int, default=None,
                   help='split + patch selection + cache key. Do NOT vary.')
    p.add_argument('--nondeterministic', action='store_true',
                   help='allow cudnn.benchmark (faster, not bit-reproducible)')

    p.add_argument('--epochs',      type=int,   default=None)
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--output_dir',  type=str,   default=None)
    p.add_argument('--patch_size',  type=int,   default=None)
    p.add_argument('--patch_depth', type=int,   default=None,
                   help='1=2D (default), >1=3D (must also set --dims 3)')
    p.add_argument('--dims',        type=int,   default=None, choices=[2, 3])

    # ── Capacity probes (Gate B of the roadmap) ──────────────────────────────
    # These exist to answer "can this model fit its own training data at all?",
    # which the 45-epoch runs leave open: raw train L1 flattens at ~0.0139 (≈8.3
    # HU) against a val MAE of 0.0148 — a 6% gap, so there is no overfitting and
    # the ceiling is capacity, optimisation, or the data itself.
    p.add_argument('--base_ch', type=int, default=None,
                   help='generator base channels (config GEN_BASE_CH=64). 32≈3.3M, '
                        '64≈13.3M, 96≈30.0M params. 96 is the parameter-matched '
                        'control for any attention claim.')
    p.add_argument('--dropout', type=float, default=None,
                   help='generator dropout (config GEN_DROPOUT=0.2). Set 0.0 for the '
                        'deliberate-overfit probe.')
    p.add_argument('--generator_norm', type=str, default=None,
                   choices=['instance', 'group', 'batch'],
                   help="generator norm (config GEN_NORM='instance'). A TILING "
                        'decision: instance/group statistics span the patch, so a '
                        "tile's own content shifts every output voxel — the seam "
                        'cause (config.py, and norm_attribution.py / erf.py in '
                        '../synthetic_CECT/scripts).')
    p.add_argument('--max_train_cases', type=int, default=None,
                   help='cap TRAIN cases (val/test untouched, so the split stays '
                        'comparable). Use with --dropout 0 and L1 only: if train MAE '
                        'does not reach <2 HU on ~5 cases, capacity/optimisation is '
                        'the bottleneck, not data.')
    p.add_argument('--no_early_stop', action='store_true',
                   help='disable early stopping (probe runs must reach the plateau)')

    # ── Phase conditioning (multi-phase training) ────────────────────────────
    p.add_argument('--target_phases', type=str, nargs='+', default=None,
                   help='phases to synthesise, e.g. --target_phases venous arterial. '
                        'The FIRST decides case membership, so the train/val/test '
                        'split matches a single-phase run on that phase.')
    p.add_argument('--phase_cond_dim', type=int, default=None,
                   help='width of the conditioning embedding (default 64)')
    p.add_argument('--cond_organs', type=str, nargs='*', default=None,
                   help='organs whose median HU conditions the decoder, e.g. '
                        '--cond_organs aorta. Needs splits/levels.json from '
                        '../synthetic_CECT/scripts/dump_levels.py. ORACLE information: evaluate with '
                        'infer_volume.py --level_mode and report the oracle-minus-'
                        'population gap, never oracle alone.')
    p.add_argument('--n_input_slices', type=int, default=None,
                   help='2.5-D: odd number of adjacent axial slices fed as channels '
                        '(1=plain 2-D, 5=k2 gives 7.5mm of z, 11=k5 gives 16.5mm). '
                        'Sets in_channels automatically.')

    # ── Diffusion (DIFFUSION_PLAN.md §5) ─────────────────────────────────────
    p.add_argument('--parameterisation', type=str, default=None, choices=['v', 'x0'],
                   help="what the network regresses. 'v' (default) is well "
                        "conditioned at every noise level; 'x0' couples most "
                        "directly to the organ losses. NOT eps — see config.py.")
    p.add_argument('--diffusion_steps', type=int, default=None,
                   help='length of the training noise schedule (default 1000). '
                        'Sampling step count is an INFERENCE flag '
                        '(infer_volume.py --ddim_steps), not this.')
    p.add_argument('--cfg_drop_prob', type=float, default=None,
                   help='probability of dropping the NCCT condition during '
                        'training, enabling classifier-free guidance at sampling '
                        'time. 0 disables guidance entirely.')
    p.add_argument('--aux_max_t', type=int, default=None,
                   help='apply the organ/HU-profile losses to the predicted x0 '
                        'only below this timestep (default 700 of 1000).')
    p.add_argument('--diffusion_sample_every', type=int, default=None,
                   help='epochs between DDIM sampling passes on val patches. '
                        'These are for looking at; selection uses the '
                        'deterministic denoising loss regardless.')
    p.add_argument('--lambda_nll', type=float, default=None,
                   help='weight of the Gaussian NLL term (--use_hetero)')

    # Organ-weighting / curriculum knobs (see config.py for the rationale)
    p.add_argument('--selection_metric', type=str, default=None,
                   choices=['val_org_ssim', 'val_ssim', 'val_loss'],
                   help='metric picking best_model.pth (default val_org_ssim)')
    p.add_argument('--organ_weight_preset', type=str, default=None,
                   choices=['tiered', 'gi_zero'],
                   help="'tiered' full scheme | 'gi_zero' control (GI excluded, rest 1.0)")
    p.add_argument('--sample_mode', type=str, default=None, choices=['random', 'fixed'],
                   help='val patches in the per-epoch sample grid (default random)')
    p.add_argument('--sample_n',    type=int, default=None,
                   help='rows in the sample grid (default 4)')
    p.add_argument('--lambda_organ',       type=float, default=None)
    p.add_argument('--lambda_hu_profile',  type=float, default=None)
    p.add_argument('--lambda_l1_floor',    type=float, default=None)
    p.add_argument('--l1_decay_start_epoch', type=int, default=None)
    p.add_argument('--l1_decay_end_epoch',   type=int, default=None)

    # Loss flags: --use_X / --no_X
    for flag in ['ssim', 'gradient', 'frequency', 'organ',
                 'l1_decay', 'per_organ_weights', 'hu_profile', 'phase_cond',
                 'hetero', 'diffusion']:
        g = p.add_mutually_exclusive_group()
        g.add_argument(f'--use_{flag}',  dest=f'use_{flag}', action='store_true', default=None)
        g.add_argument(f'--no_{flag}',   dest=f'use_{flag}', action='store_false')
    return p.parse_args()


def _apply(cfg: dict, args) -> dict:
    c = cfg.copy()
    if args.device      is not None: c['device']       = args.device
    if args.epochs      is not None: c['epochs']       = args.epochs
    if args.batch_size  is not None: c['batch_size']   = args.batch_size
    if args.lr          is not None: c['learning_rate']= args.lr
    if args.output_dir  is not None: c['output_dir']   = Path(args.output_dir)
    if args.patch_size  is not None: c['patch_size']   = args.patch_size
    if args.patch_depth is not None: c['patch_depth']  = args.patch_depth
    if args.dims        is not None: c['dims']         = args.dims
    if args.base_ch     is not None: c['generator_base_channels'] = args.base_ch
    if args.dropout     is not None: c['generator_dropout']       = args.dropout
    if args.generator_norm is not None: c['generator_norm']       = args.generator_norm
    # A huge patience is used rather than a separate flag so the trainer's
    # early-stop bookkeeping (and its history channel) stays on one code path.
    if args.no_early_stop:           c['early_stop_patience']     = 10 ** 9
    if args.target_phases is not None:
        c['target_phases'] = args.target_phases
        # Keep the scalar in step: it is what the manifest, the phase evaluator
        # and every existing run_config.json read.
        c['target_phase'] = args.target_phases[0]
    if args.n_input_slices is not None:
        c['n_input_slices'] = args.n_input_slices
        c['in_channels'] = args.n_input_slices     # derived, never set separately
    if args.cond_organs is not None:
        c['cond_organs'] = args.cond_organs
    for k in ['selection_metric', 'sample_mode', 'sample_n', 'max_train_cases',
              'phase_cond_dim',
              'lambda_organ', 'lambda_hu_profile', 'lambda_l1_floor',
              'l1_decay_start_epoch', 'l1_decay_end_epoch',
              'seed', 'data_seed',
              'parameterisation', 'diffusion_steps', 'cfg_drop_prob',
              'aux_max_t', 'diffusion_sample_every', 'lambda_nll']:
        v = getattr(args, k, None)
        if v is not None:
            c[k] = v
    for flag in ['ssim', 'gradient', 'frequency', 'organ',
                 'l1_decay', 'per_organ_weights', 'hu_profile', 'phase_cond',
                 'hetero', 'diffusion']:
        v = getattr(args, f'use_{flag}', None)
        if v is not None:
            c[f'use_{flag}'] = v

    # Per-organ weights are resolved from organ NAMES against the TS label map,
    # so the LUT has to be rebuilt whenever the CLI toggles the flag.
    if args.use_per_organ_weights is not None or args.organ_weight_preset is not None:
        c['organ_weight_preset'] = args.organ_weight_preset or c.get('organ_weight_preset')
        c['organ_weights'] = resolve_organ_weights(
            enabled = args.use_per_organ_weights,
            preset  = args.organ_weight_preset,
        )

    # Auto-derive dims from patch_depth if not explicitly given
    if args.dims is None and 'patch_depth' in c:
        c['dims'] = 3 if c['patch_depth'] > 1 else 2

    return c


def _ckpt_epoch(p: Path) -> int:
    m = re.search(r'ckpt_ep(\d+)', p.name)
    return int(m.group(1)) if m else 0


def _resumable_ckpts(out: Path):
    """Checkpoints newest-first. Sorted by parsed epoch number, not filename, so
    ckpt_ep100 doesn't sort before ckpt_ep99."""
    return sorted(out.glob('ckpt_ep*.pth'), key=_ckpt_epoch, reverse=True)


def _latest_ckpt(out: Path):
    ckpts = _resumable_ckpts(out)
    if not ckpts:
        return None, 0
    return ckpts[0], _ckpt_epoch(ckpts[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = _parse()
    config = _apply(train_config, args)
    out    = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out / 'run_config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    # Add file log
    fh = logging.FileHandler(out / 'train.log', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(fh)

    log.info('=' * 65)
    kind = ('DIFFUSION' if config.get('use_diffusion')
            else 'HETEROSCEDASTIC' if config.get('use_hetero') else 'DETERMINISTIC')
    log.info(f'{kind} NCCT → CECT')
    log.info('=' * 65)
    log.info(f"Output     : {out}")
    log.info(f"Device     : {config['device']}")

    # If we asked for CUDA but it isn't usable, fall back to CPU rather than
    # crashing — and warn loudly, since this is exactly the "silently on CPU"
    # trap that makes an epoch take ~45 min instead of seconds.
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        log.warning("Requested CUDA but torch.cuda.is_available() is False — "
                    "falling back to CPU.")
        config['device'] = 'cpu'
    if config['device'] == 'cpu':
        log.warning('=' * 65)
        log.warning("RUNNING ON CPU — training will be VERY slow.")
        log.warning(f"torch {torch.__version__} | cuda build: {torch.version.cuda} "
                    f"| cuda available: {torch.cuda.is_available()} "
                    f"| visible GPUs: {torch.cuda.device_count()}")
        if torch.version.cuda is None:
            log.warning("This is a CPU-only torch build (torch.version.cuda is None).")
            log.warning("Install a CUDA build in the env, e.g.:")
            log.warning("  pip install --force-reinstall torch "
                        "--index-url https://download.pytorch.org/whl/cu121")
        else:
            log.warning("torch has CUDA support but no GPU is visible — check "
                        "`nvidia-smi`, drivers, and CUDA_VISIBLE_DEVICES.")
        log.warning("Then re-run with --resume (default) to continue from the "
                    "last saved epoch on the GPU.")
        log.warning('=' * 65)
    log.info(f"Dims       : {config.get('dims', 2)}-D  "
             f"(patch_depth={config.get('patch_depth', 1)}, "
             f"patch_size={config.get('patch_size')})")
    log.info(f"Target     : NCCT → {config['target_phase']}")
    log.info(f"HU window  : [{config['hu_min']}, {config['hu_max']}] → [0,1]")
    log.info(f"RAM budget : {config['max_train_patches']:,} train / "
             f"{config['max_val_patches']:,} val patches")
    base = ('MSE(%s)' % config['parameterisation'] if config.get('use_diffusion')
            else 'GaussianNLL' if config.get('use_hetero') else 'L1')
    active = [base] + [f for f in ['ssim', 'gradient', 'frequency',
                                   'organ', 'hu_profile']
                       if config.get(f'use_{f}')]
    log.info(f"Losses     : {' + '.join(active)}")
    log.info(f"Selection  : {config.get('selection_metric', 'val_org_ssim')}")

    if config.get('use_diffusion') and config.get('use_hetero'):
        raise SystemExit(
            "--use_diffusion and --use_hetero are two different answers to the "
            "same question (how to represent p(y|x)) and cannot be combined. "
            "Run them as separate scenarios and compare them — that comparison "
            "IS the result (DIFFUSION_PLAN.md §8).")

    set_seed(int(config.get('seed', 42)),
             deterministic=not args.nondeterministic)
    if config.get('data_seed', 42) != 42:
        log.warning("data_seed != 42 — the train/val/test split has CHANGED. "
                    "Benchmarks against runs with data_seed=42 are invalid.")

    if config.get('use_l1_decay'):
        log.info(f"L1 decay   : {config['lambda_l1']} → {config['lambda_l1_floor']} "
                 f"over epochs {config['l1_decay_start_epoch']}–{config['l1_decay_end_epoch']}")
    ow = config.get('organ_weights')
    if ow:
        zeroed = sorted(k for k, v in ow.items() if v == 0.0)
        log.info(f"Organ wts  : per-organ LUT over {len(ow)} labels "
                 f"({len(zeroed)} zero-weighted: {zeroed})")
    elif config.get('use_organ'):
        log.info(f"Organ wts  : uniform {config.get('organ_weight')}× on masked voxels")
    log.info('')

    # ── Data ────────────────────────────────────────────────────────────────
    log.info("Building loaders …")
    train_loader, val_loader = build_loaders(config)

    if len(train_loader) == 0:
        log.error("No training batches — check data path and HU/patch settings.")
        return

    # ── Train ────────────────────────────────────────────────────────────────
    # One trainer per model family. Diffusion's training step has nothing in
    # common with the deterministic one, so it is a separate class rather than a
    # branch — see trainer_diffusion.py.
    trainer = (DiffusionTrainer if config.get('use_diffusion') else Trainer)(config)
    start   = 0
    ckpt, _ = _latest_ckpt(out)
    if args.fresh:
        if ckpt:
            log.info(f"--fresh: ignoring existing checkpoint {ckpt.name}, "
                     f"starting from scratch")
    elif ckpt:
        # A checkpoint truncated by an interrupted write (or a full disk) raises
        # from torch.load — typically "PytorchStreamReader failed locating file
        # data/N". Rather than abort the whole run, walk back to the next-newest
        # checkpoint: keep_last_n_checkpoints normally leaves 2 older ones, so
        # the cost of a corrupt tail is a few epochs, not the entire scenario.
        start, used = 0, None
        for cand in _resumable_ckpts(out):
            try:
                start = trainer.load_checkpoint(str(cand))
                used = cand
                break
            except Exception as e:
                bad = cand.with_suffix('.pth.corrupt')
                log.warning(f"checkpoint {cand.name} is unreadable ({type(e).__name__}: {e}); "
                            f"renaming to {bad.name} and trying the next-newest")
                try:
                    cand.rename(bad)
                except OSError:
                    pass
        if used is None:
            log.warning("no readable checkpoint in the output dir — starting fresh. "
                        "(Any *.pth.corrupt files are the unreadable ones; a truncated "
                        "write usually means the job was killed mid-save or the disk filled.)")
        else:
            if start >= config['epochs']:
                log.info(f"Checkpoint epoch {start} ≥ target {config['epochs']} epochs — "
                         f"nothing left to train. Use --epochs to extend or --fresh to restart.")
                return
            log.info(f"Auto-resuming from {used.name}: will train epochs "
                     f"{start + 1} → {config['epochs']}")
    else:
        log.info("No checkpoint in output dir — starting fresh.")

    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start)
    except KeyboardInterrupt:
        log.info("Interrupted — saving emergency checkpoint …")
        trainer._save_checkpoint(trainer.current_epoch, is_best=False)

    log.info("Done.")


if __name__ == '__main__':
    main()
