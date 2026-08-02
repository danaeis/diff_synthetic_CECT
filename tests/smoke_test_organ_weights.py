"""
Synthetic smoke test for the per-organ loss weighting + L1 decay curriculum.

Covers the pieces that are easy to get silently wrong:
  1. The weight LUT lands on the right voxels (per label id, not binarised).
  2. A zero-weighted organ receives NO gradient from the organ term.
  3. An all-zero weight scheme raises instead of silently training nothing.
  4. `_l1_w()` holds, ramps, and floors at the configured epochs.
  5. The TRAIN split now yields MULTI-LABEL masks — this is the one that was
     actually broken: mask_multilabel used to be val/test-only, so per-organ
     weights would have silently collapsed onto a single weight.
  6. The composite loss still anchors zero-weighted regions via the L1 floor
     (they must NOT be gradient-free overall, or they can drift into artefacts).

No real data — writes tiny NIfTI triples to a temp dir.
"""

import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import resolve_organ_weights
from dataset import CTPairDataset
from losses import CompositeLoss, OrganWeightedLoss

D, H, W = 4, 256, 256
AORTA, BOWEL = 52, 18            # weight 6.0 and 0.0 under the 'tiered' preset

_ok = True


def check(cond, label):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    _ok &= bool(cond)


# ---------------------------------------------------------------------------
def test_lut_weighting():
    print("1-2. weight LUT + zero-weight gradient exclusion")
    w = resolve_organ_weights(enabled=True, preset='tiered')
    check(w[AORTA] == 6.0 and w[BOWEL] == 0.0, "aorta=6.0, small_bowel=0.0")

    loss = OrganWeightedLoss(organ_weights=w, background_weight=0.5)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :] = AORTA
    mask[..., 2:, :] = BOWEL

    pred = torch.zeros(1, 1, 4, 4)
    tgt = torch.zeros(1, 1, 4, 4); tgt[..., 2:, :] = 1.0     # error only in bowel
    check(float(loss(pred, tgt, mask)) == 0.0, "error confined to bowel → loss 0")

    tgt = torch.zeros(1, 1, 4, 4); tgt[..., :2, :] = 1.0     # error only in aorta
    check(float(loss(pred, tgt, mask)) > 0.0, "error in aorta → loss > 0")

    p = torch.zeros(1, 1, 4, 4, requires_grad=True)
    loss(p, torch.ones(1, 1, 4, 4), mask).backward()
    g = p.grad[0, 0]
    check(float(g[2:].abs().sum()) == 0.0, "no gradient into zero-weighted bowel")
    check(float(g[:2].abs().sum()) > 0.0, "gradient into weighted aorta")

    # A binarised mask must NOT be silently accepted as if it carried labels:
    # every voxel would collapse onto the label-1 weight.
    binm = (mask > 0).float()
    check(float(loss(pred, tgt, binm)) != float(loss(pred, tgt, mask)),
          "binarised mask gives a different (wrong) loss — labels do matter")


def test_all_zero_guard():
    print("3. all-zero weight guard")
    try:
        OrganWeightedLoss(organ_weights={AORTA: 0.0},
                          default_weight=0.0, background_weight=0.0)
        check(False, "all-zero scheme rejected")
    except ValueError:
        check(True, "all-zero scheme rejected")


def test_decay_schedule():
    print("4. L1 decay curriculum")
    cfg = dict(lambda_l1=100.0, use_l1_decay=True,
               l1_decay_start_epoch=10, l1_decay_end_epoch=30, lambda_l1_floor=25.0)
    C = CompositeLoss(cfg)
    got = {}
    for e in (0, 10, 20, 30, 80):
        C.set_epoch(e); got[e] = round(C._l1_w(), 4)
    check(got[0] == 100.0 and got[10] == 100.0, f"holds at 100 through ep10 ({got[0]}, {got[10]})")
    check(got[20] == 62.5, f"midpoint interpolates to 62.5 (got {got[20]})")
    check(got[30] == 25.0 and got[80] == 25.0, f"floors at 25 from ep30 ({got[30]}, {got[80]})")

    C2 = CompositeLoss(dict(cfg, use_l1_decay=False))
    C2.set_epoch(80)
    check(C2._l1_w() == 100.0, "decay disabled → lambda_l1 constant")

    # The regression this used to guard (adversarial pinning lambda_l1 to the
    # reduced value, so the curriculum ran 25→25 and did nothing for 65 epochs)
    # cannot recur: lambda_l1 is no longer parametric on any other loss in this
    # fork — the three terms it backed off for are gone. What CAN still go wrong
    # is the heteroscedastic head, which replaces L1 entirely; the logged
    # lambda_l1 must then be 0 and not the nominal 100, or a hetero run reads as
    # an L1 run in history.json and curves.png.
    C3 = CompositeLoss(dict(cfg, use_hetero=True))
    trace = []
    for e in (0, 10, 20, 30, 40):
        C3.set_epoch(e); trace.append(round(C3._l1_w(), 2))
    check(all(v == 0.0 for v in trace),
          f"heteroscedastic head reports lambda_l1=0, not the nominal 100 "
          f"(got {trace})")


def test_train_split_is_multilabel():
    print("5. train-split masks are MULTI-LABEL when organ_weights is set")
    tmp = tempfile.mkdtemp()
    case = Path(tmp) / 'caseA'; case.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ncct = rng.normal(40, 30, size=(W, H, D)).astype(np.float32)
    seg = np.zeros((W, H, D), np.int16)
    seg[40:120, 40:120, :] = AORTA
    seg[120:200, 40:120, :] = BOWEL
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(ncct, aff), case / 'S_1_deeds.nii.gz')
    nib.save(nib.Nifti1Image(ncct + 60, aff), case / 'S_2_deeds.nii.gz')
    nib.save(nib.Nifti1Image(seg, aff), case / 'S_2_deeds_seg_reg.nii.gz')

    pair = {'source_path': str(case / 'S_1_deeds.nii.gz'),
            'target_path': str(case / 'S_2_deeds.nii.gz'),
            'seg_path':    str(case / 'S_2_deeds_seg_reg.nii.gz'),
            'case_id': 'caseA'}
    base = dict(patch_size=64, patch_depth=1, overlap=0.0, seed=0,
                hu_min=-200, hu_max=400, min_patch_std=1.0, min_patch_mean=-800.0,
                min_patch_max=-500.0, out_dir=tmp, use_organ=True,
                organ_focus_frac=0.0)

    w = resolve_organ_weights(enabled=True, preset='tiered')
    ds = CTPairDataset([pair], {**base, 'organ_weights': w},
                       max_patches=200, split_name='train')
    check(ds.mask_multilabel, "mask_multilabel is True on the TRAIN split")
    ids = set()
    for m in ds.mask_patches:
        ids |= {int(v) for v in np.unique(m)}
    check({AORTA, BOWEL} <= ids, f"raw label ids preserved in train patches ({sorted(ids)})")

    # Without organ_weights the train split stays binary (unchanged legacy path).
    ds2 = CTPairDataset([pair], dict(base), max_patches=200, split_name='train')
    ids2 = set()
    for m in ds2.mask_patches:
        ids2 |= {int(v) for v in np.unique(m)}
    check(not ds2.mask_multilabel and ids2 <= {0, 1},
          f"legacy path unchanged: binary train mask ({sorted(ids2)})")

    # The two must not share a patch cache: one holds label ids, the other 0/1.
    cfg_w = {**base, 'organ_weights': w, 'cache_dir': tmp}
    cfg_b = {**base, 'cache_dir': tmp}
    check(ds._cache_path([pair], cfg_w, 200) != ds2._cache_path([pair], cfg_b, 200),
          "cache key differs between multi-label and binary mask runs")


def test_zero_region_still_anchored():
    print("6. zero-weighted regions remain anchored by the global L1 floor")
    w = resolve_organ_weights(enabled=True, preset='tiered')
    C = CompositeLoss(dict(use_adversarial=False, use_perceptual=False,
                           use_feature_matching=False, lambda_l1=100.0,
                           use_organ=True, organ_weights=w, lambda_organ=5.0))
    mask = torch.zeros(2, 1, 16, 16)
    mask[..., :8, :] = AORTA
    mask[..., 8:, :] = BOWEL
    p = torch.rand(2, 1, 16, 16, requires_grad=True)
    total, d = C(pred=p, target=torch.rand(2, 1, 16, 16), mask=mask)
    total.backward()
    bowel_grad = float(p.grad[..., 8:, :].abs().sum())
    check(bowel_grad > 0.0,
          f"bowel still receives global-L1 gradient ({bowel_grad:.3f}) — "
          f"prevents unconstrained drift")
    check('lambda_l1' in d, "lambda_l1 logged in the loss dict for auditing")


def test_hu_profile_loss():
    print("7. organ HU-profile loss")
    from losses import OrganHUProfileLoss
    w = resolve_organ_weights(enabled=True, preset='tiered')
    L = OrganHUProfileLoss(organ_weights=w)

    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = AORTA       # weight 6
    mask[..., 4:, :] = BOWEL       # weight 0

    # Same organ MEANS but wildly different texture → this term must be ~0.
    # That is the defining property: it constrains level, not appearance.
    torch.manual_seed(0)
    tgt = torch.full((1, 1, 8, 8), 0.5)
    pred = tgt.clone()
    noise = torch.randn(1, 1, 4, 8) * 0.2
    pred[..., :4, :] = 0.5 + (noise - noise.mean())      # mean preserved exactly
    check(float(L(pred, tgt, mask)) < 1e-5,
          f"texture differs but organ means match → ~0 (got {float(L(pred,tgt,mask)):.2e})")

    # A constant offset inside a weighted organ → exactly that offset.
    # Weights normalise out when only one organ contributes.
    pred = tgt.clone(); pred[..., :4, :] += 0.10
    got = float(L(pred, tgt, mask))
    check(abs(got - 0.10) < 1e-6, f"0.10 offset in aorta → 0.10 (got {got:.6f})")

    # An offset in a ZERO-weighted organ must be ignored entirely.
    pred = tgt.clone(); pred[..., 4:, :] += 0.50
    check(float(L(pred, tgt, mask)) == 0.0,
          f"0.50 offset in zero-weighted bowel → 0 (got {float(L(pred,tgt,mask)):.6f})")

    # Gradient must reach the weighted organ and not the zero-weighted one.
    p = tgt.clone().requires_grad_(True)
    L(p, tgt + 0.1, mask).backward()
    g = p.grad[0, 0]
    check(float(g[:4].abs().sum()) > 0 and float(g[4:].abs().sum()) == 0.0,
          "gradient flows to aorta only, not bowel")

    # Organs below min_voxels are skipped rather than contributing a noisy mean.
    tiny = torch.zeros(1, 1, 8, 8); tiny[..., 0, :3] = AORTA     # 3 voxels < 16
    check(float(L(tgt + 0.2, tgt, tiny)) == 0.0, "organ below min_voxels skipped")

    # No mask → no-op (must not silently fall back to a global term).
    check(float(L(tgt + 0.3, tgt, None)) == 0.0, "no mask → 0")


def main():
    for t in (test_lut_weighting, test_all_zero_guard, test_decay_schedule,
              test_train_split_is_multilabel, test_zero_region_still_anchored,
              test_hu_profile_loss):
        t()
    print("\nOrgan-weighting smoke check " + ("PASSED." if _ok else "FAILED."))
    return 0 if _ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
