"""Unit tests for metrics.py — the properties that catch a silently wrong metric."""

import numpy as np

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import metrics as M

_ok = True


def check(cond, label):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    _ok &= bool(cond)


def near(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_identity():
    print("1. identity (gen == real)")
    rng = np.random.default_rng(0)
    v = rng.random((32, 32, 8))
    check(near(M.mae(v, v), 0.0), "MAE 0")
    check(near(M.mse(v, v), 0.0), "MSE 0")
    check(M.psnr(v, v) == 100.0, "PSNR 100 sentinel")
    check(near(M.pcc(v, v), 1.0), "PCC 1")
    check(near(M.ssim(v, v), 1.0, 1e-4), f"SSIM 1 (got {M.ssim(v,v):.6f})")


def test_offset():
    print("2. constant offset")
    rng = np.random.default_rng(1)
    v = rng.random((16, 16, 4))
    d = 0.1
    check(near(M.mae(v, v + d), d), f"MAE = offset ({M.mae(v, v+d):.4f})")
    check(near(M.mse(v, v + d), d * d), f"MSE = offset^2 ({M.mse(v, v+d):.4f})")
    # A pure shift leaves correlation unchanged.
    check(near(M.pcc(v, v + d), 1.0), "PCC 1 under a pure shift")


def test_anticorrelated():
    print("3. anti-correlation")
    rng = np.random.default_rng(2)
    v = rng.random((16, 16, 4))
    check(near(M.pcc(v, -v), -1.0), f"PCC -1 for negated ({M.pcc(v, -v):.4f})")


def test_pcc_constant():
    print("4. degenerate PCC")
    v = np.random.default_rng(3).random((8, 8, 2))
    c = np.full_like(v, 0.5)
    check(np.isnan(M.pcc(v, c)), "PCC NaN when one side is constant")


def test_ssim_monotonic():
    print("5. SSIM decreases with noise")
    rng = np.random.default_rng(4)
    v = rng.random((64, 64, 4))
    s_clean = M.ssim(v, v)
    s_small = M.ssim(v, v + rng.normal(0, 0.05, v.shape))
    s_big = M.ssim(v, v + rng.normal(0, 0.20, v.shape))
    check(s_clean > s_small > s_big, f"1 > {s_small:.3f} > {s_big:.3f}")
    check(0.0 <= s_big <= 1.0 and s_small <= 1.0, "SSIM stays in range")


def test_ssim_matches_skimage_if_available():
    print("6. SSIM agrees with skimage (if installed)")
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError:
        print("     skimage not installed — skipped (runs on the server)")
        return
    rng = np.random.default_rng(5)
    a = rng.random((80, 80))
    b = a + rng.normal(0, 0.1, a.shape)
    mine = M._ssim_2d(a, b, data_range=1.0)
    theirs = sk_ssim(a, b, data_range=1.0, gaussian_weights=True,
                     sigma=1.5, use_sample_covariance=False)
    check(near(mine, theirs, 5e-3), f"mine {mine:.5f} vs skimage {theirs:.5f}")


def test_masked():
    print("7. masked (organ-region) metrics")
    rng = np.random.default_rng(6)
    real = rng.random((16, 16, 4))
    gen = real.copy()
    mask = np.zeros_like(real)
    mask[4:12, 4:12, :] = 5           # an "organ" label
    gen[mask > 0] += 0.1              # error only inside the organ
    m = M.masked_metrics(gen, real, mask)
    check(near(m['mae'], 0.1, 1e-9), f"masked MAE = in-organ offset ({m['mae']:.4f})")
    check(near(M.masked_metrics(real, real, mask)['pcc'], 1.0), "masked PCC 1 on identity")
    empty = M.masked_metrics(gen, real, np.zeros_like(mask))
    check(all(np.isnan(v) for v in empty.values()), "empty mask → all NaN")


def _textured(shape=(96, 96, 24), seed=7):
    """Broadband noise volume — has energy in every frequency band, so blurring it
    is guaranteed to be visible to a spectral metric."""
    return np.random.default_rng(seed).random(shape)


def test_raps_blur():
    print("8. RAPS detects blur (the l1_only vs adv_organ signature)")
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(8)

    # The situation the whole metric exists for. Ground truth = smooth anatomy
    # plus fine texture. Two competing models:
    #   `blur` — predicts the anatomy and drops the texture. This is what a
    #            per-voxel loss converges to: the conditional mean.
    #   `gan`  — reproduces texture with the RIGHT statistics but, because of
    #            residual registration error, not in the right places.
    # `blur` must win MAE (it does — that is why l1_only leads PSNR) and must
    # lose RAPS (which is the finding the current metric suite cannot express).
    structure = gaussian_filter(rng.random((96, 96, 24)), (4, 4, 0))
    real = structure + gaussian_filter(rng.random(structure.shape), (0.6, 0.6, 0)) * 0.30
    blur = gaussian_filter(real, (1.2, 1.2, 0))
    gan = structure + gaussian_filter(rng.random(structure.shape), (0.6, 0.6, 0)) * 0.30

    r_blur, r_gan = M.raps_hf_ratio(blur, real), M.raps_hf_ratio(gan, real)
    check(near(M.raps_hf_ratio(real, real), 1.0, 1e-9), "identity ratio 1.0")
    check(r_blur < 0.5, f"blur loses HF energy (ratio {r_blur:.4f} < 0.5)")
    check(M.mae(blur, real) < M.mae(gan, real),
          f"blur wins MAE ({M.mae(blur,real):.4f} < {M.mae(gan,real):.4f}) — as l1_only does")
    check(abs(r_gan - 1) < abs(r_blur - 1),
          f"but loses RAPS ({r_gan:.3f} vs {r_blur:.3f}) — the metric is not fooled")


def test_raps_noise_is_not_better():
    print("9. RAPS is two-sided (added noise overshoots, not improves)")
    real = _textured()
    noisy = real + np.random.default_rng(9).normal(0, 0.15, real.shape)
    check(M.raps_hf_ratio(noisy, real) > 1.0,
          "extra HF energy reads > 1, so it cannot be reported as 'sharper'")


def test_grad_hist():
    print("10. gradient-magnitude W1 distance")
    from scipy.ndimage import gaussian_filter
    real = _textured()
    check(near(M.grad_hist_distance(real, real), 0.0, 1e-12), "identity distance 0")
    d_blur = M.grad_hist_distance(gaussian_filter(real, (1.2, 1.2, 0)), real)
    check(d_blur > 0, f"blur has a nonzero distance ({d_blur:.5f})")

    # Insensitive to a pure shift of edge POSITION (the registration-error case)
    # while remaining sensitive to edge STRENGTH — the property that makes it
    # useful where per-voxel metrics are capped.
    d_shift = M.grad_hist_distance(np.roll(real, 3, axis=0), real)
    check(d_shift < d_blur * 0.5,
          f"shifted-but-sharp ({d_shift:.5f}) << blurred ({d_blur:.5f})")

    mask = np.zeros_like(real); mask[20:60, 20:60, :] = 5
    check(np.isnan(M.grad_hist_distance(real, real, np.zeros_like(mask))),
          "empty mask → NaN")
    check(near(M.grad_hist_distance(real, real, mask), 0.0, 1e-12),
          "masked identity distance 0")


def test_tile_starts():
    print("11. tile_starts geometry")
    check(M.tile_starts(64, 128, 64) == [0], "volume smaller than patch → one tile")
    s = M.tile_starts(267, 128, 64)
    check(s[0] == 0 and s[-1] == 267 - 128, f"covers both edges ({s[0]}, {s[-1]})")
    check(all(b - a == 64 for a, b in zip(s, s[1:])) or s[-1] - s[-2] < 64,
          "uniform stride except a flush-to-edge final tile")


def test_seam_energy():
    print("12. seam energy")
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(11)
    # Smooth, like real CT. Seam detectability is the size of the step over the
    # local gradient floor, so this must not be white noise — on white noise a
    # 0.05 DC step is genuinely invisible, and the metric correctly says so.
    clean = gaussian_filter(rng.random((267, 267, 12)), (3, 3, 0))
    check(abs(M.seam_energy(clean, 128, 0.5) - 1.0) < 0.15,
          f"seamless volume ≈ 1.0 (got {M.seam_energy(clean,128,0.5):.4f})")

    # Inject a DC step at each tile boundary — the InstanceNorm failure mode.
    seamed = clean.copy()
    for s in M.tile_starts(267, 128, 64):
        if s > 0:
            seamed[s:, :, :] += 0.02
    check(M.seam_energy(seamed, 128, 0.5) > 2.0,
          f"DC step at tile starts is detected ({M.seam_energy(seamed,128,0.5):.2f})")


def test_z_flicker():
    print("13. z-flicker")
    rng = np.random.default_rng(12)
    real = gaussian_filter_z(rng.random((48, 48, 40)))
    check(near(M.z_flicker(real, real), 1.0, 1e-9), "identity ratio 1.0")
    flick = real + rng.normal(0, 0.05, real.shape)     # independent per slice
    check(M.z_flicker(flick, real) > 1.0,
          f"per-slice independent noise reads > 1 ({M.z_flicker(flick,real):.3f})")
    smooth = gaussian_filter_z(real, 2.0)
    check(M.z_flicker(smooth, real) < 1.0,
          f"over-smoothed in Z reads < 1 ({M.z_flicker(smooth,real):.3f})")


def test_z_anisotropy_survives_blur():
    print("13b. z-anisotropy separates flicker from blur")
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(13)
    real = gaussian_filter(rng.random((64, 64, 40)), (1.5, 1.5, 1.5))

    # Slice-independent generation AND in-plane blur, together — the actual
    # l1_only situation, and the case that measured 0.876 on real data.
    gen = gaussian_filter(real, (2.5, 2.5, 0)) + rng.normal(0, 0.004, real.shape)

    zf, za = M.z_flicker(gen, real), M.z_flicker_anisotropy(gen, real)
    check(zf < 1.0, f"raw z_flicker is fooled by the blur, reads < 1 ({zf:.3f})")
    check(za > 1.0, f"anisotropy still exposes the Z artifact ({za:.3f} > 1)")
    check(near(M.z_flicker_anisotropy(real, real), 1.0, 1e-6), "identity 1.0")


def gaussian_filter_z(v, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(v, (0, 0, sigma))


def test_bundles():
    print("14. bundles")
    real = _textured()
    mask = np.zeros_like(real); mask[20:60, 20:60, :] = 5
    t = M.texture_metrics(real, real, mask)
    check(set(t) == {'raps_hf', 'grad_w1', 'org_grad_w1'}, "texture keys")
    check(near(t['raps_hf'], 1.0, 1e-9) and near(t['grad_w1'], 0.0, 1e-12),
          "texture identity values")
    c = M.consistency_metrics(real, real, patch_size=128, overlap=0.5)
    check(set(c) == {'seam', 'zflicker', 'zaniso'}, "consistency keys")
    check(near(c['zflicker'], 1.0, 1e-9) and near(c['zaniso'], 1.0, 1e-6),
          "flicker + anisotropy identity 1.0")
    check(np.isnan(M.consistency_metrics(real, real)['seam']),
          "no tiling geometry → seam NaN, not a wrong number")


def main():
    for t in (test_identity, test_offset, test_anticorrelated, test_pcc_constant,
              test_ssim_monotonic, test_ssim_matches_skimage_if_available, test_masked,
              test_raps_blur, test_raps_noise_is_not_better, test_grad_hist,
              test_tile_starts, test_seam_energy, test_z_flicker,
              test_z_anisotropy_survives_blur, test_bundles):
        t()
    print("\nmetrics " + ("PASSED." if _ok else "FAILED."))
    return 0 if _ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
