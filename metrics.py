"""
CT-native image-similarity metrics for the NCCT→CECT benchmark.

Scope, and what is deliberately NOT here:
  * PSNR, SSIM, MAE, MSE, PCC — the standard pixel metrics the papers report.
  * FID / LPIPS are intentionally omitted: they use ImageNet-pretrained networks
    built for natural RGB images, so their absolute values are not meaningful on
    grayscale CT. `benchmark.py` leaves a hook if they are ever wanted for
    paper-comparability.

Two conventions that make the numbers comparable across models:
  * Everything is computed on the shared HU window mapped to [0, 1] (`to_unit`),
    so every model is scored in the same domain regardless of how it was trained.
  * SSIM is Gaussian-windowed per 2-D slice and averaged (the Wang et al. form the
    papers use), NOT the single-value global approximation the live training loop
    uses. The two differ, so benchmark tables must use this one.

SSIM uses scipy (always present) rather than scikit-image, so the harness runs
without an extra dependency; results match skimage's `gaussian_weights=True` mode.

TEXTURE AND CONSISTENCY (added for the blur/seam question — see the plan and
../synthetic_CECT/PROJECT_PLAN.md §1.2)
--------------------------------------------------------------------------
Every metric above this line — and every per-organ metric in `phase_eval.py` —
scores either a per-voxel difference or a median HU *level* inside a mask. None of
them measures texture. That is why an L1-blurred volume can lead PSNR/SSIM while
visibly smoothing away parenchymal detail, and why the r=0.97 cross-model
correlation in ../synthetic_CECT/PROJECT_PLAN.md §1.2 bounds the *level* axis only: it says nothing
about sharpness, because nothing here measured sharpness.

  * `raps_hf_ratio` — radially-averaged power spectrum, generated/real, in a
    high-frequency band. < 1 means the model is systematically smoother than the
    ground truth. This is the direct, quantitative form of "it got its score by
    blurring".
  * `grad_hist_distance` — W1 distance between gradient-magnitude distributions,
    globally and inside a mask. Reads edge sharpness rather than edge position, so
    it is not defeated by the residual registration error that caps the per-voxel
    metrics.
  * `seam_energy` — first-difference energy on patch-tile boundaries over the same
    on interior lines. 1.0 = invisible seam. Reconstruction artifact, not a model
    quality measure; it should be driven to 1.0 and then ignored.
  * `z_flicker` — inter-slice difference of the generated volume over the real
    one's. 1.0 = matched through-plane smoothness.
  * `z_flicker_anisotropy` — the same, but each volume normalised by its OWN
    in-plane gradient first. Needed because `z_flicker` conflates two opposing
    effects: independent per-slice generation raises it, in-plane blur lowers it,
    and on a real l1 volume blur wins (measured 0.876, i.e. it reads as "too
    smooth in Z" even though every slice was generated independently). Dividing
    out in-plane sharpness isolates the slice-independence artifact.

Sharpness metrics are one-sided on purpose: a model can score a *better* HF ratio
than real by adding noise or hallucinating texture, so `raps_hf_ratio` must be read
as a distance from 1.0, never as "higher is better". The same applies to `seam`,
`z_flicker` and `z_flicker_anisotropy` — rank all four by |value - 1|.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, label

__all__ = ['to_unit', 'mae', 'mse', 'psnr', 'pcc', 'ssim',
           'volume_metrics', 'masked_metrics', 'body_mask',
           'raps', 'raps_hf_ratio', 'grad_hist_distance', 'tile_starts',
           'seam_energy', 'z_flicker', 'z_flicker_anisotropy',
           'texture_metrics', 'consistency_metrics',
           'crps_ensemble', 'interval_coverage', 'variance_ratio',
           'pit_values', 'calibration_metrics', 'min_samples_for_level',
           'achieved_level']


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def to_unit(vol: np.ndarray, hu_min: float = -200.0, hu_max: float = 400.0) -> np.ndarray:
    """Clip to the HU window and rescale to [0, 1] — the shared scoring domain."""
    v = np.clip(vol.astype(np.float64), hu_min, hu_max)
    return (v - hu_min) / (hu_max - hu_min)


# ---------------------------------------------------------------------------
# Scalar metrics (operate on whatever array is passed — full volume or masked 1-D)
# ---------------------------------------------------------------------------

def mae(p: np.ndarray, t: np.ndarray) -> float:
    return float(np.mean(np.abs(p - t)))


def mse(p: np.ndarray, t: np.ndarray) -> float:
    return float(np.mean((p - t) ** 2))


def psnr(p: np.ndarray, t: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(p, t)
    # 100 dB sentinel for an exact match, matching trainer._metric_set so the
    # benchmark and the live metric agree on the degenerate case.
    return 100.0 if m == 0 else float(10.0 * np.log10(data_range ** 2 / m))


def pcc(p: np.ndarray, t: np.ndarray) -> float:
    """Pearson correlation coefficient. Identical in form to the trainer's 'NCC'
    (trainer.py:_metric_set), just without the +1e-8 denominator fudge; returns
    NaN when a side is constant (correlation is undefined, not 0)."""
    p = p.ravel().astype(np.float64)
    t = t.ravel().astype(np.float64)
    pm, tm = p.mean(), t.mean()
    den = p.std() * t.std()
    if den == 0:
        return float('nan')
    return float(np.mean((p - pm) * (t - tm)) / den)


# ---------------------------------------------------------------------------
# SSIM — Gaussian-windowed, 2-D
# ---------------------------------------------------------------------------

def _ssim_2d(p: np.ndarray, t: np.ndarray, data_range: float = 1.0,
             sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03) -> float:
    """Single-slice SSIM with an 11-tap Gaussian window (sigma=1.5), the standard
    Wang et al. configuration. Matches skimage.structural_similarity with
    gaussian_weights=True, sigma=1.5, use_sample_covariance=False."""
    p = p.astype(np.float64)
    t = t.astype(np.float64)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    def flt(x):
        return gaussian_filter(x, sigma, truncate=3.5)   # ~11-tap window

    mu_p, mu_t = flt(p), flt(t)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    var_p = flt(p * p) - mu_p2
    var_t = flt(t * t) - mu_t2
    cov = flt(p * t) - mu_pt

    num = (2 * mu_pt + C1) * (2 * cov + C2)
    den = (mu_p2 + mu_t2 + C1) * (var_p + var_t + C2)
    smap = num / den
    # Crop the border the Gaussian window can't fully cover (skimage does this).
    pad = int(3.5 * sigma + 0.5)
    if smap.shape[0] > 2 * pad and smap.shape[1] > 2 * pad:
        smap = smap[pad:-pad, pad:-pad]
    return float(smap.mean())


def ssim(p_vol: np.ndarray, t_vol: np.ndarray, data_range: float = 1.0,
         slice_axis: int = -1) -> float:
    """Volume SSIM = mean of per-slice 2-D SSIM over `slice_axis` (default the
    last axis, which is axial for these VinDr volumes). Slices smaller than the
    Gaussian window are skipped."""
    p = np.moveaxis(p_vol, slice_axis, 0)
    t = np.moveaxis(t_vol, slice_axis, 0)
    vals = [_ssim_2d(p[i], t[i], data_range)
            for i in range(p.shape[0])
            if min(p[i].shape) >= 11]
    return float(np.mean(vals)) if vals else float('nan')


# ---------------------------------------------------------------------------
# Texture — spectral
# ---------------------------------------------------------------------------

def _slices(vol: np.ndarray, slice_axis: int, max_slices: Optional[int]) -> np.ndarray:
    """Move `slice_axis` to front and optionally subsample evenly. Subsampling is
    for speed only; the metrics below are means over slices, so an even subsample
    is an unbiased estimate of the full-volume value."""
    v = np.moveaxis(vol, slice_axis, 0)
    if max_slices and v.shape[0] > max_slices:
        v = v[np.linspace(0, v.shape[0] - 1, max_slices).astype(int)]
    return v


def raps(vol: np.ndarray, slice_axis: int = -1, n_bins: int = 64,
         max_slices: Optional[int] = 64) -> np.ndarray:
    """Radially-averaged power spectrum, averaged over 2-D slices.

    Returns `n_bins` values indexed by spatial frequency from 0 to Nyquist (0.5
    cycles/voxel). A 2-D Hann window is applied before the FFT: without it the
    implicit periodic wrap puts a bright cross of spurious high-frequency energy in
    every spectrum, which would swamp the very band this metric reads.

    Slices are mean-subtracted, so bin 0 carries no DC term.
    """
    v = _slices(np.asarray(vol, np.float64), slice_axis, max_slices)
    if v.ndim != 3 or min(v.shape[1:]) < 8:
        return np.full(n_bins, np.nan)
    h, w = v.shape[1:]

    win = np.outer(np.hanning(h), np.hanning(w))

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)                 # 0 .. ~0.707, Nyquist = 0.5
    # Bin over [0, 0.5]; the corners beyond Nyquist are anisotropic and dropped.
    idx = np.clip((r / 0.5 * n_bins).astype(int), 0, n_bins)
    keep = idx < n_bins
    idx_k = idx[keep]
    counts = np.bincount(idx_k, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = np.nan                   # empty bin → NaN, not 0

    acc = np.zeros(n_bins, np.float64)
    for sl in v:
        s = sl - sl.mean()
        p = np.abs(np.fft.fft2(s * win)) ** 2
        acc += np.bincount(idx_k, weights=p[keep], minlength=n_bins)
    return acc / len(v) / counts


def raps_hf_ratio(gen: np.ndarray, real: np.ndarray, slice_axis: int = -1,
                  lo: float = 0.5, hi: float = 1.0, n_bins: int = 64,
                  max_slices: Optional[int] = 64) -> float:
    """sqrt(generated / real) spectral energy over the band [lo, hi] x Nyquist.

    The square root puts the result in amplitude units, so 0.7 reads as "70% of the
    ground truth's high-frequency detail" rather than 49%.

    < 1 → smoother than real (blur).  > 1 → more high-frequency energy than real
    (noise or hallucinated texture). Read as |value - 1|; direction says which
    failure mode.
    """
    pg = raps(gen, slice_axis, n_bins, max_slices)
    pr = raps(real, slice_axis, n_bins, max_slices)
    b0, b1 = int(lo * n_bins), int(hi * n_bins)
    g = np.nansum(pg[b0:b1])
    r = np.nansum(pr[b0:b1])
    if not np.isfinite(g) or not np.isfinite(r) or r <= 0:
        return float('nan')
    return float(np.sqrt(g / r))


# ---------------------------------------------------------------------------
# Texture — gradient distribution
# ---------------------------------------------------------------------------

def _grad_mag(vol: np.ndarray, slice_axis: int) -> np.ndarray:
    """In-plane gradient magnitude per 2-D slice, as one flat array.

    In-plane only: the through-plane gradient of these volumes is dominated by
    slice thickness and by exactly the flicker `z_flicker` measures separately, so
    including it would conflate two different failures.

    float32 throughout — these are ~21M-voxel volumes and this is fed to a
    histogram, where float64 buys nothing but doubles the footprint.
    """
    v = np.moveaxis(np.asarray(vol, np.float32), slice_axis, 0)
    gy, gx = np.gradient(v, axis=(1, 2))
    np.square(gy, out=gy)
    np.square(gx, out=gx)
    gy += gx
    del gx
    return np.sqrt(gy, out=gy).ravel()


def _w1_from_grads(g: np.ndarray, r: np.ndarray, n_bins: int,
                   vmax: Optional[float]) -> float:
    """Closed-form 1-D W1 on a shared fixed binning."""
    if g.size == 0 or r.size == 0:
        return float('nan')
    if vmax is None:
        vmax = float(max(np.percentile(g, 99.9), np.percentile(r, 99.9)))
    if not np.isfinite(vmax) or vmax <= 0:
        return float('nan')
    edges = np.linspace(0.0, vmax, n_bins + 1)
    hg, _ = np.histogram(g, bins=edges)
    hr, _ = np.histogram(r, bins=edges)
    if hg.sum() == 0 or hr.sum() == 0:
        return float('nan')
    cg = np.cumsum(hg / hg.sum())
    cr = np.cumsum(hr / hr.sum())
    return float(np.sum(np.abs(cg - cr)) * (vmax / n_bins))


def grad_hist_distance(gen: np.ndarray, real: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       slice_axis: int = -1, n_bins: int = 256,
                       vmax: Optional[float] = None) -> float:
    """1-D Wasserstein (W1) distance between gradient-magnitude distributions.

    Closed-form W1 on a shared fixed binning (sum |CDF_g - CDF_r| * binwidth)
    rather than `scipy.stats.wasserstein_distance`, which sorts every sample — at
    ~20M voxels per volume that is the difference between seconds and minutes, and
    the histogram form is deterministic.

    This compares the *distribution* of edge strengths, not where the edges are, so
    unlike the per-voxel metrics it is not capped by residual registration error.
    Lower is better; 0 means identical edge-strength statistics.
    """
    g = _grad_mag(gen, slice_axis)
    r = _grad_mag(real, slice_axis)
    if mask is not None:
        sel = (np.moveaxis(np.asarray(mask), slice_axis, 0) > 0).ravel()
        if int(sel.sum()) < 16:
            return float('nan')
        g, r = g[sel], r[sel]
    return _w1_from_grads(g, r, n_bins, vmax)


# ---------------------------------------------------------------------------
# Reconstruction consistency — seams and through-plane flicker
# ---------------------------------------------------------------------------

def tile_starts(length: int, patch: int, stride: int) -> List[int]:
    """Tile-start indices covering [0, length), always including a final
    flush-to-edge tile. Canonical copy — `infer_volume._starts` mirrors this, and
    the seam metric must use the identical geometry to look at the right lines."""
    if length <= patch:
        return [0]
    s = list(range(0, length - patch + 1, stride))
    if s[-1] != length - patch:
        s.append(length - patch)
    return s


def _seam_axis(vol: np.ndarray, axis: int, patch: int, stride: int,
               guard: int = 1) -> float:
    """Boundary/interior first-difference ratio along one axis."""
    n = vol.shape[axis]
    starts = tile_starts(n, patch, stride)
    d = np.abs(np.diff(vol, axis=axis))          # index i = step i -> i+1
    prof = np.moveaxis(d, axis, 0).reshape(d.shape[axis], -1).mean(axis=1)
    if prof.size == 0:
        return float('nan')

    # A tile beginning at s makes the step (s-1 -> s) a seam.
    seam = {s - 1 for s in starts if 0 < s <= prof.size}
    # Exclude a guard band around each seam from the interior baseline so a
    # slightly mislocated seam cannot contaminate its own reference.
    near = {j for s in seam for j in range(s - guard, s + guard + 1)}
    interior = [i for i in range(prof.size) if i not in near]
    seam = sorted(seam)
    if not seam or not interior:
        return float('nan')

    den = float(np.mean(prof[interior]))
    if den <= 0:
        return float('nan')
    return float(np.mean(prof[seam]) / den)


def seam_energy(vol: np.ndarray, patch_size, overlap: float,
                in_plane_axes: Sequence[int] = (0, 1)) -> float:
    """Mean boundary/interior gradient ratio over the tiled in-plane axes.

    1.0 = tile boundaries are indistinguishable from anywhere else. > 1 = visible
    seams. Needs the run's own patch geometry; volumes not produced by tiling (or
    runs whose geometry is unknown) should not be scored with this.

    Defaults assume the saved (X, Y, Z) NIfTI layout that `infer_volume` writes,
    where axes 0 and 1 are the tiled in-plane axes and 2 is axial.
    """
    ps = (int(patch_size), int(patch_size)) if isinstance(patch_size, (int, float)) \
        else (int(patch_size[0]), int(patch_size[1]))
    vals = []
    for ax, p in zip(in_plane_axes, ps):
        stride = max(1, int(p * (1.0 - overlap)))
        v = _seam_axis(np.asarray(vol, np.float64), ax, p, stride)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def _z_and_inplane(vol: np.ndarray, slice_axis: int):
    """(mean |through-plane diff|, mean |in-plane diff|) for one volume."""
    v = np.moveaxis(np.asarray(vol, np.float32), slice_axis, 0)
    if v.shape[0] < 2 or min(v.shape[1:]) < 2:
        return float('nan'), float('nan')
    dz = float(np.mean(np.abs(np.diff(v, axis=0))))
    dxy = float(0.5 * (np.mean(np.abs(np.diff(v, axis=1))) +
                       np.mean(np.abs(np.diff(v, axis=2)))))
    return dz, dxy


def z_flicker(gen: np.ndarray, real: np.ndarray, slice_axis: int = -1) -> float:
    """Generated inter-slice difference over the real volume's.

    Normalising by the real volume removes the per-case variation in how fast
    anatomy changes through the body. 1.0 = through-plane smoothness matches the
    ground truth.

    CAUTION — this term conflates two opposite effects, and measurement on a real
    volume showed the conflation is not hypothetical: independent per-slice
    generation pushes it up, while in-plane blur pushes it down, and blur can win.
    A value below 1 therefore does NOT rule out slice-independence artifacts. Use
    `z_flicker_anisotropy` to separate them; this term is kept because it is the
    directly interpretable "is Z as rough as it should be" number.
    """
    dg, _ = _z_and_inplane(gen, slice_axis)
    dr, _ = _z_and_inplane(real, slice_axis)
    if not np.isfinite(dg) or not np.isfinite(dr) or dr <= 0:
        return float('nan')
    return float(dg / dr)


def z_flicker_anisotropy(gen: np.ndarray, real: np.ndarray,
                         slice_axis: int = -1) -> float:
    """Ratio-of-ratios: (dz/dxy)_gen / (dz/dxy)_real.

    Dividing by each volume's OWN in-plane gradient cancels overall sharpness, so
    unlike `z_flicker` this isolates through-plane *anisotropy* — whether Z is
    rougher relative to in-plane than the ground truth is — which is the actual
    signature of generating each axial slice independently.

    1.0 = correct anisotropy. > 1 = Z rougher than it should be relative to
    in-plane detail, i.e. slice-independence flicker, and it stays > 1 even when
    the model is blurred in-plane. < 1 = over-smoothed through-plane.
    """
    dgz, dgxy = _z_and_inplane(gen, slice_axis)
    drz, drxy = _z_and_inplane(real, slice_axis)
    if not all(np.isfinite(x) for x in (dgz, dgxy, drz, drxy)):
        return float('nan')
    if dgxy <= 0 or drxy <= 0 or drz <= 0:
        return float('nan')
    return float((dgz / dgxy) / (drz / drxy))


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

def texture_metrics(gen: np.ndarray, real: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    slice_axis: int = -1, n_bins: int = 256) -> Dict[str, float]:
    """Sharpness bundle. Inputs already in [0,1]. `mask` restricts the gradient
    distance to organ voxels (the spectral metric is inherently global — a
    windowed FFT of an arbitrary voxel set is not well defined).

    Gradient magnitudes are computed once and reused for the global and masked
    variants; at ~21M voxels per volume recomputing them is the dominant cost.
    """
    g = _grad_mag(gen, slice_axis)
    r = _grad_mag(real, slice_axis)
    # One shared vmax so the global and masked numbers sit on the same scale and
    # stay comparable across models.
    vmax = float(max(np.percentile(g, 99.9), np.percentile(r, 99.9)))

    org = float('nan')
    if mask is not None:
        sel = (np.moveaxis(np.asarray(mask), slice_axis, 0) > 0).ravel()
        if int(sel.sum()) >= 16:
            org = _w1_from_grads(g[sel], r[sel], n_bins, vmax)
    return {
        'raps_hf':     raps_hf_ratio(gen, real, slice_axis),
        'grad_w1':     _w1_from_grads(g, r, n_bins, vmax),
        'org_grad_w1': org,
    }


def consistency_metrics(gen: np.ndarray, real: np.ndarray,
                        patch_size=None, overlap: Optional[float] = None,
                        slice_axis: int = -1) -> Dict[str, float]:
    """Reconstruction-artifact bundle. `patch_size`/`overlap` come from the run's
    own `run_config.json`; without them the seam term is NaN (correct for models
    that do not tile, e.g. whole-slice or external baselines) and only the
    tiling-independent flicker term is reported."""
    seam = float('nan')
    if patch_size is not None and overlap is not None:
        in_plane = tuple(a for a in range(gen.ndim) if a != slice_axis % gen.ndim)
        seam = seam_energy(gen, patch_size, overlap, in_plane)
    return {'seam': seam,
            'zflicker': z_flicker(gen, real, slice_axis),
            'zaniso':   z_flicker_anisotropy(gen, real, slice_axis)}


def volume_metrics(gen: np.ndarray, real: np.ndarray, data_range: float = 1.0,
                   slice_axis: int = -1) -> Dict[str, float]:
    """Global pixel metrics over the whole volume. Inputs already in [0,1]."""
    return {
        'psnr': psnr(gen, real, data_range),
        'ssim': ssim(gen, real, data_range, slice_axis),
        'mae':  mae(gen, real),
        'mse':  mse(gen, real),
        'pcc':  pcc(gen, real),
    }


def masked_metrics(gen: np.ndarray, real: np.ndarray, mask: np.ndarray,
                   data_range: float = 1.0, min_voxels: int = 16) -> Dict[str, float]:
    """Organ-region metrics over mask>0 voxels.

    SSIM here is the single-value (global) form, not the windowed one: a windowed
    SSIM cannot be restricted to an arbitrary voxel set cleanly, and this keeps
    organ-SSIM consistent with the trainer's `val_org_ssim` so the benchmark and
    the training logs are comparable."""
    sel = mask > 0
    if int(sel.sum()) < min_voxels:
        return {k: float('nan') for k in ('psnr', 'ssim', 'mae', 'mse', 'pcc')}
    p, t = gen[sel].astype(np.float64), real[sel].astype(np.float64)

    mu1, mu2 = p.mean(), t.mean()
    s1, s2 = p.std(), t.std()
    s12 = float(np.mean((p - mu1) * (t - mu2)))
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_sv = float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                    ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2)))
    return {
        'psnr': psnr(p, t, data_range),
        'ssim': ssim_sv,
        'mae':  mae(p, t),
        'mse':  mse(p, t),
        'pcc':  float(s12 / (s1 * s2)) if s1 * s2 > 0 else float('nan'),
    }

def body_mask(vol_hu: np.ndarray, hu_thr: float = -500.0,
              slice_axis: int = -1, close_iter: int = 2) -> np.ndarray:
    """Patient-interior mask from a HU volume: excludes scanner air, the table,
    and everything outside the body contour.

    Why this exists: `to_unit` clips to [hu_min, hu_max], so every voxel below
    hu_min (air, lung, much of the fat) maps to exactly 0.0 and every voxel above
    hu_max (cortical bone) to exactly 1.0 — identically in the prediction and the
    target. Those voxels contribute zero error for EVERY model, and the global
    PSNR/SSIM/MAE/MSE/PCC average over them. On abdominal CT that is a large
    fraction of the volume, which is why the global metrics separate models far
    less than the organ-region ones (measured: global MAE 5.9 HU vs organ 19 HU).

    Computed per slice: a 3-D component leaks through the table where it touches
    the patient, and a 3-D hole-fill is much slower for no gain.

    Always build this from the REAL volume, never the generated one, so the mask
    is identical across models being compared.
    """
    v = np.moveaxis(np.asarray(vol_hu), slice_axis, 0)
    out = np.zeros(v.shape, dtype=bool)
    for i, sl in enumerate(v):
        m = sl > hu_thr
        if not m.any():
            continue
        if close_iter:
            m = binary_closing(m, iterations=close_iter)
        lab, n = label(m)
        if n > 1:                       # drop the table and any stray components
            cnt = np.bincount(lab.ravel())
            cnt[0] = 0
            m = lab == int(cnt.argmax())
        out[i] = binary_fill_holes(m)
    return np.moveaxis(out, 0, slice_axis)


# ---------------------------------------------------------------------------
# Calibration — scoring a PREDICTIVE DISTRIBUTION rather than a point estimate
# ---------------------------------------------------------------------------
#
# Everything above this line scores one number against one number. A diffusion
# model emits a SAMPLE from p(y|x), so scoring it that way answers the wrong
# question: DIFFUSION_PLAN.md §3 shows a single honest draw is expected to be
# ~1.41x WORSE on a point metric than the conditional mean, because for a
# Gaussian with spread s the mean predictor has MAE 0.798s and one draw has
# 1.128s. Sampling the true distribution COSTS you on MAE. That is not a defect.
#
# What the model should be judged on instead is whether its predicted spread
# matches the true conditional spread — which is knowable here, uniquely:
# analysis/BASELINE_REFERENCE.md records aortic level sd 24.4 HU across cases and
# a measured var(gen)/var(real) of 0.176 against a target of 1.0.
#
# All four functions below take an ENSEMBLE (n_samples,) or (n_samples, ...) and
# a truth of the matching trailing shape. They are pure NumPy and make no
# assumption that the predictive distribution is Gaussian — the empirical form is
# used throughout, because a diffusion ensemble has no closed form.
#
# TWO LEVELS, and they measure different things (see scripts/calibration_eval.py):
#   per-ORGAN  the predictive distribution of an organ's MEDIAN HU for one case.
#              This is the headline: it is the quantity phase_eval.py reads, the
#              quantity audit_enhancement.py regresses, and the level at which the
#              residual was measured to be aleatoric.
#   per-VOXEL  the predictive distribution of one voxel's value. This is what a
#              heteroscedastic NLL head natively produces, and the only level at
#              which it can compete — its per-voxel sigmas are independent, so
#              they average away to nothing at organ scale.

def crps_ensemble(samples: np.ndarray, truth) -> float:
    """Continuous Ranked Probability Score from an ensemble. Lower is better.

    Uses the energy form

        CRPS = E|X - y| - 0.5 * E|X - X'|

    with X, X' independent draws from the predictive distribution. This is exact
    for the empirical distribution of the ensemble and needs no CDF integration,
    no binning and no distributional assumption.

    Note the two terms pull against each other, which is the property that makes
    CRPS the right headline scalar here: the first rewards being close to the
    truth, the second PENALISES an ensemble that is too wide — but it is only
    half-weighted, so an over-confident ensemble (all samples identical) is not
    rewarded either. A point predictor scores CRPS = |mu - y|, i.e. its MAE, so
    CRPS and MAE are directly comparable and a probabilistic model that cannot
    beat the deterministic one's MAE has not earned its sampling cost.

    `samples` is (n,) for a scalar quantity or (n, ...) for a field; `truth` is a
    scalar or an array of the trailing shape. Returns the mean over all non-sample
    axes. NaNs propagate as NaN rather than being silently dropped — a NaN here
    means an organ was absent, and averaging over the rest would quietly change
    which organs a model was scored on.
    """
    s = np.asarray(samples, np.float64)
    y = np.asarray(truth, np.float64)
    n = s.shape[0]
    if n < 2:
        return float('nan')
    term1 = np.mean(np.abs(s - y[None, ...] if y.ndim else s - y), axis=0)
    # E|X - X'| over all ordered pairs. Sorting gives it in O(n log n) instead of
    # the O(n^2) pairwise difference matrix, which matters at per-voxel scale
    # where this runs over ~1e7 voxels: for sorted x, sum_ij |x_i - x_j| =
    # 2 * sum_k (2k - n + 1) x_(k).
    xs = np.sort(s, axis=0)
    k = np.arange(n, dtype=np.float64).reshape((n,) + (1,) * (s.ndim - 1))
    term2 = 2.0 * np.sum((2 * k - n + 1) * xs, axis=0) / (n * n)
    return float(np.mean(term1 - 0.5 * term2))


def min_samples_for_level(level: float = 0.90) -> int:
    """Smallest ensemble size that can express a central `level` interval at all.

    n samples split the line into n+1 gaps, and the truth is equally likely to
    fall in any of them under calibration. The widest expressible interval is
    [min, max], which contains n-1 of those gaps, so the largest achievable
    coverage is (n-1)/(n+1). Requiring that to reach `level` gives

        n >= (1 + level) / (1 - level)

    which is 19 at level=0.90 and 3 at level=0.50. Below it, NO ensemble can
    score the nominal value however well calibrated it is.
    """
    return int(np.ceil((1.0 + level) / (1.0 - level) - 1e-9))


def interval_coverage(samples: np.ndarray, truth, level: float = 0.90) -> float:
    """Fraction of truths falling inside the ensemble's central `level` interval.

    Target = `level` itself. Below it the model is OVER-confident (predicted
    spread too narrow — the current deterministic failure mode, var ratio 0.176);
    above it, under-confident.

    ORDER STATISTICS, NOT np.quantile. The obvious implementation — interpolated
    empirical quantiles at (1±level)/2 — is BIASED DOWNWARD for small ensembles,
    badly: measured on perfectly calibrated Gaussian draws it reports 0.726 at
    n=8, 0.813 at n=20 and only reaches 0.891 at n=200. Reading any of those as
    "the model is over-confident" would be wrong, and at the N=8 exploration
    setting the bias (-0.17) is larger than any real miscalibration this project
    is likely to see.

    So the interval is the (n+1)-gap order-statistic pair instead: with the truth
    equally likely to fall in any of the n+1 gaps between sorted samples, taking
    the k-th smallest and k-th largest gives exactly (n+1-2k)/(n+1) expected
    coverage under calibration. k is chosen to put that as close to `level` as
    possible, and the estimator is then UNBIASED at every n.

    Returns NaN below `min_samples_for_level(level)`, where the nominal level is
    not merely noisy but unreachable.
    """
    s = np.asarray(samples, np.float64)
    y = np.asarray(truth, np.float64)
    n = s.shape[0]
    if n < min_samples_for_level(level):
        return float('nan')
    # k = number of samples excluded at each end. Round rather than floor, so the
    # achieved level straddles the nominal one instead of always exceeding it.
    k = int(round((n + 1) * (1.0 - level) / 2.0))
    k = max(1, min(k, (n - 1) // 2))
    xs = np.sort(s, axis=0)
    lo, hi = xs[k - 1], xs[n - k]
    return float(np.mean((y >= lo) & (y <= hi)))


def achieved_level(n: int, level: float = 0.90) -> float:
    """The coverage `interval_coverage` actually targets with n samples.

    Report this next to the measured value. The order-statistic interval can only
    take the discrete values (n+1-2k)/(n+1), so with n=20 and a nominal 0.90 the
    estimator is really aiming at 19/21 = 0.905 — a distinction worth stating
    rather than letting a reader assume exactly 0.900.
    """
    if n < min_samples_for_level(level):
        return float('nan')
    k = int(round((n + 1) * (1.0 - level) / 2.0))
    k = max(1, min(k, (n - 1) // 2))
    return (n + 1 - 2 * k) / (n + 1)


def variance_ratio(pred_means: np.ndarray, truths: np.ndarray,
                   pred_vars: Optional[np.ndarray] = None) -> float:
    """var(predicted) / var(true) ACROSS CASES — the headline of §7.

    Two readings, and the difference matters:

      * `pred_vars=None` — the classic form `audit_enhancement.py` computes: the
        variance of the model's POINT predictions across cases over the variance
        of the truths. This measures whether the model's outputs spread out as
        much as reality does. Currently 0.176.

      * `pred_vars` given — the mean PREDICTIVE variance (what the model claims
        its uncertainty is) over the variance of the truths. This is the
        calibration reading: a model can have correctly spread outputs and still
        misstate its own confidence, and vice versa.

    Target is 1.0 in both readings. Below 1 = under-dispersed.
    """
    p = np.asarray(pred_means, np.float64).ravel()
    t = np.asarray(truths, np.float64).ravel()
    ok = np.isfinite(t)
    if pred_vars is None:
        ok &= np.isfinite(p)
    else:
        v = np.asarray(pred_vars, np.float64).ravel()
        ok &= np.isfinite(v)
    if ok.sum() < 3:
        return float('nan')
    den = float(np.var(t[ok]))
    if den <= 0:
        return float('nan')
    num = float(np.var(p[ok])) if pred_vars is None else float(np.mean(v[ok]))
    return num / den


def pit_values(samples: np.ndarray, truth) -> np.ndarray:
    """Probability Integral Transform: the truth's rank within the ensemble, in
    [0, 1]. Feed the result to a histogram.

    A perfectly calibrated model gives a FLAT histogram. The shape names the
    failure, which "coverage = 0.4" alone does not:
      U-shaped   truth lands outside the ensemble too often → over-confident
      humped     truth lands mid-ensemble too often         → under-confident
      sloped     the predictions are biased, not mis-spread

    Ties are broken by counting strictly-less-than plus half the equalities, the
    standard randomised-PIT correction for a discrete ensemble; without it an
    ensemble with repeated values piles mass on exact 0 and 1 and reads as
    over-confident when it is merely coarse.
    """
    s = np.asarray(samples, np.float64)
    y = np.asarray(truth, np.float64)
    n = s.shape[0]
    below = np.sum(s < y, axis=0)
    equal = np.sum(s == y, axis=0)
    return np.asarray((below + 0.5 * equal) / n, np.float64)


def calibration_metrics(samples: np.ndarray, truth,
                        level: float = 0.90) -> Dict[str, float]:
    """Bundle for one quantity: CRPS, coverage at two levels, spread, mean error.

    `ens_sd` and `abs_err` are reported alongside the scores because a CRPS on
    its own cannot be read: a model can improve CRPS either by moving its mean
    closer or by narrowing its spread, and those are different results. Reporting
    both makes which one happened visible.

    Coverage is reported at BOTH `level` and 0.50. The 50% interval needs only 3
    samples where a 90% one needs 19 (`min_samples_for_level`), so it is the only
    coverage number available from a small exploratory run — and it is a real
    calibration check, not a consolation prize. `coverage_target` /
    `coverage50_target` give the value a perfectly calibrated model would score
    at this exact n, which is not quite the nominal level.
    """
    s = np.asarray(samples, np.float64)
    y = np.asarray(truth, np.float64)
    n = s.shape[0]
    return {
        'crps':             crps_ensemble(s, y),
        'coverage':         interval_coverage(s, y, level),
        'coverage_target':  achieved_level(n, level),
        'coverage50':       interval_coverage(s, y, 0.50),
        'coverage50_target': achieved_level(n, 0.50),
        'ens_sd':   float(np.mean(np.std(s, axis=0, ddof=1))) if n > 1 else float('nan'),
        'abs_err':  float(np.mean(np.abs(np.mean(s, axis=0) - y))),
    }
