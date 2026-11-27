"""Comparison utilities: dB conversion, multilooking, co-registration and similarity metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, shift as nd_shift, uniform_filter
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation


def db(x, floor: float = 1e-5):
    return 10 * np.log10(np.maximum(x, floor))


def multilook(x, k: int = 5):
    """Boxcar multilook of an intensity image (k x k)."""
    return uniform_filter(np.asarray(x, float), k)


def interior_box(mask, min_valid: float = 0.995):
    """Largest centred box (shrinking from the bounding box) that is (almost) fully valid."""
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    rm, cm = (r0 + r1) / 2, (c0 + c1) / 2
    for thr in (min_valid, 0.98, 0.95, 0.9):          # relax only if no clean box exists
        for s in np.linspace(0.95, 0.2, 31):
            hr, hc = s * (r1 - r0) / 2, s * (c1 - c0) / 2
            box = (slice(int(rm - hr), int(rm + hr)), slice(int(cm - hc), int(cm + hc)))
            if mask[box].mean() >= thr:
                return box
    raise ValueError("no valid interior box")


def coregister(ref_db, mov_db, mask, upsample: int = 20):
    """Sub-pixel shift (rows, cols) that moves ``mov`` onto ``ref`` (cross-correlation
    of the mean-removed dB images inside a fully valid interior box)."""
    box = interior_box(mask)
    a = ref_db[box] - ref_db[box].mean()
    b = mov_db[box] - mov_db[box].mean()
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)
    return np.asarray(shift), box


def apply_shift(img, shift):
    return nd_shift(img, shift, order=1, mode="constant", cval=0.0)


def metrics(ref_db, sim_db, mask, box=None, n_sample: int = 200_000, seed: int = 0):
    """Similarity of two dB images on valid pixels."""
    a, b = ref_db[mask], sim_db[mask]
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.size, min(n_sample, a.size), replace=False)
    out = {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "spearman_rho": float(spearmanr(a[idx], b[idx]).statistic),
        "bias_db": float(np.mean(a - b)),
        "rmse_db": float(np.sqrt(np.mean((a - b) ** 2))),
        "rmse_db_bias_removed": float(np.std(a - b)),
        "n_pixels": int(a.size),
    }
    if box is not None:
        x, y = ref_db[box], sim_db[box]
        lo, hi = np.percentile(np.r_[x.ravel(), y.ravel()], [1, 99])
        nx = np.clip((x - lo) / (hi - lo), 0, 1)
        ny = np.clip((y - lo) / (hi - lo), 0, 1)
        out["ssim"] = float(structural_similarity(nx, ny, data_range=1.0))
    return out


def sample_radar_to_map(img, rows, cols):
    """Back-geocode a radar-geometry image to map (DEM) geometry by bilinear sampling."""
    return map_coordinates(img, [rows, cols], order=1, mode="constant", cval=np.nan)
