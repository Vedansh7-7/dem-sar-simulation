"""DEM-based SAR image simulation in real Sentinel-1 GRD geometry.

For every (oversampled) DEM facet:
  1. ECEF position X, surface normal n and true facet area A.
  2. Range-Doppler geocoding with the real orbit -> azimuth line, slant range R, GRD pixel.
  3. Local incidence angle  cos(theta_loc) = -u . n   (u = unit look vector sensor->X).
  4. Shadow / layover by ordering facets along each azimuth line by ground position.
  5. Backscatter sigma0(theta_loc) * A splatted (bilinear) into the GRD grid and
     normalised by the ellipsoid pixel area (daz * dgr), i.e. directly comparable with
     ESA-calibrated GRD sigma0.

The work is split in two: ``facet_geometry`` (expensive, done once per scene/DEM) and
``render`` (cheap, one call per backscatter model).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .geodesy import llh2ecef, up_vector


# ------------------------------------------------------------- backscatter
def sigma0_model(cos_loc, model: str = "muhleman", n: float = 1.0, k: float = 1.0):
    """Incidence-angle backscatter models (linear sigma0)."""
    c = np.clip(cos_loc, 0, 1)
    s = np.sqrt(1 - c**2)
    if model == "cos":
        return k * c**n
    if model == "lambert":
        return k * c**2
    if model == "muhleman":  # Muhleman (1964), as used by many SAR simulators
        return k * 0.0133 * c / (s + 0.1 * c) ** 3
    if model == "const":
        return k * (c > 0).astype(float)
    raise ValueError(model)


# ------------------------------------------------------- grouped running ops
def _grouped_cummax(values, groups, span):
    """Running max of ``values`` restarting at every new ``groups`` id (groups sorted)."""
    off = (groups - groups.min()) * span
    return np.maximum.accumulate(values + off) - off


def shadow_layover(line, psi, alpha, R, tol_alpha=2e-6, tol_R=2.0):
    """Shadow & layover flags for scattered facets.

    line  : azimuth line of each facet (rounded to integers to define range lines)
    psi   : ground position along range (monotonic with distance from the ground track)
    alpha : off-nadir look angle from the sensor to the facet
    R     : slant range
    A facet is in *shadow* if some facet nearer to the ground track (smaller psi) on the
    same azimuth line is seen at a larger look angle (the ray hits that one first).
    A facet is in *layover* if a nearer facet has a larger slant range, or a farther
    facet a smaller slant range (their echoes arrive mixed in the same range cells).
    """
    g = np.round(line).astype(np.int64)
    order = np.lexsort((psi, g))
    gs, a, r = g[order], alpha[order], R[order]
    shadow_s = a < _grouped_cummax(a, gs, 10.0) - tol_alpha
    span = r.max() - r.min() + 10.0
    lay_near = r < _grouped_cummax(r, gs, span) - tol_R
    rev = slice(None, None, -1)
    gr = gs[rev].max() - gs[rev]  # groups ascending again in reversed order
    lay_far = (-r[rev] < _grouped_cummax(-r[rev], gr, span) - tol_R)[rev]
    shadow = np.empty_like(shadow_s)
    layover = np.empty_like(shadow_s)
    shadow[order] = shadow_s
    layover[order] = lay_near | lay_far
    return shadow, layover


# ------------------------------------------------------------------ splatting
def splat(rows, cols, w, shape):
    """Bilinear accumulation of weights w at fractional (rows, cols) into an image."""
    img = np.zeros(shape[0] * shape[1])
    r0, c0 = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    fr, fc = rows - r0, cols - c0
    for dr, dc, ww in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                       (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        rr, cc = r0 + dr, c0 + dc
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        img += np.bincount(rr[ok] * shape[1] + cc[ok], weights=(w * ww)[ok], minlength=img.size)
    return img.reshape(shape)


# -------------------------------------------------------------- facet geometry
def facet_geometry(prod, dem: dict, geoid_N: float, oversample: int = 3, block_rows: int = 256):
    """Geocode every (oversampled) DEM facet into the GRD product. Expensive; do once.

    Processed in blocks of DEM rows (with a one-row halo for the normals) and stored in
    float32 to keep the memory footprint at ~25 bytes per facet.
    """
    h = dem["h_orth"] + geoid_N
    lat, lon = dem["lat"], dem["lon"]
    hz = zoom(h, oversample, order=1) if oversample > 1 else h
    latz = np.interp(np.linspace(0, len(lat) - 1, hz.shape[0]), np.arange(len(lat)), lat)
    lonz = np.interp(np.linspace(0, len(lon) - 1, hz.shape[1]), np.arange(len(lon)), lon)
    ny, nx = hz.shape
    n = ny * nx
    f32 = lambda: np.empty(n, np.float32)  # noqa: E731
    line, pixel, R, cos_loc, cos_ell, area, alpha, psi = (f32() for _ in range(8))
    for r0 in range(0, ny, block_rows):
        r1 = min(ny, r0 + block_rows)
        a0, a1 = max(0, r0 - 1), min(ny, r1 + 1)          # halo rows for the gradient
        LA, LO = np.meshgrid(latz[a0:a1], lonz, indexing="ij")
        X = llh2ecef(LA, LO, hz[a0:a1])
        nv = np.cross(np.gradient(X, axis=1), np.gradient(X, axis=0) if a1 - a0 > 1 else np.zeros_like(X))
        inner = slice(r0 - a0, r0 - a0 + (r1 - r0))
        X, nv, LA, LO = X[inner], nv[inner], LA[inner], LO[inner]
        ar = np.linalg.norm(nv, axis=-1)
        nv /= ar[..., None]
        up = up_vector(LA, LO)
        nv[np.einsum("...k,...k->...", nv, up) < 0] *= -1
        Xf, nf, upf = X.reshape(-1, 3), nv.reshape(-1, 3), up.reshape(-1, 3)
        g = prod.geocode(Xf)
        u = (Xf - g["S"]) / g["R"][:, None]
        Shat = g["S"] / np.linalg.norm(g["S"], axis=1)[:, None]
        sl = slice(r0 * nx, r1 * nx)
        line[sl], pixel[sl], R[sl] = g["line"], g["pixel"], g["R"]
        cos_loc[sl] = -np.einsum("ij,ij->i", u, nf)
        cos_ell[sl] = -np.einsum("ij,ij->i", u, upf)
        area[sl] = ar.ravel()
        alpha[sl] = np.arccos(np.clip(-np.einsum("ij,ij->i", u, Shat), -1, 1))
        psi[sl] = np.arccos(np.clip(np.einsum("ij,ij->i", Xf / np.linalg.norm(Xf, axis=1)[:, None], Shat), -1, 1))
    shadow, layover = shadow_layover(line, psi.astype(np.float64), alpha.astype(np.float64), R.astype(np.float64))
    del alpha, psi
    return {
        "line": line, "pixel": pixel, "R": R,
        "cos_loc": cos_loc, "cos_ell": cos_ell, "area": area,
        "shadow": shadow, "layover": layover,
        "grid_shape": hz.shape, "dem_shape": h.shape, "oversample": oversample,
        "pixel_area": prod.daz * prod.dgr,
    }


def render(geo: dict, row0: int, col0: int, shape, model: str = "muhleman", n: float = 1.0,
           k: float = 1.0, with_masks: bool = True):
    """Render a simulated sigma0 image (and optional masks) from precomputed facet geometry."""
    rows, cols = geo["line"] - row0, geo["pixel"] - col0
    sig = sigma0_model(geo["cos_loc"], model, n, k)
    lit = (~geo["shadow"]) & (geo["cos_loc"] > 0)
    w = np.where(lit, sig * geo["area"], 0.0) / geo["pixel_area"]
    out = {"sigma0": splat(rows, cols, w, shape)}
    if with_masks:
        area_img = splat(rows, cols, np.where(geo["shadow"], 0, geo["area"]) / geo["pixel_area"], shape)
        out["area_norm"] = area_img
        out["radar_shadow"] = (splat(rows, cols, geo["shadow"].astype(float), shape) > 0) & (area_img == 0)
        out["radar_layover"] = splat(rows, cols, geo["layover"].astype(float), shape) > 0
        out["footprint"] = splat(rows, cols, np.ones_like(w), shape) > 0
        k_ = geo["oversample"]
        hs, ds = geo["grid_shape"], geo["dem_shape"]
        sub = lambda a: a.reshape(hs)[::k_, ::k_][: ds[0], : ds[1]]  # noqa: E731
        out["map"] = {
            "row": sub(rows), "col": sub(cols),
            "theta_loc": np.degrees(np.arccos(np.clip(sub(geo["cos_loc"]), -1, 1))),
            "theta_ell": np.degrees(np.arccos(np.clip(sub(geo["cos_ell"]), -1, 1))),
            "shadow": sub(geo["shadow"]), "layover": sub(geo["layover"]), "R": sub(geo["R"]),
        }
    return out


def simulate_grd(prod, dem: dict, geoid_N: float, row0: int, col0: int, shape,
                 oversample: int = 3, model: str = "muhleman", n: float = 1.0, k: float = 1.0):
    """One-shot simulation of sigma0 in the GRD window [row0:+shape[0], col0:+shape[1]]."""
    geo = facet_geometry(prod, dem, geoid_N, oversample)
    return render(geo, row0, col0, shape, model, n, k)
