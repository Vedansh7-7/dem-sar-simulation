"""End-to-end pipeline used by the Streamlit app (and usable headless).

DEM input (upload or bounding box), memory guard, the two geometry modes (generic
Sentinel-1-like and real Sentinel-1), evaluation against the real image and export of
the results as a ZIP of GeoTIFFs plus metadata.
"""
from __future__ import annotations

import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, distance_transform_edt

from . import compare, rdsim
from .generic import DEFAULT_HEADING, H_S1, generic_geometry

MAX_FACETS = 4_000_000          # keeps the peak memory well below about 2 GB
MAX_BBOX_DEG = 1.0              # largest side of a downloaded bounding box
ML_EVAL = 5                     # multilook used for the comparison with real data

# fitted on the ascending Innsbruck scene by run_all.py
FITTED_N, FITTED_K = 1.98, 0.3354
MODELS = {
    "fitted": ("Fitted cos^1.98 (default)", dict(model="cos", n=FITTED_N, k=FITTED_K)),
    "muhleman": ("Muhleman", dict(model="muhleman")),
    "lambert": ("Lambert (cos^2)", dict(model="lambert")),
    "cos": ("Cosine", dict(model="cos", n=1.0)),
    "const": ("Constant", dict(model="const", k=0.1)),
}

PRESETS = {
    "Innsbruck, Austria (Alps)": (11.25, 47.15, 11.55, 47.35),
    "Innsbruck centre, small and fast": (11.33, 47.22, 11.45, 47.30),
    "Mount Etna, Italy": (14.90, 37.68, 15.10, 37.82),
    "Mount Fuji, Japan": (138.65, 35.30, 138.80, 35.42),
    "Grand Canyon, USA": (-112.20, 36.02, -112.00, 36.15),
}


def software_version() -> str:
    try:
        from importlib.metadata import version

        return version("sar-sim")
    except Exception:  # pragma: no cover
        return "0.1.0"


# ===================================================================== DEM input
def _fill_nodata(h):
    bad = ~np.isfinite(h)
    if bad.all():
        raise ValueError("The DEM contains no valid heights.")
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        h = h[tuple(idx)]
    return h


def _dem_from_dataset(src, bbox=None) -> dict:
    """Read a rasterio dataset as an EPSG:4326 DEM dict (reprojecting if needed)."""
    if src.crs is None:
        raise ValueError("The GeoTIFF has no coordinate reference system.")
    if src.crs.to_epsg() == 4326:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths() if bbox else None
        h = src.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
        tr = src.window_transform(win) if win is not None else src.transform
    else:
        tr, w, hh = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        h = np.full((hh, w), np.nan)
        reproject(rasterio.band(src, 1), h, src_nodata=src.nodata, dst_transform=tr, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    if src.nodata is not None:
        h[h == src.nodata] = np.nan
    h[(h < -1000) | (h > 9000)] = np.nan
    h = _fill_nodata(h)
    if min(h.shape) < 16:
        raise ValueError("The DEM is too small (need at least 16 x 16 pixels).")
    return make_dem(h, tr)


def make_dem(h, tr: Affine) -> dict:
    ny, nx = h.shape
    lon = tr.c + tr.a * (np.arange(nx) + 0.5)
    lat = tr.f + tr.e * (np.arange(ny) + 0.5)
    return {"h_orth": h, "lat": lat, "lon": lon, "transform": tr, "crs": "EPSG:4326"}


def dem_from_bytes(raw: bytes) -> dict:
    with MemoryFile(raw) as mf, mf.open() as src:
        return _dem_from_dataset(src)


def dem_from_file(path, bbox=None) -> dict:
    with rasterio.open(path) as src:
        return _dem_from_dataset(src, bbox)


def check_bbox(bbox):
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError("Invalid bounding box: need West < East and South < North.")
    if max(e - w, n - s) > MAX_BBOX_DEG:
        raise ValueError(f"The bounding box is too large (limit {MAX_BBOX_DEG} deg per side).")
    if min(e - w, n - s) < 0.01:
        raise ValueError("The bounding box is too small (at least 0.01 deg per side).")


def dem_from_bbox(bbox) -> dict:
    """Copernicus GLO-30 for bbox, re-using any cached DEM in data/ that covers it."""
    from . import data

    check_bbox(bbox)
    for f in sorted(data.DATA.glob("dem_*.tif")) if data.DATA.exists() else []:
        with rasterio.open(f) as src:
            b = src.bounds
            if src.crs and src.crs.to_epsg() == 4326 and b.left <= bbox[0] and b.bottom <= bbox[1] \
                    and b.right >= bbox[2] and b.top >= bbox[3]:
                return dem_from_file(f, bbox)
    name = "bbox_" + "_".join(f"{v:.4f}" for v in bbox)
    return dem_from_file(data.get_dem(tuple(bbox), name))


def dem_bbox(dem) -> tuple:
    tr, (ny, nx) = dem["transform"], dem["h_orth"].shape
    return (tr.c, tr.f + tr.e * ny, tr.c + tr.a * nx, tr.f)


def downsample_dem(dem: dict, factor: float) -> dict:
    """Area-average the DEM onto a grid ``factor`` times coarser."""
    h, tr = dem["h_orth"], dem["transform"]
    ny, nx = int(np.ceil(h.shape[0] / factor)), int(np.ceil(h.shape[1] / factor))
    tr2 = tr @ Affine.scale(h.shape[1] / nx, h.shape[0] / ny)
    out = np.empty((ny, nx))
    reproject(h, out, src_transform=tr, src_crs="EPSG:4326", dst_transform=tr2, dst_crs="EPSG:4326",
              resampling=Resampling.average)
    return make_dem(out, tr2)


def plan_resolution(dem: dict, max_facets: int = MAX_FACETS, max_oversample: int = 3):
    """Choose the oversampling (3 to 2) and, if needed, a DEM downsampling factor so that
    the number of facets stays below ``max_facets``. Returns (dem, oversample, factor)."""
    npx = dem["h_orth"].size
    for o in range(max_oversample, 1, -1):
        if npx * o * o <= max_facets:
            return dem, o, 1.0
    factor = float(np.sqrt(npx * 4 / max_facets)) * 1.01
    return downsample_dem(dem, factor), 2, factor


def dem_stats(dem: dict) -> dict:
    from .generic import metres_per_degree

    h = dem["h_orth"]
    m_lat, m_lon = metres_per_degree(float(dem["lat"].mean()))
    tr = dem["transform"]
    dy, dx = abs(tr.e) * m_lat, abs(tr.a) * m_lon
    return {"shape": list(h.shape), "pixel_m": [float(dx), float(dy)],
            "width_km": float(h.shape[1] * dx / 1e3), "height_km": float(h.shape[0] * dy / 1e3),
            "h_min": float(h.min()), "h_max": float(h.max()), "h_mean": float(h.mean()),
            "bbox": [float(v) for v in dem_bbox(dem)]}


def hillshade(dem: dict, azdeg=315, altdeg=45):
    from matplotlib.colors import LightSource

    s = dem_stats(dem)
    return LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(dem["h_orth"], dx=s["pixel_m"][0], dy=s["pixel_m"][1])


# ============================================================== shared rendering
def apply_speckle(img, looks: int, seed: int = 0):
    """Fully developed speckle (exponential intensity) followed by a boxcar multilook."""
    rng = np.random.default_rng(seed)
    out = img * rng.exponential(1.0, img.shape)
    return compare.multilook(out, looks) if looks > 1 else out


def _render(geo, row0, col0, shape, model_key, speckle, looks):
    kw = MODELS[model_key][1]
    sim = rdsim.render(geo, row0, col0, shape, **kw)
    sim["sigma0_clean"] = sim["sigma0"]
    if speckle:
        sim["sigma0"] = apply_speckle(sim["sigma0"], looks)
    m = sim["map"]
    sim["map"]["sigma0"] = compare.sample_radar_to_map(sim["sigma0"], m["row"], m["col"])
    return sim


def binned_curve(theta, val_db, good, step=2.0, min_count=50):
    edges = np.arange(0, 90 + step, step)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = good & (theta >= a) & (theta < b)
        if m.sum() >= min_count:
            cen.append((a + b) / 2)
            med.append(float(np.median(val_db[m])))
    return np.array(cen), np.array(med)


def _summary(sim, dem):
    m = sim["map"]
    return {"layover_pct": float(m["layover"].mean() * 100), "shadow_pct": float(m["shadow"].mean() * 100),
            "theta_loc_median_deg": float(np.nanmedian(m["theta_loc"]))}


# ================================================================== generic mode
def run_generic(dem: dict, pass_dir="ascending", look_deg=37.0, heading_deg=None, spacing=10.0,
                model="fitted", speckle=False, looks=4, H=H_S1, max_facets=MAX_FACETS,
                geometry=None, progress=None):
    """Generic Sentinel-1-like simulation. Returns a result dict (see ``build_zip``)."""
    t0 = time.time()
    say = progress or (lambda *_: None)
    heading = DEFAULT_HEADING[pass_dir] if heading_deg is None else float(heading_deg)
    if geometry is None:
        say("Planning resolution", 0.05)
        dem2, o, factor = plan_resolution(dem, max_facets)
        say("Projecting facets", 0.2)
        geo = generic_geometry(dem2, look_deg, heading, o, H, spacing)
    else:
        dem2, o, factor, geo = geometry
    say("Rendering backscatter", 0.7)
    sim = _render(geo, 0, 0, geo["image_shape"], model, speckle, looks)
    res = {
        "mode": "generic", "dem": dem2, "sim": sim, "radar_sigma0": sim["sigma0"],
        "pass": pass_dir, "oversample": o, "downsample_factor": factor, "n_facets": int(geo["line"].size),
        "sensor": {"type": "generic Sentinel-1-like (flat Earth, straight orbit, right looking)",
                   "pass": pass_dir, "look_angle_deg": look_deg, "heading_deg": heading,
                   "orbit_height_m": H, "pixel_spacing_m": spacing,
                   "radar_rows": "azimuth (along track)", "radar_cols": "ground range (away from sensor)"},
        "model": {"key": model, "label": MODELS[model][0], **MODELS[model][1]},
        "speckle": {"enabled": bool(speckle), "looks": int(looks) if speckle else 1},
        "frame": geo["frame"],
    }
    res["summary"] = _summary(sim, dem2)
    res["runtime_s"] = time.time() - t0
    say("Done", 1.0)
    return res


# ===================================================================== real mode
def find_scene(bbox, dates: str, orbit_state=None):
    """Most recent IW GRD scene fully covering bbox, or None."""
    from . import data

    try:
        return data.find_s1(tuple(bbox), dates, orbit_state=orbit_state)
    except IndexError:
        return None


def scene_key(bbox):
    import sarsim

    if np.allclose(bbox, sarsim.AOI_BBOX, atol=1e-6):
        return None
    return "bbox_" + "_".join(f"{v:.4f}" for v in bbox)


def load_scene(item, dem: dict, key=None):
    """Download (cached) the GRD window covering the DEM. Returns (prod, sigma0, meta)."""
    from . import data

    d = data.get_s1(item, dem, key=key)
    return data.load_s1(d)


def evaluate(s0, sim):
    """Multilook, co-register, calibrate the level with one constant, and score (as run_all.py)."""
    real_ml = compare.multilook(s0, ML_EVAL)
    sim_ml = compare.multilook(sim["sigma0_clean"], ML_EVAL)
    foot = binary_erosion(binary_fill_holes(binary_closing(sim["footprint"], iterations=3)), iterations=8)
    shift, box = compare.coregister(compare.db(real_ml), compare.db(sim_ml), foot)
    sim_reg = compare.apply_shift(sim_ml, shift)
    foot_reg = compare.apply_shift(foot.astype(float), shift) > 0.99
    valid = foot_reg & (real_ml > 0) & (sim_reg > 0)
    bias = float(np.median(compare.db(real_ml)[valid] - compare.db(sim_reg)[valid]))
    met = compare.metrics(compare.db(real_ml), compare.db(sim_reg) + bias, valid, box)
    met.update(level_offset_db=bias, shift_lines=float(shift[0]), shift_pixels=float(shift[1]))
    return met, {"real_ml": real_ml, "sim_reg": sim_reg, "valid": valid, "shift": shift, "bias": bias}


def run_real(dem: dict, scene, model="fitted", speckle=False, looks=4, max_facets=MAX_FACETS,
             geometry=None, progress=None):
    """Simulate in the exact GRD geometry of ``scene`` = (item, prod, s0, meta)."""
    t0 = time.time()
    say = progress or (lambda *_: None)
    item, prod, s0, meta = scene
    if geometry is None:
        dem2, o, factor = plan_resolution(dem, max_facets)
        say("Range-Doppler geocoding of all facets", 0.2)
        geo = rdsim.facet_geometry(prod, dem2, meta["geoid_N"], oversample=o)
    else:
        dem2, o, factor, geo = geometry
    say("Rendering backscatter", 0.6)
    sim = _render(geo, meta["row0"], meta["col0"], s0.shape, model, speckle, looks)
    say("Co-registration and metrics", 0.8)
    met, ev = evaluate(s0, sim)
    m = sim["map"]
    real_map = compare.sample_radar_to_map(ev["real_ml"], m["row"] + ev["shift"][0], m["col"] + ev["shift"][1])
    res = {
        "mode": "real", "dem": dem2, "sim": sim, "radar_sigma0": sim["sigma0"], "real_sigma0": s0,
        "pass": meta.get("orbit_state") or prod.pass_direction.lower(),
        "oversample": o, "downsample_factor": factor, "n_facets": int(geo["line"].size),
        "metrics": met, "eval": ev, "real_map": real_map,
        "sensor": {"type": "real Sentinel-1 IW GRD geometry (orbit state vectors, zero-Doppler)",
                   "scene_id": meta["id"], "platform": meta.get("platform"), "datetime": meta.get("datetime"),
                   "pass": meta.get("orbit_state"), "polarisation": meta.get("pol", "vv").upper(),
                   "incidence_mid_swath_deg": prod.inc_mid, "pixel_spacing_m": [prod.dgr, prod.daz],
                   "grd_window_row0": meta["row0"], "grd_window_col0": meta["col0"],
                   "geoid_N_m": meta["geoid_N"]},
        "model": {"key": model, "label": MODELS[model][0], **MODELS[model][1]},
        "speckle": {"enabled": bool(speckle), "looks": int(looks) if speckle else 1},
    }
    res["summary"] = _summary(sim, dem2)
    res["runtime_s"] = time.time() - t0
    say("Done", 1.0)
    return res


# ======================================================================== export
def _tif_bytes(arr, tr=None, crs=None, nodata=None, desc=None) -> bytes:
    arr = np.asarray(arr)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
                compress="deflate")
    if tr is not None:
        prof.update(transform=tr, crs=crs)
    if nodata is not None:
        prof["nodata"] = nodata
    with MemoryFile() as mf:
        with mf.open(**prof) as dst:
            dst.write(arr, 1)
            if desc:
                dst.set_band_description(1, desc)
        return mf.read()


def build_zip(res: dict, inputs: dict | None = None) -> bytes:
    """ZIP with map-geometry GeoTIFFs, the radar-geometry image (GeoTIFF + NPY) and metadata.json."""
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    dem, m = res["dem"], res["sim"]["map"]
    tr, crs = dem["transform"], "EPSG:4326"
    s0 = np.asarray(m["sigma0"], np.float32)
    files = {
        "map/sigma0_linear.tif": _tif_bytes(s0, tr, crs, np.nan, "simulated sigma0 (linear)"),
        "map/sigma0_db.tif": _tif_bytes(compare.db(s0).astype(np.float32), tr, crs, np.nan, "simulated sigma0 [dB]"),
        "map/local_incidence_deg.tif": _tif_bytes(m["theta_loc"].astype(np.float32), tr, crs, None, "local incidence [deg]"),
        "map/layover_mask.tif": _tif_bytes(m["layover"].astype(np.uint8), tr, crs, None, "layover (1)"),
        "map/shadow_mask.tif": _tif_bytes(m["shadow"].astype(np.uint8), tr, crs, None, "shadow (1)"),
        "map/dem_used.tif": _tif_bytes(dem["h_orth"].astype(np.float32), tr, crs, None, "DEM heights [m]"),
    }
    rad = np.asarray(res["radar_sigma0"], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        files["radar/sigma0_linear.tif"] = _tif_bytes(rad, desc="simulated sigma0 (linear), radar geometry")
    buf = io.BytesIO()
    np.save(buf, rad)
    files["radar/sigma0_linear.npy"] = buf.getvalue()

    meta = {
        "software": {"name": "sar-sim (DEM-based SAR simulation)", "version": software_version(),
                     "python": platform.python_version(), "numpy": np.__version__,
                     "rasterio": rasterio.__version__},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": res["mode"], "inputs": inputs or {}, "sensor": res["sensor"], "model": res["model"],
        "speckle": res["speckle"],
        "dem": {**dem_stats(dem), "crs": crs, "transform": list(tr)[:6],
                "downsample_factor": res["downsample_factor"], "oversample": res["oversample"],
                "n_facets": res["n_facets"], "heights": "orthometric (EGM2008) as provided"},
        "summary": res["summary"], "runtime_s": res["runtime_s"],
        "radar_image_shape": list(rad.shape),
        "files": {
            "map/*.tif": "map geometry, EPSG:4326, same grid as map/dem_used.tif",
            "radar/sigma0_linear.*": "simulated sigma0 in radar geometry (rows azimuth, columns ground range)",
        },
    }
    if res["mode"] == "generic":
        meta["frame"] = res["frame"]
    if res.get("metrics"):
        meta["metrics_vs_real"] = {k: float(v) for k, v in res["metrics"].items()}
        meta["metrics_note"] = ("noise-free simulation vs real sigma0, both 5 x 5 multilooked, sub-pixel "
                                "co-registered, one constant level offset removed")
    files["metadata.json"] = json.dumps(meta, indent=2, default=str).encode()

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in files.items():
            z.writestr(name, b)
    return out.getvalue()


def save_zip(res, path: Path, inputs=None):
    Path(path).write_bytes(build_zip(res, inputs))
