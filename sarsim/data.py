"""Data access: Copernicus GLO-30 DEM and Sentinel-1 GRD from Microsoft Planetary
Computer (anonymous SAS tokens, no account needed). Everything is cached in ./data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.windows import Window

from .geodesy import geoid_undulation, llh2ecef
from .s1 import S1Product

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _catalog():
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(STAC, modifier=pc.sign_inplace)


# ------------------------------------------------------------------------ DEM
def get_dem(bbox, name: str = "aoi") -> Path:
    """Copernicus GLO-30 crop for bbox=(W,S,E,N) -> data/dem_<name>.tif (EGM2008 heights)."""
    out = DATA / f"dem_{name}.tif"
    if out.exists():
        return out
    DATA.mkdir(exist_ok=True)
    items = list(_catalog().search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    srcs = [rasterio.open(i.assets["data"].href) for i in items]
    arr, tr = merge(srcs, bounds=bbox)
    prof = srcs[0].profile
    prof.update(driver="GTiff", height=arr.shape[1], width=arr.shape[2], transform=tr, compress="deflate")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    return out


def load_dem(path):
    """Return dict(h_orth, lat, lon, transform, crs) with 1-D lat/lon pixel-centre axes."""
    with rasterio.open(path) as src:
        h = src.read(1).astype(np.float64)
        tr = src.transform
        lon = tr.c + tr.a * (np.arange(src.width) + 0.5)
        lat = tr.f + tr.e * (np.arange(src.height) + 0.5)
        return {"h_orth": h, "lat": lat, "lon": lon, "transform": tr, "crs": src.crs}


# ----------------------------------------------------------------- land cover
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
    70: ("Snow & ice", "#f0f0f0"), 80: ("Water", "#0064c8"), 90: ("Wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"), 100: ("Moss & lichen", "#fae6a0"),
}


def get_worldcover(bbox, name: str = "aoi") -> Path:
    """ESA WorldCover 10 m land cover crop -> data/worldcover_<name>.tif."""
    out = DATA / f"worldcover_{name}.tif"
    if out.exists():
        return out
    DATA.mkdir(exist_ok=True)
    items = list(_catalog().search(collections=["esa-worldcover"], bbox=bbox).items())
    items.sort(key=lambda i: i.properties.get("esa_worldcover:product_version", ""), reverse=True)
    ver = items[0].properties.get("esa_worldcover:product_version")
    items = [i for i in items if i.properties.get("esa_worldcover:product_version") == ver]
    srcs = [rasterio.open(i.assets["map"].href) for i in items]
    arr, tr = merge(srcs, bounds=bbox)
    prof = srcs[0].profile
    prof.update(driver="GTiff", height=arr.shape[1], width=arr.shape[2], transform=tr, compress="deflate")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    return out


def worldcover_on_dem(path, dem: dict):
    """Nearest-neighbour sample of the land-cover map at the DEM pixel centres."""
    with rasterio.open(path) as src:
        lc = src.read(1)
        tr = src.transform
    LA, LO = np.meshgrid(dem["lat"], dem["lon"], indexing="ij")
    col = np.clip(((LO - tr.c) / tr.a).astype(int), 0, lc.shape[1] - 1)
    row = np.clip(((LA - tr.f) / tr.e).astype(int), 0, lc.shape[0] - 1)
    return lc[row, col]


# ----------------------------------------------------------------- Sentinel-1
def find_s1(bbox, dates: str, orbit_state: str | None = None, index: int = 0):
    """Most recent IW GRDH scene fully covering bbox (optionally ascending/descending)."""
    from shapely.geometry import box, shape

    aoi = box(*bbox).buffer(0.05)
    items = _catalog().search(collections=["sentinel-1-grd"], bbox=bbox, datetime=dates).items()
    ok = []
    for it in items:
        p = it.properties
        if p.get("sar:instrument_mode") != "IW" or not shape(it.geometry).contains(aoi):
            continue
        if orbit_state and p.get("sat:orbit_state") != orbit_state:
            continue
        ok.append(it)
    ok.sort(key=lambda i: (i.datetime, len(i.id)), reverse=True)
    return ok[index]


def _sibling(href: str, rel_from: str, rel_to: str) -> str:
    base, tok = href.split("?", 1)
    return base.replace(rel_from, rel_to) + "?" + tok


def get_s1(item, dem: dict, pol: str = "vv", margin: int = 60) -> Path:
    """Download annotation + calibration XML and the GRD window covering the DEM.

    Planetary Computer renames SAFE files; the product annotation lives next to the
    calibration folder as ``annotation/iw-<pol>.xml`` (the 'schema-product' asset is
    actually the RFI report for recent products).
    """
    d = DATA / item.id
    d.mkdir(parents=True, exist_ok=True)
    cal_href = item.assets[f"schema-calibration-{pol}"].href
    files = {
        "annotation.xml": _sibling(cal_href, f"calibration/calibration-iw-{pol}.xml", f"iw-{pol}.xml"),
        "calibration.xml": cal_href,
        "noise.xml": item.assets[f"schema-noise-{pol}"].href,
    }
    for name, url in files.items():
        if not (d / name).exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            (d / name).write_text(r.text, encoding="utf-8")

    win_tif = d / f"{pol}_window.tif"
    if win_tif.exists():
        return d

    prod = S1Product((d / "annotation.xml").read_text(encoding="utf-8"))
    # geocode a sparse set of DEM points (all heights) to find the GRD window
    lat, lon = dem["lat"], dem["lon"]
    s = max(1, len(lat) // 60)
    LA, LO = np.meshgrid(lat[::s], lon[::s], indexing="ij")
    N = geoid_undulation(float(lat.mean()), float(lon.mean()))
    H = dem["h_orth"][::s, ::s] + N
    g = prod.geocode(llh2ecef(LA, LO, H).reshape(-1, 3))
    r0 = int(max(0, np.floor(g["line"].min()) - margin))
    r1 = int(min(prod.n_lines, np.ceil(g["line"].max()) + margin))
    c0 = int(max(0, np.floor(g["pixel"].min()) - margin))
    c1 = int(min(prod.n_samples, np.ceil(g["pixel"].max()) + margin))
    with rasterio.open(item.assets[pol].href) as src:
        dn = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    with rasterio.open(win_tif, "w", driver="GTiff", height=dn.shape[0], width=dn.shape[1],
                       count=1, dtype=dn.dtype, compress="deflate") as dst:
        dst.write(dn, 1)
    meta = {"id": item.id, "row0": r0, "col0": c0, "pol": pol, "geoid_N": N,
            "datetime": str(item.datetime), "orbit_state": item.properties.get("sat:orbit_state"),
            "platform": item.properties.get("platform")}
    (d / "window.json").write_text(json.dumps(meta, indent=2))
    return d


def load_s1(scene_dir: Path, pol: str = "vv"):
    """Return (S1Product, sigma0 window [linear], meta)."""
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "window.json").read_text())
    prod = S1Product((scene_dir / "annotation.xml").read_text(encoding="utf-8"),
                     (scene_dir / "calibration.xml").read_text(encoding="utf-8"))
    with rasterio.open(scene_dir / f"{pol}_window.tif") as src:
        dn = src.read(1)
    return prod, prod.sigma0(dn, meta["row0"], meta["col0"]), meta
