"""Unit tests for the geometric core of the simulator."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarsim.geodesy import llh2ecef  # noqa: E402
from sarsim.rdsim import shadow_layover, sigma0_model, splat  # noqa: E402

H = 693e3


def profile(z, y):
    Y = 485e3 + y
    return shadow_layover(np.zeros_like(y), Y, np.arctan2(Y, H - z), np.sqrt(Y**2 + (H - z) ** 2),
                          tol_alpha=1e-7, tol_R=0.05)


def test_flat_ground_has_no_shadow_or_layover():
    y = np.linspace(-1000, 1000, 2001)
    sh, lo = profile(np.zeros_like(y), y)
    assert not sh.any() and not lo.any()


def test_steep_foreslope_is_layover_and_steep_backslope_is_shadow():
    y = np.linspace(-2000, 2000, 8001)
    z = np.where(y < 0, 1000 + y * np.tan(np.radians(60)), 1000 - y * np.tan(np.radians(70)))
    z = np.maximum(z, 0)
    sh, lo = profile(z, y)
    fore = (y > -500) & (y < -50)          # on the 60° fore-slope (look angle 35°)
    back = (y > 50) & (y < 300)            # on the 70° back-slope (> 90° - 35°)
    assert lo[fore].all()
    assert sh[back].all()
    assert not sh[y < -700].any()          # in front of the mountain nothing is shadowed


def test_gentle_slopes_are_neither():
    y = np.linspace(-2000, 2000, 4001)
    z = 300 * np.exp(-(y / 900) ** 2)      # max slope ~16°
    sh, lo = profile(z, y)
    assert not sh.any() and not lo.any()


def test_sigma0_models_monotonic_and_zero_beyond_90():
    c = np.cos(np.radians(np.linspace(0, 89, 50)))
    for m in ("cos", "lambert", "muhleman"):
        s = sigma0_model(c, m)
        assert np.all(np.diff(s) <= 1e-12)
        assert sigma0_model(np.array([-0.2]), m)[0] == 0


def test_splat_conserves_energy():
    rng = np.random.default_rng(0)
    r, c = rng.uniform(1, 48, 1000), rng.uniform(1, 48, 1000)
    w = rng.uniform(0, 1, 1000)
    assert splat(r, c, w, (50, 50)).sum() == pytest.approx(w.sum())


def test_ecef_equator_and_pole():
    assert np.allclose(llh2ecef(0.0, 0.0, 0.0), [6378137.0, 0, 0])
    assert np.allclose(llh2ecef(90.0, 0.0, 0.0), [0, 0, 6356752.314245], atol=1e-3)


def test_blockwise_facet_geometry_is_block_size_independent():
    """Chunked facet geometry must not depend on the block size (halo rows for normals)."""
    from sarsim import data
    from sarsim.rdsim import facet_geometry
    from sarsim.s1 import S1Product

    scenes = [p for p in data.DATA.glob("S1*") if (p / "annotation.xml").exists()]
    dems = list(data.DATA.glob("dem_*.tif"))
    if not scenes or not dems:
        pytest.skip("no cached data")
    prod = S1Product((scenes[0] / "annotation.xml").read_text(encoding="utf-8"))
    d = data.load_dem(dems[0])
    sub = {"h_orth": d["h_orth"][300:340, 500:560], "lat": d["lat"][300:340], "lon": d["lon"][500:560]}
    a = facet_geometry(prod, sub, 49.6, oversample=2, block_rows=7)
    b = facet_geometry(prod, sub, 49.6, oversample=2, block_rows=10_000)
    for k in ("line", "pixel", "cos_loc", "area"):
        assert np.allclose(a[k], b[k], rtol=1e-6, atol=1e-4), k
    assert (a["shadow"] == b["shadow"]).all() and (a["layover"] == b["layover"]).all()


def test_geocoding_matches_esa_grid():
    """Range-Doppler geocoding reproduces ESA's geolocation grid (needs cached data)."""
    from sarsim import data

    scenes = [p for p in data.DATA.glob("S1*") if (p / "annotation.xml").exists()]
    if not scenes:
        pytest.skip("no cached Sentinel-1 annotation")
    from sarsim.s1 import S1Product

    import xml.etree.ElementTree as ET

    xml = (scenes[0] / "annotation.xml").read_text(encoding="utf-8")
    prod = S1Product(xml)
    g = prod.grid
    res = prod.geocode(llh2ecef(g["latitude"], g["longitude"], g["height"]))
    # ESA's 'line' column is nominal; each grid point carries its exact zero-Doppler time
    pts = ET.fromstring(xml).findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    t_esa = np.array([prod.sec(np.datetime64(p.find("azimuthTime").text, "ns")) for p in pts])
    assert np.abs(res["t"] - t_esa).max() < 1e-4             # < 0.07 lines
    assert np.abs(res["line"] - g["line"]).max() < 0.25
    assert np.abs(res["pixel"] - g["pixel"]).max() < 0.1
