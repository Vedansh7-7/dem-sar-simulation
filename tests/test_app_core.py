"""Headless tests of the GUI pipeline (generic mode, memory guard, upload, export) and the app."""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sarsim import pipeline as P  # noqa: E402

DEM = ROOT / "data" / "dem_innsbruck.tif"
CROP = (11.36, 47.24, 11.42, 47.28)          # about 4.5 x 4.5 km around the Nordkette foot
needs_dem = pytest.mark.skipif(not DEM.exists(), reason="data/dem_innsbruck.tif not cached")


@pytest.fixture(scope="module")
def crop():
    return P.dem_from_file(DEM, CROP)


@needs_dem
@pytest.mark.parametrize("pass_dir", ["ascending", "descending"])
def test_generic_pipeline(crop, pass_dir):
    res = P.run_generic(crop, pass_dir, look_deg=37.0, spacing=10.0, model="fitted")
    m, sim = res["sim"]["map"], res["sim"]
    assert m["sigma0"].shape == crop["h_orth"].shape == m["theta_loc"].shape
    assert np.isfinite(m["sigma0"]).all() and (m["sigma0"] >= 0).all()
    assert np.isfinite(m["theta_loc"]).all()
    assert sim["sigma0"].ndim == 2 and np.isfinite(sim["sigma0"]).all()
    assert sim["sigma0"][sim["footprint"]].mean() > 0
    assert m["layover"].dtype == bool and m["shadow"].dtype == bool
    assert m["layover"].any()                                    # steep alpine slopes
    assert 0 <= res["summary"]["shadow_pct"] < 20
    # flat-Earth sanity: median local incidence close to the look angle
    assert abs(res["summary"]["theta_loc_median_deg"] - 37.0) < 10
    # the fitted law at the median incidence is about -5 dB; image median within a few dB
    med = np.median(10 * np.log10(m["sigma0"][(~m["shadow"]) & (~m["layover"])]))
    assert -12 < med < 0


@needs_dem
def test_speckle_and_models(crop):
    for key in P.MODELS:
        res = P.run_generic(crop, "ascending", 37.0, spacing=15.0, model=key, speckle=True, looks=3)
        assert np.isfinite(res["radar_sigma0"]).all()


@needs_dem
def test_zip_export(crop):
    res = P.run_generic(crop, "ascending", 37.0, spacing=10.0)
    z = zipfile.ZipFile(io.BytesIO(P.build_zip(res, {"test": True})))
    names = set(z.namelist())
    for f in ("map/sigma0_linear.tif", "map/sigma0_db.tif", "map/local_incidence_deg.tif",
              "map/layover_mask.tif", "map/shadow_mask.tif", "radar/sigma0_linear.tif",
              "radar/sigma0_linear.npy", "metadata.json"):
        assert f in names
    meta = json.loads(z.read("metadata.json"))
    assert meta["mode"] == "generic" and meta["model"]["key"] == "fitted"
    with MemoryFile(z.read("map/sigma0_db.tif")) as mf, mf.open() as src:
        assert src.crs.to_epsg() == 4326
        assert src.shape == crop["h_orth"].shape
        assert src.transform == crop["transform"]
    rad = np.load(io.BytesIO(z.read("radar/sigma0_linear.npy")))
    assert rad.shape == res["radar_sigma0"].shape


def test_memory_guard_downsamples():
    h = np.random.default_rng(0).normal(500, 50, (1200, 1500))
    dem = P.make_dem(h, from_origin(11.0, 47.5, 1 / 3600, 1 / 3600))
    dem2, o, factor = P.plan_resolution(dem, max_facets=1_000_000)
    assert o == 2 and factor > 1
    assert dem2["h_orth"].size * o * o <= 1_000_000
    _, o, factor = P.plan_resolution(dem, max_facets=20_000_000)
    assert o == 3 and factor == 1.0


def test_upload_projected_dem():
    """A UTM GeoTIFF with a nodata hole is reprojected to EPSG:4326 and filled."""
    y, x = np.mgrid[0:120, 0:100]
    h = (800 + 3 * x + 2 * y).astype(np.float32)
    h[50:55, 40:45] = -9999
    buf = io.BytesIO()
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=120, width=100, count=1, dtype="float32", nodata=-9999,
                     crs="EPSG:32632", transform=from_origin(680000, 5240000, 30, 30)) as dst:
            dst.write(h, 1)
        buf.write(mf.read())
    dem = P.dem_from_bytes(buf.getvalue())
    assert np.isfinite(dem["h_orth"]).all()
    assert 700 < dem["h_orth"].min() and dem["h_orth"].max() < 1400
    assert 47.0 < dem["lat"].mean() < 47.4 and 11.3 < dem["lon"].mean() < 11.5
    res = P.run_generic(dem, "descending", 38.0, spacing=15.0)
    assert np.isfinite(res["sim"]["map"]["sigma0"]).all()


def test_streamlit_app_loads():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any(b.label == "Run simulation" for b in at.button)


@needs_dem
def test_streamlit_app_runs_generic():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    at.selectbox[0].set_value("Innsbruck centre, small and fast")
    next(b for b in at.button if b.label == "Run simulation").click().run()
    assert not at.exception, at.exception
    assert not at.error
    assert "res" in at.session_state and "zip" in at.session_state
    labels = [mtr.label for mtr in at.metric]
    assert {"Area", "Layover", "Shadow", "Run time"} <= set(labels)
