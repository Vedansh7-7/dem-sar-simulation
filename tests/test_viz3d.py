"""Tests of the interactive 3D terrain view."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sarsim import pipeline as P  # noqa: E402
from sarsim import viz3d  # noqa: E402

DEM = ROOT / "data" / "dem_innsbruck.tif"
CROP = (11.36, 47.24, 11.42, 47.28)
needs_dem = pytest.mark.skipif(not DEM.exists(), reason="data/dem_innsbruck.tif not cached")


@pytest.fixture(scope="module")
def res():
    return P.run_generic(P.dem_from_file(DEM, CROP), "ascending", look_deg=37.0, spacing=10.0)


@needs_dem
@pytest.mark.parametrize("layer", ["sigma0", "classes", "theta", "height"])
def test_every_layer_builds(res, layer):
    fig = viz3d.terrain_figure(res, layer, exaggeration=2.0)
    surf = fig.data[0]
    assert surf.type == "surface"
    z, c = np.asarray(surf.z), np.asarray(surf.surfacecolor)
    assert z.shape == c.shape and z.ndim == 2
    assert np.isfinite(z).all() and np.isfinite(c).all()
    assert max(z.shape) <= 250                        # browser-friendly grid
    assert [t.type for t in fig.data[1:]] == ["cone", "scatter3d"]


@needs_dem
def test_class_layer_values_and_no_radar(res):
    fig = viz3d.terrain_figure(res, "classes", show_radar=False)
    c = np.asarray(fig.data[0].surfacecolor)
    assert set(np.unique(c)) <= {0.0, 1.0, 2.0}
    assert len(fig.data) == 1


def test_look_direction_follows_the_pass():
    asc = {"pass": "ascending", "sensor": {"look_angle_deg": 37.0}}
    desc = {"pass": "descending", "sensor": {"incidence_mid_swath_deg": 39.0}}
    az_a, off_a = viz3d.look_geometry(asc)
    az_d, off_d = viz3d.look_geometry(desc)
    assert 45 < az_a < 135 and off_a == 37.0          # ascending looks east
    assert 225 < az_d < 315 and off_d == 39.0         # descending looks west


def test_real_layer_only_in_real_mode():
    assert "real" not in viz3d.available_layers({"mode": "generic"})
    assert viz3d.available_layers({"mode": "real", "real_map": np.ones((2, 2))})[1] == "real"
