"""WGS84 geodesy helpers (geodetic <-> ECEF, local up vector, geoid undulation)."""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
C = 299_792_458.0  # speed of light [m/s]


def llh2ecef(lat, lon, h):
    """Geodetic latitude/longitude [deg] and ellipsoidal height [m] -> ECEF [m] (..., 3)."""
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    x = (n + h) * np.cos(la) * np.cos(lo)
    y = (n + h) * np.cos(la) * np.sin(lo)
    z = (n * (1 - WGS84_E2) + h) * np.sin(la)
    return np.stack([x, y, z], axis=-1)


def up_vector(lat, lon):
    """Ellipsoid normal (local 'up') unit vector in ECEF."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def geoid_undulation(lat: float, lon: float) -> float:
    """EGM2008 geoid height N [m] (ellipsoidal h = orthometric H + N).

    Uses PROJ's network grids when reachable; otherwise falls back to a regional
    constant (value for the Innsbruck area) and says so.
    """
    try:
        import pyproj

        pyproj.network.set_network_enabled(True)
        tr = pyproj.Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
        _, _, h = tr.transform(lon, lat, 0.0)
        if np.isfinite(h) and abs(h) > 1e-3:
            return float(h)
    except Exception:  # pragma: no cover - network/grid issues
        pass
    print("geoid_undulation: PROJ grid unavailable, using regional constant 47.5 m")
    return 47.5
