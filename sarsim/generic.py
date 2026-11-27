"""Generic Sentinel-1-like simulation that works anywhere, offline.

A flat-Earth, straight-orbit, right-looking sensor at height H flies along a chosen
heading over a local metric (east, north, up) frame centred on the DEM. Every
(oversampled) DEM facet is projected to the flat reference plane at its slant range
(zero-Doppler: the facet is imaged at its own along-track position), which gives
ground-range / azimuth image coordinates. Local incidence, shadow and layover are
computed exactly as in the real-geometry simulator, and the resulting facet table has
the same layout as ``rdsim.facet_geometry`` so that ``rdsim.render`` can be reused.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .geodesy import WGS84_A, WGS84_E2
from .rdsim import shadow_layover

H_S1 = 693e3                       # Sentinel-1 orbit height above the reference plane [m]
DEFAULT_HEADING = {"ascending": -12.0, "descending": 192.0}   # flight azimuth [deg from north]


def metres_per_degree(lat0: float):
    """Metres per degree of latitude and longitude on the WGS84 ellipsoid at lat0."""
    p = np.radians(lat0)
    w = np.sqrt(1 - WGS84_E2 * np.sin(p) ** 2)
    m_merid = WGS84_A * (1 - WGS84_E2) / w**3
    n_prime = WGS84_A / w
    return np.radians(1) * m_merid, np.radians(1) * n_prime * np.cos(p)


def generic_geometry(dem: dict, look_deg: float = 37.0, heading_deg: float = -12.0,
                     oversample: int = 2, H: float = H_S1, spacing: float = 10.0):
    """Facet geometry of a DEM for a flat-Earth, straight-orbit, right-looking sensor.

    dem         : dict(h_orth, lat, lon) on a regular EPSG:4326 grid (1-D lat/lon axes)
    look_deg    : off-nadir look angle at the scene centre (equals the incidence angle on
                  the flat reference plane)
    heading_deg : flight direction, clockwise from north (the radar looks 90 deg to the right)
    spacing     : ground-range and azimuth pixel spacing of the output image [m]

    Returns a dict compatible with ``rdsim.render`` (line, pixel, R, cos_loc, cos_ell,
    area, shadow, layover, grid_shape, dem_shape, oversample, pixel_area) plus the
    image shape and frame parameters.
    """
    h = np.asarray(dem["h_orth"], np.float64)
    lat, lon = np.asarray(dem["lat"]), np.asarray(dem["lon"])
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    m_lat, m_lon = metres_per_degree(lat0)

    hz = zoom(h, oversample, order=1) if oversample > 1 else h
    latz = np.interp(np.linspace(0, len(lat) - 1, hz.shape[0]), np.arange(len(lat)), lat)
    lonz = np.interp(np.linspace(0, len(lon) - 1, hz.shape[1]), np.arange(len(lon)), lon)
    north = (latz - lat0) * m_lat                       # 1-D, metres
    east = (lonz - lon0) * m_lon
    h_ref = float(np.median(h))
    Z = (hz - h_ref).astype(np.float64)

    # slopes and true facet area in the local ENU frame
    dz_dn, dz_de = np.gradient(Z, north, east)
    S = np.sqrt(1 + dz_de**2 + dz_dn**2)
    cell = abs(north[1] - north[0]) * abs(east[1] - east[0]) if min(hz.shape) > 1 else 1.0
    area = cell * S

    hd = np.radians(heading_deg)
    f_e, f_n = np.sin(hd), np.cos(hd)                   # along-track unit vector
    l_e, l_n = np.cos(hd), -np.sin(hd)                  # right-looking ground-range direction
    E, N = np.meshgrid(east, north)
    x_along = E * f_e + N * f_n
    y_c = H * np.tan(np.radians(look_deg))
    Y = y_c + E * l_e + N * l_n                         # ground distance from the ground track
    Hz = H - Z
    R = np.sqrt(Y**2 + Hz**2)
    alpha = np.arctan2(Y, Hz)                           # off-nadir look angle
    cos_loc = (Y * (l_e * dz_de + l_n * dz_dn) + Hz) / (R * S)    # -u . n
    cos_ell = Hz / R
    y_ref = np.sqrt(np.maximum(R**2 - H**2, 0.0))       # where the SAR puts the facet on z = 0
    del E, N, dz_dn, dz_de, S, Hz

    x0, y0 = x_along.min() - 2 * spacing, y_ref.min() - 2 * spacing
    line = (x_along - x0) / spacing
    pixel = (y_ref - y0) / spacing
    shape = (int(np.ceil(line.max())) + 3, int(np.ceil(pixel.max())) + 3)
    shadow, layover = shadow_layover(line.ravel(), Y.ravel(), alpha.ravel(), R.ravel(),
                                     tol_alpha=1e-6, tol_R=0.5)
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).ravel()  # noqa: E731
    return {
        "line": f32(line), "pixel": f32(pixel), "R": f32(R),
        "cos_loc": f32(cos_loc), "cos_ell": f32(cos_ell), "area": f32(area),
        "shadow": shadow, "layover": layover,
        "grid_shape": hz.shape, "dem_shape": h.shape, "oversample": oversample,
        "pixel_area": spacing * spacing,
        "image_shape": shape,
        "frame": {"lat0": lat0, "lon0": lon0, "h_ref_m": h_ref, "y_center_m": y_c,
                  "x0_m": float(x0), "yref0_m": float(y0), "spacing_m": spacing,
                  "H_m": H, "look_deg": look_deg, "heading_deg": heading_deg},
    }
