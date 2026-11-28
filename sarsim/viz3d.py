"""Interactive 3D terrain view of a simulation result (Plotly).

The DEM is shown as a surface in a local metric frame (east, north in km; height in m, with
vertical exaggeration) and one map-geometry layer is draped over it: simulated sigma0, the
layover and shadow classes, the local incidence angle, the height, or (real mode) the real
Sentinel-1 sigma0 back-geocoded to the DEM grid. A cone marks the radar look direction.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

LAYERS = {
    "sigma0": "Simulated sigma0 [dB]",
    "classes": "Layover and shadow",
    "theta": "Local incidence angle [deg]",
    "height": "Height [m]",
    "real": "Real Sentinel-1 sigma0 [dB]",
}
LAYOVER, SHADOW, NEUTRAL = "#C8423B", "#2B59A8", "#C9CFCA"
# flight heading [deg from north] used when the result does not record one (real mode)
DEFAULT_HEADING = {"ascending": 348.0, "descending": 192.0}


def available_layers(res: dict) -> list[str]:
    keys = ["sigma0", "classes", "theta", "height"]
    if res.get("mode") == "real" and res.get("real_map") is not None:
        keys.insert(1, "real")
    return keys


def _db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-5))


def _stride(shape, max_side):
    return max(1, int(np.ceil(max(shape) / max_side)))


def look_geometry(res: dict) -> tuple[float, float]:
    """(look azimuth from north, off-nadir angle) of the radar for the result, in degrees."""
    sen = res.get("sensor", {})
    heading = sen.get("heading_deg")
    if heading is None:
        heading = DEFAULT_HEADING.get(str(res.get("pass", "ascending")).lower(), 348.0)
    off_nadir = sen.get("look_angle_deg") or sen.get("incidence_mid_swath_deg") or 37.0
    return (float(heading) + 90.0) % 360.0, float(off_nadir)   # right-looking sensor


def terrain_figure(res: dict, layer: str = "sigma0", exaggeration: float = 1.5, max_side: int = 250,
                   show_radar: bool = True) -> go.Figure:
    """Build the 3D figure. ``res`` is the result dict of ``pipeline.run_generic``/``run_real``."""
    dem, m = res["dem"], res["sim"]["map"]
    s = _stride(dem["h_orth"].shape, max_side)
    h = np.asarray(dem["h_orth"], float)[::s, ::s]
    lat, lon = np.asarray(dem["lat"])[::s], np.asarray(dem["lon"])[::s]
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x = (lon - lon0) * 111.320 * np.cos(np.radians(lat0))          # km east
    y = (lat - lat0) * 111.132                                     # km north
    sub = lambda a: np.asarray(a)[::s, ::s][: h.shape[0], : h.shape[1]]  # noqa: E731

    if layer == "sigma0":
        c = _db(sub(m["sigma0"]))
        lo, hi = np.nanpercentile(c, [2, 98])
        kw = dict(colorscale="Gray", cmin=lo, cmax=hi, colorbar=dict(title="dB", thickness=14))
    elif layer == "real":
        c = _db(np.nan_to_num(sub(res["real_map"]), nan=1e-5))
        lo, hi = np.nanpercentile(c[c > -49], [2, 98]) if np.any(c > -49) else (-25, 0)
        kw = dict(colorscale="Gray", cmin=lo, cmax=hi, colorbar=dict(title="dB", thickness=14))
    elif layer == "classes":
        c = np.zeros(h.shape)
        c[sub(m["layover"]).astype(bool)] = 1
        c[sub(m["shadow"]).astype(bool)] = 2
        kw = dict(colorscale=[[0, NEUTRAL], [0.33, NEUTRAL], [0.34, LAYOVER], [0.66, LAYOVER],
                              [0.67, SHADOW], [1, SHADOW]], cmin=0, cmax=2,
                  colorbar=dict(title="", tickvals=[0.33, 1, 1.67], ticktext=["none", "layover", "shadow"],
                                thickness=14))
    elif layer == "theta":
        c = sub(m["theta_loc"])
        kw = dict(colorscale="Cividis", cmin=0, cmax=90, colorbar=dict(title="deg", thickness=14))
    elif layer == "height":
        c = h
        kw = dict(colorscale="Earth", colorbar=dict(title="m", thickness=14))
    else:
        raise ValueError(f"unknown layer {layer!r}")

    shaded = layer in ("height", "classes")        # radar layers already carry the terrain shading
    surf = go.Surface(
        x=x, y=y, z=h, surfacecolor=c, showscale=True, name=LAYERS[layer],
        lighting=dict(ambient=0.55 if shaded else 0.95, diffuse=0.6 if shaded else 0.1, specular=0.05,
                      roughness=0.9),
        hovertemplate="E %{x:.2f} km<br>N %{y:.2f} km<br>h %{z:.0f} m<br>%{surfacecolor:.2f}<extra></extra>",
        **kw)
    traces = [surf]

    if show_radar:
        az, off = look_geometry(res)
        u = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])     # ground look direction
        half = 0.5 * max(np.ptp(x), np.ptp(y))
        start = -u * half * 0.95
        z_top = float(np.nanmax(h)) + 0.35 * float(np.ptp(h) + 1)
        vec = np.array([u[0] * np.sin(np.radians(off)), u[1] * np.sin(np.radians(off)),
                        -np.cos(np.radians(off))])
        scale_xy = half * 0.25
        traces.append(go.Cone(x=[start[0]], y=[start[1]], z=[z_top], u=[vec[0] * scale_xy], v=[vec[1] * scale_xy],
                              w=[vec[2] * scale_xy * 1000 / exaggeration], anchor="tail", sizemode="absolute",
                              sizeref=half * 0.12, colorscale=[[0, "#E08A1E"], [1, "#E08A1E"]], showscale=False,
                              hoverinfo="skip", name="Radar look direction"))
        traces.append(go.Scatter3d(x=[start[0]], y=[start[1]], z=[z_top], mode="text", text=["Radar"],
                                   textposition="top center", textfont=dict(color="#B86A0C", size=13),
                                   hoverinfo="skip", showlegend=False))

    zr = float(np.ptp(h) + 1)
    xr = float(max(np.ptp(x), 1e-3))
    yr = float(max(np.ptp(y), 1e-3))
    ref = max(xr, yr)
    fig = go.Figure(traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=640, showlegend=False, paper_bgcolor="white",
        scene=dict(
            xaxis=dict(title="East [km]", backgroundcolor="#F3F5F2", gridcolor="#DDE2DE"),
            yaxis=dict(title="North [km]", backgroundcolor="#F3F5F2", gridcolor="#DDE2DE"),
            zaxis=dict(title="Height [m]", backgroundcolor="#F3F5F2", gridcolor="#DDE2DE"),
            aspectmode="manual",
            aspectratio=dict(x=xr / ref, y=yr / ref, z=min(1.0, zr / 1000 / ref * exaggeration)),
            camera=dict(eye=dict(x=-1.25, y=-1.35, z=0.9)),
        ),
    )
    return fig
