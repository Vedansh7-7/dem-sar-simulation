"""DEM-based SAR simulator: Streamlit GUI.

Launch:  uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from sarsim import compare, rdsim  # noqa: E402
from sarsim import pipeline as P  # noqa: E402
from sarsim.generic import DEFAULT_HEADING, H_S1, generic_geometry  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:  # pragma: no cover
    pass

GREEN, LAYOVER, SHADOW, INK, MUTED = "#1F6B3A", "#C8423B", "#2B59A8", "#1C2420", "#6B746F"
CREDIT = ("Vedansh Shrivastava (230003086), supervisor Dr. Unmesh Khati, "
          "Indian Institute of Technology Indore")

st.set_page_config(page_title="DEM SAR Simulator", page_icon=":satellite:", layout="wide",
                   initial_sidebar_state="expanded")

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": "#B8BFBA", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "image.interpolation": "nearest",
})

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
  [data-testid="stMetric"] {background: #F3F5F2; border-radius: 8px; padding: 10px 14px;}
  h1 {font-size: 1.65rem !important; margin-bottom: 0 !important;}
  .subtitle {color: #6B746F; margin-top: -4px; font-size: 0.95rem;}
  .empty {border: 1px dashed #C9D0CB; border-radius: 10px; padding: 36px; color: #6B746F; text-align: center;}
</style>
""", unsafe_allow_html=True)


# =========================================================================== caching
@st.cache_data(max_entries=4, show_spinner=False)
def cached_dem_bytes(raw: bytes):
    return P.dem_from_bytes(raw)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_dem_bbox(bbox: tuple):
    return P.dem_from_bbox(bbox)


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_generic_geometry(dem_key, look, heading, spacing, H, max_facets, _dem):
    dem2, o, factor = P.plan_resolution(_dem, max_facets)
    return dem2, o, factor, generic_geometry(dem2, look, heading, o, H, spacing)


@st.cache_resource(max_entries=4, ttl=1800, show_spinner=False)
def cached_find_scene(bbox, dates, state):
    return P.find_scene(bbox, dates, state)


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_real_geometry(item_id, dem_key, max_facets, _item, _dem):
    dem2, o, factor = P.plan_resolution(_dem, max_facets)
    prod, s0, meta = P.load_scene(_item, dem2, P.scene_key(P.dem_bbox(_dem)))
    geo = rdsim.facet_geometry(prod, dem2, meta["geoid_N"], oversample=o)
    return (_item, prod, s0, meta), (dem2, o, factor, geo)


# =========================================================================== figures
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def oriented(img, state):
    """Radar geometry displayed roughly north up: ascending flips rows, descending flips columns."""
    return img[::-1, :] if state == "ascending" else img[:, ::-1]


def radar_extent(shape, dx, dy, state):
    w, h = shape[1] * dx / 1e3, shape[0] * dy / 1e3
    return [0, w, 0, h] if state == "ascending" else [w, 0, h, 0]


def map_extent(dem):
    b = P.dem_bbox(dem)
    return [b[0], b[2], b[1], b[3]]


def map_aspect(dem):
    return 1 / np.cos(np.radians(float(np.mean(dem["lat"]))))


def db_limits(a, lo=2, hi=98):
    v = compare.db(a[np.isfinite(a) & (a > 0)])
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (-25.0, 5.0)


def fig_map_image(dem, arr, title, cmap, vmin, vmax, cbar):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(arr, extent=map_extent(dem), cmap=cmap, vmin=vmin, vmax=vmax, aspect=map_aspect(dem))
    ax.set(title=title, xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
    fig.tight_layout()
    return fig


def fig_radar_image(img, dx, dy, state, title, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = matplotlib.colormaps["gray"].with_extremes(bad="#F3F5F2")
    shown = np.where(np.isfinite(img) & (img > 0), compare.db(img), np.nan)   # outside the footprint
    im = ax.imshow(oriented(shown, state), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=radar_extent(img.shape, dx, dy, state), aspect="equal")
    ax.set(title=title, xlabel="Ground range [km]", ylabel="Azimuth [km]")
    fig.colorbar(im, ax=ax, shrink=0.85, label="sigma0 [dB]")
    fig.tight_layout()
    return fig


def fig_hillshade(dem):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ext = map_extent(dem)
    ax.imshow(P.hillshade(dem), cmap="gray", extent=ext, aspect=map_aspect(dem))
    im = ax.imshow(dem["h_orth"], cmap="terrain", alpha=0.35, extent=ext, aspect=map_aspect(dem))
    ax.set(title="Terrain (hillshade with height tint)", xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Height [m]")
    fig.tight_layout()
    return fig


def fig_masks(dem, m):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ext = map_extent(dem)
    ax.imshow(P.hillshade(dem), cmap="gray", extent=ext, aspect=map_aspect(dem), vmin=-0.3, vmax=1.1)
    ov = np.zeros(m["layover"].shape + (4,))
    ov[m["layover"]] = matplotlib.colors.to_rgba(LAYOVER, 0.85)
    ov[m["shadow"]] = matplotlib.colors.to_rgba(SHADOW, 0.85)
    ax.imshow(ov, extent=ext, aspect=map_aspect(dem))
    ax.legend(handles=[Patch(color=LAYOVER, label=f"Layover {m['layover'].mean() * 100:.1f} %"),
                       Patch(color=SHADOW, label=f"Shadow {m['shadow'].mean() * 100:.1f} %")],
              loc="lower right", frameon=True, framealpha=0.9, fontsize=8)
    ax.set(title="Layover and radar shadow (map geometry)", xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    fig.tight_layout()
    return fig


def fig_hist(theta, ref_angle, ref_label):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(theta[np.isfinite(theta)].ravel(), bins=90, range=(0, 90), color=GREEN, alpha=0.9,
            edgecolor="white", linewidth=0.3)
    if ref_angle is not None:
        ax.axvline(ref_angle, color=INK, ls=":", lw=1.2)
        ax.text(ref_angle + 1, ax.get_ylim()[1] * 0.92, ref_label, color=INK, fontsize=8)
    ax.set(title="Distribution of local incidence angle", xlabel="Local incidence angle [deg]",
           ylabel="DEM cells", xlim=(0, 90))
    fig.tight_layout()
    return fig


def fig_curve(res):
    m = res["sim"]["map"]
    kw = P.MODELS[res["model"]["key"]][1]
    th = np.linspace(0, 89, 300)
    law = compare.db(rdsim.sigma0_model(np.cos(np.radians(th)), **kw))
    good = (~m["shadow"]) & (~m["layover"]) & np.isfinite(m["sigma0"]) & (m["sigma0"] > 0)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(th, law, color=GREEN, lw=2, label=f"Model: {res['model']['label']}")
    c, v = P.binned_curve(m["theta_loc"], compare.db(m["sigma0"]), good)
    ax.plot(c, v, "o", ms=4, color=MUTED, label="Simulated image (median per 2 deg)")
    if res["mode"] == "real":
        rm = res["real_map"]
        good_r = good & np.isfinite(rm) & (rm > 0)
        c, v = P.binned_curve(m["theta_loc"], compare.db(np.where(good_r, rm, 1)) - res["eval"]["bias"], good_r)
        ax.plot(c, v, "s", ms=4, color="#B5651D", label="Real Sentinel-1 (level matched)")
    ax.set(title="sigma0 versus local incidence angle", xlabel="Local incidence angle [deg]",
           ylabel="sigma0 [dB]", xlim=(0, 90), ylim=(-30, 12))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def fig_scatter(res):
    ev = res["eval"]
    a = compare.db(ev["real_ml"])[ev["valid"]]
    b = compare.db(ev["sim_reg"])[ev["valid"]] + ev["bias"]
    lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    h = ax.hist2d(b, a, bins=160, range=[[lo, hi], [lo, hi]], cmap="Greens", norm=LogNorm(), rasterized=True)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--")
    ax.set(title="Real versus simulated (pixel density)", xlabel="Simulated sigma0 [dB]",
           ylabel="Real sigma0 [dB]", aspect="equal")
    fig.colorbar(h[3], ax=ax, shrink=0.85, label="Pixels")
    fig.tight_layout()
    return fig


# =========================================================================== header
head_l, head_r = st.columns([14, 1], vertical_alignment="center")
with head_l:
    st.markdown("# DEM-based SAR Simulator")
    st.markdown('<div class="subtitle">Simulate Sentinel-1 backscatter, layover and shadow from any '
                'elevation model</div>', unsafe_allow_html=True)
with head_r:
    with st.popover("ⓘ", help="How to use the app and how the simulation works"):
        st.markdown("""
#### How to use
1. **Terrain**: pick a preset, type a bounding box (Copernicus GLO-30 is downloaded and cached), or upload a DEM GeoTIFF in any projection.
2. **Geometry**: *Generic* works offline anywhere. *Real Sentinel-1* searches a scene over your area and reproduces its exact geometry (internet needed).
3. **Backscatter**: choose a law and, if you like, add speckle.
4. Press **Run simulation**. Explore *Overview* and *Results*, then download the ZIP (GeoTIFFs, radar image, metadata).

#### How the simulation works
- **Facets.** Every DEM cell (oversampled 2 to 3 times) becomes a small tilted surface with a normal vector and a true area. Heights are orthometric; the EGM2008 geoid is added for the real geometry.
- **Projection.** Each facet is imaged at the instant its distance to the sensor is shortest (zero Doppler). Its azimuth time gives the image line and its slant range, converted to ground range, gives the column. *Generic* uses a straight orbit at 693 km over a flat Earth; *Real* uses the scene orbit and its slant to ground range polynomials.
- **Local incidence** is the angle between the radar beam and the facet normal.
- **Shadow** means a facet hidden behind terrain nearer to the sensor. **Layover** means terrain whose echoes arrive mixed with others in the same range cells, because a slope faces the radar more steeply than the look angle.
- **Backscatter.** sigma0 follows a law of the local incidence angle, by default sigma0 = 0.3354 cos^1.98(theta), fitted to Sentinel-1 VV over the Alps.
- **Area normalisation.** Each lit facet adds sigma0 times its area to its image pixel, and the sum is divided by the pixel area, so values are comparable with calibrated Sentinel-1 sigma0.
- **Comparison** (real mode): both images are 5 x 5 multilooked, co-registered to sub-pixel precision and one constant level offset is removed before computing r, SSIM and RMSE.
""")
        st.caption(CREDIT)

# =========================================================================== sidebar
with st.sidebar:
    st.markdown("### Terrain")
    src = st.radio("DEM source", ["Preset", "Bounding box", "Upload GeoTIFF"], horizontal=True,
                   label_visibility="collapsed")
    upload, bbox, preset = None, None, None
    if src == "Preset":
        preset = st.selectbox("Area", list(P.PRESETS), index=1)
        bbox = P.PRESETS[preset]
        st.caption("W {:.2f}, S {:.2f}, E {:.2f}, N {:.2f}".format(*bbox))
    elif src == "Bounding box":
        c1, c2 = st.columns(2)
        w = c1.number_input("West [deg]", -180.0, 180.0, 11.33, 0.01, format="%.4f")
        e = c2.number_input("East [deg]", -180.0, 180.0, 11.45, 0.01, format="%.4f")
        s = c1.number_input("South [deg]", -90.0, 90.0, 47.22, 0.01, format="%.4f")
        n = c2.number_input("North [deg]", -90.0, 90.0, 47.30, 0.01, format="%.4f")
        bbox = (w, s, e, n)
        st.caption(f"Copernicus GLO-30 is downloaded once and cached. Limit {P.MAX_BBOX_DEG:.0f} deg per side.")
    else:
        upload = st.file_uploader("DEM GeoTIFF", type=["tif", "tiff"],
                                  help="Any projection; reprojected to WGS84 latitude and longitude. "
                                       "Heights in metres above the geoid.")

    st.markdown("### Geometry")
    mode = st.radio("Mode", ["Generic Sentinel-1-like", "Real Sentinel-1"], label_visibility="collapsed",
                    help="Generic runs offline anywhere. Real searches Microsoft Planetary Computer for a scene.")
    if mode.startswith("Generic"):
        pass_dir = st.segmented_control("Pass", ["ascending", "descending"], default="ascending", required=True,
                                        format_func=lambda x: f"{x.capitalize()} (looks {'east' if x == 'ascending' else 'west'})")
        pass_dir = pass_dir or "ascending"
        look = st.slider("Look angle [deg]", 20.0, 50.0, 37.0, 0.5,
                         help="Off-nadir angle at the scene centre. Sentinel-1 IW covers about 30 to 46 deg.")
        with st.expander("Sensor details"):
            heading = st.number_input("Heading [deg from north]", -180.0, 360.0, DEFAULT_HEADING[pass_dir], 1.0,
                                      key=f"heading_{pass_dir}", help="Flight direction; the radar looks to the right.")
            spacing = st.select_slider("Pixel spacing [m]", [10, 15, 20, 30], value=10)
            height_km = st.number_input("Orbit height [km]", 400.0, 1000.0, H_S1 / 1e3, 1.0)
    else:
        d = st.date_input("Acquisition dates", (date(2025, 8, 1), date(2025, 9, 30)))
        orbit = st.segmented_control("Pass", ["any", "ascending", "descending"], default="ascending",
                                     required=True)
        orbit = orbit or "any"
        st.caption("Uses the most recent IW GRD scene (VV) that fully covers the area.")

    st.markdown("### Backscatter")
    model = st.selectbox("Model", list(P.MODELS), format_func=lambda k: P.MODELS[k][0])
    speckle = st.toggle("Add speckle", value=False)
    looks = st.slider("Multilook window [pixels]", 1, 9, 4, disabled=not speckle) if speckle else 1

    with st.expander("Memory guard"):
        max_facets = int(st.slider("Max facets [millions]", 1.0, 6.0, 4.0, 0.5) * 1e6)
        st.caption("Larger areas are downsampled to stay below this limit.")

    st.write("")
    run = st.button("Run simulation", type="primary", width="stretch")
    st.caption(CREDIT)


# =========================================================================== run
def get_dem():
    if src == "Upload GeoTIFF":
        if upload is None:
            raise ValueError("Upload a DEM GeoTIFF first.")
        raw = upload.getvalue()
        return cached_dem_bytes(raw), "upload_" + hashlib.sha1(raw).hexdigest()[:12]
    P.check_bbox(bbox)
    return cached_dem_bbox(tuple(float(v) for v in bbox)), "bbox_" + "_".join(f"{v:.4f}" for v in bbox)


if run:
    st.session_state.pop("res", None)
    t_start = time.time()
    try:
        with st.status("Running simulation", expanded=True) as status:
            bar = st.progress(0.0)

            def say(msg, frac):
                status.update(label=msg)
                bar.progress(min(max(frac, 0.0), 1.0))

            say("Loading DEM", 0.05)
            dem, dem_key = get_dem()
            inputs = {"dem_source": src, "preset": preset, "bbox": list(bbox) if bbox else None,
                      "upload_name": upload.name if upload is not None else None,
                      "mode": mode, "model": model, "speckle": speckle, "looks": looks,
                      "max_facets": max_facets}
            if mode.startswith("Generic"):
                say("Projecting DEM facets into the generic sensor geometry", 0.2)
                geom = cached_generic_geometry(dem_key, look, float(heading), float(spacing),
                                               height_km * 1e3, max_facets, dem)
                res = P.run_generic(dem, pass_dir, look, heading, float(spacing), model, speckle, looks,
                                    H=height_km * 1e3, geometry=geom, progress=say)
                inputs.update(pass_dir=pass_dir, look_deg=look, heading_deg=heading,
                              spacing_m=spacing, orbit_height_km=height_km)
            else:
                if not isinstance(d, (tuple, list)) or len(d) != 2:
                    raise ValueError("Choose a start and an end date.")
                dates = f"{d[0].isoformat()}/{d[1].isoformat()}"
                area = tuple(round(v, 6) for v in P.dem_bbox(dem))
                say("Searching Sentinel-1 scenes", 0.1)
                try:
                    item = cached_find_scene(area, dates, None if orbit == "any" else orbit)
                except Exception as exc:
                    raise RuntimeError(f"Could not reach Microsoft Planetary Computer ({exc}). "
                                       "Check the connection or use the Generic mode.") from exc
                if item is None:
                    raise LookupError("No Sentinel-1 IW scene fully covers this area in the chosen dates. "
                                      "Try a longer date range, the other pass, or a smaller area.")
                say(f"Scene {item.id[:32]}: downloading window and geocoding facets", 0.25)
                scene, geom = cached_real_geometry(item.id, dem_key, max_facets, item, dem)
                res = P.run_real(dem, scene, model, speckle, looks, geometry=geom, progress=say)
                inputs.update(dates=dates, orbit_state=orbit)
            res["runtime_s"] = time.time() - t_start
            say("Packaging downloads", 0.95)
            st.session_state["res"] = res
            st.session_state["inputs"] = inputs
            st.session_state["zip"] = P.build_zip(res, inputs)
            status.update(label=f"Finished in {res['runtime_s']:.1f} s", state="complete", expanded=False)
    except LookupError as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"{type(exc).__name__}: {exc}")

res = st.session_state.get("res")

# =========================================================================== tabs
tab_over, tab_res = st.tabs(["Overview", "Results"])

if res is None:
    for t in (tab_over, tab_res):
        with t:
            st.markdown('<div class="empty">Choose terrain and geometry in the sidebar, then press '
                        '<b>Run simulation</b>.<br>The Innsbruck presets work offline with the cached DEM.</div>',
                        unsafe_allow_html=True)
    st.stop()

dem, sim, m = res["dem"], res["sim"], res["sim"]["map"]
stats = P.dem_stats(dem)

with tab_over:
    if res["downsample_factor"] > 1:
        st.info(f"Large area: the DEM was downsampled by a factor of {res['downsample_factor']:.2f} "
                f"to {stats['shape'][1]} x {stats['shape'][0]} cells to stay below "
                f"{st.session_state['inputs']['max_facets'] / 1e6:.1f} million facets.")
    c = st.columns(6)
    c[0].metric("Area", f"{stats['width_km']:.1f} x {stats['height_km']:.1f} km")
    c[1].metric("DEM grid", f"{stats['shape'][1]} x {stats['shape'][0]}",
                help=f"Cell size {stats['pixel_m'][0]:.0f} x {stats['pixel_m'][1]:.0f} m, "
                     f"oversampled {res['oversample']} times, {res['n_facets'] / 1e6:.2f} million facets")
    c[2].metric("Height range", f"{stats['h_min']:.0f} to {stats['h_max']:.0f} m")
    c[3].metric("Layover", f"{res['summary']['layover_pct']:.1f} %")
    c[4].metric("Shadow", f"{res['summary']['shadow_pct']:.1f} %")
    c[5].metric("Run time", f"{res['runtime_s']:.1f} s")

    left, right = st.columns([3, 2])
    with left:
        show(fig_hillshade(dem))
    with right:
        sen = res["sensor"]
        st.markdown("##### Configuration")
        if res["mode"] == "generic":
            rows = {"Mode": "Generic Sentinel-1-like", "Pass": sen["pass"].capitalize(),
                    "Look angle": f"{sen['look_angle_deg']:.1f} deg",
                    "Heading": f"{sen['heading_deg']:.1f} deg",
                    "Orbit height": f"{sen['orbit_height_m'] / 1e3:.0f} km",
                    "Pixel spacing": f"{sen['pixel_spacing_m']:.0f} m"}
        else:
            rows = {"Mode": "Real Sentinel-1", "Scene": sen["scene_id"], "Platform": sen["platform"],
                    "Acquired": str(sen["datetime"])[:19], "Pass": str(sen["pass"]).capitalize(),
                    "Incidence (mid swath)": f"{sen['incidence_mid_swath_deg']:.1f} deg",
                    "Geoid undulation": f"{sen['geoid_N_m']:.1f} m"}
        rows.update({"Backscatter": res["model"]["label"],
                     "Speckle": f"{res['speckle']['looks']} x {res['speckle']['looks']} looks"
                     if res["speckle"]["enabled"] else "off",
                     "Extent": "W {:.3f}, S {:.3f}, E {:.3f}, N {:.3f}".format(*stats["bbox"])})
        st.table({"Parameter": list(rows), "Value": [str(v) for v in rows.values()]}, hide_index=True)

with tab_res:
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    top_l.markdown("##### Simulated backscatter")
    top_r.download_button("Download results (ZIP)", st.session_state["zip"],
                           file_name=f"sar_sim_{res['mode']}_{res['pass']}.zip", mime="application/zip", on_click="ignore",
                           width="stretch",
                           help="GeoTIFFs in map geometry (sigma0 linear and dB, local incidence, layover, "
                                "shadow, DEM), the radar image as GeoTIFF and NPY, and metadata.json")
    state = res["pass"] if res["pass"] in ("ascending", "descending") else "ascending"

    if res["mode"] == "real":
        met = res["metrics"]
        mc = st.columns(5)
        mc[0].metric("Pearson r", f"{met['pearson_r']:.3f}")
        mc[1].metric("SSIM", f"{met['ssim']:.3f}")
        mc[2].metric("RMSE", f"{met['rmse_db_bias_removed']:.2f} dB", help="Bias removed")
        mc[3].metric("Level offset", f"{met['level_offset_db']:+.2f} dB",
                     help="Constant added to the simulation to match the real level")
        mc[4].metric("Residual shift", f"{met['shift_lines']:+.1f}, {met['shift_pixels']:+.1f} px",
                     help="Sub-pixel co-registration shift (lines, pixels)")
        ev = res["eval"]
        prod_dx, prod_dy = res["sensor"]["pixel_spacing_m"]
        vmin, vmax = db_limits(ev["real_ml"][ev["valid"]])
        c1, c2 = st.columns(2)
        with c1:
            show(fig_radar_image(np.where(ev["valid"], ev["real_ml"], np.nan), prod_dx, prod_dy, state,
                                 "Real Sentinel-1 sigma0 (5 x 5 looks)", vmin, vmax))
        with c2:
            sim_show = np.where(ev["valid"], ev["sim_reg"] * 10 ** (ev["bias"] / 10), np.nan)
            show(fig_radar_image(sim_show, prod_dx, prod_dy, state,
                                 "Simulated sigma0 (5 x 5 looks, co-registered)", vmin, vmax))
        c1, c2 = st.columns(2)
        with c1:
            show(fig_scatter(res))
        with c2:
            show(fig_curve(res))
        st.markdown("##### Simulation products")

    dx = res["sensor"]["pixel_spacing_m"]
    dx, dy = (dx, dx) if np.isscalar(dx) else dx
    vmin, vmax = db_limits(res["radar_sigma0"])
    c1, c2 = st.columns(2)
    with c1:
        show(fig_radar_image(res["radar_sigma0"], dx, dy, state, "Simulated sigma0, radar geometry", vmin, vmax))
    with c2:
        show(fig_map_image(dem, compare.db(m["sigma0"]), "Simulated sigma0, map geometry", "gray",
                           vmin, vmax, "sigma0 [dB]"))
    c1, c2 = st.columns(2)
    with c1:
        show(fig_masks(dem, m))
    with c2:
        show(fig_map_image(dem, m["theta_loc"], "Local incidence angle", "cividis", 0, 90, "deg"))
    c1, c2 = st.columns(2)
    with c1:
        ref = res["sensor"].get("look_angle_deg", res["sensor"].get("incidence_mid_swath_deg"))
        show(fig_hist(m["theta_loc"], ref, "look angle" if res["mode"] == "generic" else "mid swath"))
    with c2:
        if res["mode"] == "generic":
            show(fig_curve(res))
        else:
            st.caption("Metrics use the noise-free simulation. Radar images are shown roughly north up.")
