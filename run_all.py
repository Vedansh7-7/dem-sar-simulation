"""Phases 1-3: data -> DEM-based SAR simulation -> comparison with real Sentinel-1.

Experiments
  E1  backscatter models (const, cos, cos^2, Muhleman, cos^n fitted to the data) - ascending
  E2  second viewing geometry: descending scene, model fitted on ascending (cross-validation)
  E3  geometric validation: geocoding vs ESA grid, residual offsets, real sigma0 per predicted class
  E4  sensitivity: no geoid correction, no DEM oversampling, 90 m DEM
  E5  land cover (ESA WorldCover): where does a DEM-only simulation fail?

Usage:  uv run python run_all.py         (first run downloads ~60 MB, total ~15 min)
Outputs: results/*.json|csv, figures/pipeline/*.png
"""
from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap  # noqa: E402
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes  # noqa: E402

import sarsim  # noqa: E402
from sarsim import compare, data, rdsim  # noqa: E402
from sarsim.geodesy import C, llh2ecef  # noqa: E402

FIG = ROOT / "figures" / "pipeline"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)
ML = 5
T0 = time.time()
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "image.cmap": "gray"})
db = compare.db


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- helpers
def esa_grid_check(prod, scene_dir):
    """Our range-Doppler geocoding vs the geolocation grid in ESA's annotation."""
    import xml.etree.ElementTree as ET

    pts = ET.fromstring((scene_dir / "annotation.xml").read_text(encoding="utf-8")).findall(
        "geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    t_esa = np.array([prod.sec(np.datetime64(p.find("azimuthTime").text, "ns")) for p in pts])
    srt = np.array([float(p.find("slantRangeTime").text) for p in pts])
    g = prod.grid
    r = prod.geocode(llh2ecef(g["latitude"], g["longitude"], g["height"]))
    return {
        "n_points": int(len(pts)),
        "dt_azimuth_ms_max": float(np.abs(r["t"] - t_esa).max() * 1e3),
        "dline_equiv_max": float(np.abs(r["t"] - t_esa).max() / prod.dt_az),
        "dslant_range_m_max": float(np.abs(r["R"] - srt * C / 2).max()),
        "dpixel_max": float(np.abs(r["pixel"] - g["pixel"]).max()),
        "dpixel_rms": float(np.sqrt(np.mean((r["pixel"] - g["pixel"]) ** 2))),
        "_pixel": g["pixel"], "_dpixel": r["pixel"] - g["pixel"], "_dline": (r["t"] - t_esa) / prod.dt_az,
    }


def evaluate(s0, sim):
    """Multilook, co-register, calibrate the absolute level with one constant, and score."""
    real_ml = compare.multilook(s0, ML)
    sim_ml = compare.multilook(sim["sigma0"], ML)
    foot = binary_erosion(binary_fill_holes(binary_closing(sim["footprint"], iterations=3)), iterations=8)
    shift, box = compare.coregister(db(real_ml), db(sim_ml), foot)
    sim_reg = compare.apply_shift(sim_ml, shift)
    foot_reg = compare.apply_shift(foot.astype(float), shift) > 0.99
    valid = foot_reg & (real_ml > 0) & (sim_reg > 0)
    bias = float(np.median(db(real_ml)[valid] - db(sim_reg)[valid]))
    met = compare.metrics(db(real_ml), db(sim_reg) + bias, valid, box)
    met.update(level_offset_db=bias, shift_lines=float(shift[0]), shift_pixels=float(shift[1]))
    return met, {"real_ml": real_ml, "sim_reg": sim_reg, "valid": valid, "shift": shift,
                 "bias": bias, "foot": foot_reg}


def to_map(ev, sim):
    m = sim["map"]
    real_map = compare.sample_radar_to_map(ev["real_ml"], m["row"] + ev["shift"][0], m["col"] + ev["shift"][1])
    sim_map = compare.sample_radar_to_map(compare.multilook(sim["sigma0"], ML), m["row"], m["col"])
    good = (~m["shadow"]) & (~m["layover"]) & np.isfinite(real_map) & (real_map > 0) & (sim_map > 0)
    return real_map, sim_map, good


def binned_curve(theta, val_db, good, step=2.0, min_count=300):
    edges = np.arange(0, 90 + step, step)
    cen, med, q25, q75 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = good & (theta >= a) & (theta < b)
        if m.sum() >= min_count:
            v = val_db[m]
            cen.append((a + b) / 2)
            med.append(np.median(v))
            q25.append(np.percentile(v, 25))
            q75.append(np.percentile(v, 75))
    return np.array(cen), np.array(med), np.array(q25), np.array(q75)


def fit_cosn(cen, med, lo=5, hi=80):
    """Least-squares fit of  sigma0[dB] = 10 log10 k + n * 10 log10 cos(theta)."""
    m = (cen >= lo) & (cen <= hi)
    x = 10 * np.log10(np.cos(np.radians(cen[m])))
    n, a = np.polyfit(x, med[m], 1)
    return float(n), float(10 ** (a / 10))


def oriented(img, state):
    """Display radar-geometry images north-up / east-right for either pass direction."""
    if state == "ascending":       # line -> north, pixel -> east
        return img[::-1, :]
    return img[:, ::-1]            # descending: line -> south, pixel -> west


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ============================================================================ PHASE 1
log("Phase 1: data")
dem = data.load_dem(data.get_dem(sarsim.AOI_BBOX, sarsim.AOI_NAME))
lc = data.worldcover_on_dem(data.get_worldcover(sarsim.AOI_BBOX, sarsim.AOI_NAME), dem)
scenes = {}
for state in ("ascending", "descending"):
    item = data.find_s1(sarsim.AOI_BBOX, sarsim.S1_DATES, orbit_state=state)
    sd = data.get_s1(item, dem)
    prod, s0, meta = data.load_s1(sd)
    scenes[state] = {"item": item, "dir": sd, "prod": prod, "s0": s0, "meta": meta}
    log(f"  {state:10s}: {item.id}  window {s0.shape}")
RESULTS = {"aoi_bbox": sarsim.AOI_BBOX, "dem_shape": list(dem["h_orth"].shape),
           "dem_height_range_m": [float(dem["h_orth"].min()), float(dem["h_orth"].max())], "scenes": {}}
for state, sc in scenes.items():
    p = sc["prod"]
    RESULTS["scenes"][state] = {
        "id": sc["item"].id, "platform": sc["meta"]["platform"], "datetime": sc["meta"]["datetime"],
        "pass": p.pass_direction, "incidence_mid_deg": p.inc_mid, "window_shape": list(sc["s0"].shape),
        "pixel_spacing_m": [p.dgr, p.daz], "geoid_N_m": sc["meta"]["geoid_N"],
        "orbit_fit_vel_rms_mm_s": p.orbit.vel_rms * 1e3,
    }

# ============================================================================ PHASE 2+3
# ---- E1: backscatter models on the ascending scene ----
log("E1: facet geometry (ascending)")
A = scenes["ascending"]
geoA = rdsim.facet_geometry(A["prod"], dem, A["meta"]["geoid_N"], oversample=3)
log("E1: render Muhleman + fit cos^n")
simA = rdsim.render(geoA, A["meta"]["row0"], A["meta"]["col0"], A["s0"].shape, "muhleman")
metA_mu, evA_mu = evaluate(A["s0"], simA)
realA_map, _, goodA = to_map(evA_mu, simA)
cenA, medA, q25A, q75A = binned_curve(simA["map"]["theta_loc"], db(np.where(goodA, realA_map, np.nan)), goodA)
n_fit, k_fit = fit_cosn(cenA, medA)
log(f"   fitted sigma0 = {k_fit:.4f} cos^{n_fit:.2f}(theta_loc)")
RESULTS["fitted_model"] = {"n": n_fit, "k": k_fit, "k_db": 10 * np.log10(k_fit), "fit_on": "ascending"}

MODELS = [("const", "Constant", dict(model="const", k=0.1)), ("cos", "cos θ", dict(model="cos", n=1)),
          ("lambert", "cos² θ (Lambert)", dict(model="lambert")), ("muhleman", "Muhleman", dict(model="muhleman")),
          ("fitted", f"Fitted cos^{n_fit:.2f} θ", dict(model="cos", n=n_fit, k=k_fit))]
model_rows, model_imgs = [], {}
for key, label, kw in MODELS:
    sim_ = simA if key == "muhleman" else rdsim.render(geoA, A["meta"]["row0"], A["meta"]["col0"], A["s0"].shape, **kw)
    met, ev = (metA_mu, evA_mu) if key == "muhleman" else evaluate(A["s0"], sim_)
    model_rows.append({"model": key, "label": label, **met})
    model_imgs[key] = (label, db(ev["sim_reg"]) + ev["bias"], ev)
    log(f"   {label:22s} r={met['pearson_r']:.3f}  SSIM={met['ssim']:.3f}  RMSE={met['rmse_db_bias_removed']:.2f} dB")
RESULTS["E1_models_ascending"] = model_rows
simA_fit = rdsim.render(geoA, A["meta"]["row0"], A["meta"]["col0"], A["s0"].shape, "cos", n_fit, k_fit)
metA, evA = evaluate(A["s0"], simA_fit)
del geoA, simA
gc.collect()

# ---- E2: descending scene with the model fitted on ascending ----
log("E2: descending scene")
D = scenes["descending"]
geoD = rdsim.facet_geometry(D["prod"], dem, D["meta"]["geoid_N"], oversample=3)
simD = rdsim.render(geoD, D["meta"]["row0"], D["meta"]["col0"], D["s0"].shape, "cos", n_fit, k_fit)
metD, evD = evaluate(D["s0"], simD)
simD_mu = rdsim.render(geoD, D["meta"]["row0"], D["meta"]["col0"], D["s0"].shape, "muhleman", with_masks=False)
simD_mu.update({k: simD[k] for k in ("footprint",)})
metD_mu, _ = evaluate(D["s0"], simD_mu)
del geoD, simD_mu
gc.collect()
realD_map, _, goodD = to_map(evD, simD)
cenD, medD, q25D, q75D = binned_curve(simD["map"]["theta_loc"], db(np.where(goodD, realD_map, np.nan)), goodD)
nD, kD = fit_cosn(cenD, medD)
RESULTS["E2_descending"] = {"fitted_on_ascending": metD, "muhleman": metD_mu,
                            "refit_on_descending": {"n": nD, "k": kD}}
log(f"   descending (asc-fitted model) r={metD['pearson_r']:.3f} SSIM={metD['ssim']:.3f}; refit n={nD:.2f}")
RESULTS["E1_E2_summary"] = {"ascending_fitted": metA, "descending_fitted": metD}

# ---- E3: geometric validation ----
log("E3: geometric validation")
grid_checks = {s: esa_grid_check(scenes[s]["prod"], scenes[s]["dir"]) for s in scenes}
RESULTS["E3_geocoding_vs_esa"] = {s: strip(v) for s, v in grid_checks.items()}
class_stats = {}
for state, sim_, ev in (("ascending", simA_fit, evA), ("descending", simD, evD)):
    sh = compare.apply_shift(sim_["radar_shadow"].astype(float), ev["shift"]) > 0.5
    lo = compare.apply_shift(sim_["radar_layover"].astype(float), ev["shift"]) > 0.5
    an = compare.apply_shift(sim_["area_norm"], ev["shift"])
    valid = ev["valid"] | (ev["foot"] & sh)
    rdb = db(ev["real_ml"])
    classes = {"shadow": valid & sh, "layover": valid & lo & ~sh,
               "foreshortened (area > 2x)": valid & ~sh & ~lo & (an > 2),
               "other terrain": valid & ~sh & ~lo & (an <= 2)}
    class_stats[state] = {k: rdb[m] for k, m in classes.items()}
RESULTS["E3_real_sigma0_by_predicted_class"] = {
    s: {k: {"n": int(v.size), "median_db": float(np.median(v)), "p25_db": float(np.percentile(v, 25)),
            "p75_db": float(np.percentile(v, 75))} for k, v in d.items()} for s, d in class_stats.items()}

# ---- E4: sensitivity (ascending, fitted model) ----
log("E4: sensitivity")
sens = [{"case": "baseline (GLO-30, 3x oversampled, geoid)", **metA}]
s0A, rA, cA = A["s0"], A["meta"]["row0"], A["meta"]["col0"]


def holes(sim_):
    f = binary_erosion(binary_fill_holes(binary_closing(sim_["footprint"], iterations=3)), iterations=3)
    return float(np.mean((sim_["area_norm"] == 0)[f] & ~sim_["radar_shadow"][f]))


sens[0]["hole_fraction"] = holes(simA_fit)
g_ = rdsim.facet_geometry(A["prod"], dem, 0.0, oversample=3)
s_ = rdsim.render(g_, rA, cA, s0A.shape, "cos", n_fit, k_fit)
m_, _ = evaluate(s0A, s_)
sens.append({"case": "no geoid correction (N = 0)", **m_, "hole_fraction": holes(s_)})
del g_, s_
gc.collect()
log("   no geoid done")
g_ = rdsim.facet_geometry(A["prod"], dem, A["meta"]["geoid_N"], oversample=1)
s_ = rdsim.render(g_, rA, cA, s0A.shape, "cos", n_fit, k_fit)
m_, _ = evaluate(s0A, s_)
sens.append({"case": "no DEM oversampling", **m_, "hole_fraction": holes(s_)})
holes_img = (s_["area_norm"] == 0) & ~s_["radar_shadow"]
del g_, s_
gc.collect()
log("   no oversampling done")
ny, nx = (np.array(dem["h_orth"].shape) // 3) * 3
dem90 = {"h_orth": dem["h_orth"][:ny, :nx].reshape(ny // 3, 3, nx // 3, 3).mean(axis=(1, 3)),
         "lat": dem["lat"][:ny].reshape(-1, 3).mean(1), "lon": dem["lon"][:nx].reshape(-1, 3).mean(1)}
g_ = rdsim.facet_geometry(A["prod"], dem90, A["meta"]["geoid_N"], oversample=9)
s_ = rdsim.render(g_, rA, cA, s0A.shape, "cos", n_fit, k_fit)
m_, _ = evaluate(s0A, s_)
sens.append({"case": "90 m DEM (GLO-30 aggregated 3x3)", **m_, "hole_fraction": holes(s_)})
del g_
RESULTS["E4_sensitivity"] = sens
for r_ in sens:
    log(f"   {r_['case']:42s} r={r_['pearson_r']:.3f} shift=({r_['shift_lines']:+.2f},{r_['shift_pixels']:+.2f}) holes={r_['hole_fraction']*100:.2f}%")

# ---- E5: land cover ----
log("E5: land cover")
_, simA_map, _ = to_map(evA, simA_fit)
realA_map2, _, goodA2 = to_map(evA, simA_fit)
lc_rows = []
resid = db(realA_map2) - (db(simA_map) + evA["bias"])
for code, (name, colr) in data.WORLDCOVER_CLASSES.items():
    m = goodA2 & (lc == code)
    if m.sum() < 2000:
        continue
    lc_rows.append({"code": code, "class": name, "color": colr, "n": int(m.sum()),
                    "share_pct": float(100 * m.sum() / goodA2.sum()),
                    "mean_residual_db": float(np.mean(resid[m])), "std_residual_db": float(np.std(resid[m])),
                    "pearson_r": float(np.corrcoef(db(realA_map2[m]), db(simA_map[m]))[0, 1])})
    log(f"   {name:15s} {lc_rows[-1]['share_pct']:5.1f}%  residual {lc_rows[-1]['mean_residual_db']:+.2f} dB  r={lc_rows[-1]['pearson_r']:.2f}")
RESULTS["E5_landcover_ascending"] = lc_rows

# ============================================================================ FIGURES
log("figures")
ls = LightSource(azdeg=315, altdeg=45)
mlat = 111_132.0
mlon = 111_320.0 * np.cos(np.radians(dem["lat"].mean()))
hs = ls.hillshade(dem["h_orth"], dx=abs(dem["lon"][1] - dem["lon"][0]) * mlon, dy=abs(dem["lat"][1] - dem["lat"][0]) * mlat)
ext = [dem["lon"][0], dem["lon"][-1], dem["lat"][-1], dem["lat"][0]]
asp = 1 / np.cos(np.radians(dem["lat"].mean()))

# P1 models
fig, ax = plt.subplots(2, 3, figsize=(17, 11))
fA = evA_mu["foot"]
show = lambda img: oriented(np.where(fA, img, np.nan), "ascending")  # noqa: E731
ax[0, 0].imshow(show(db(evA_mu["real_ml"])), vmin=-22, vmax=3)
ax[0, 0].set_title("REAL Sentinel-1 σ⁰ VV (ascending)")
for a, (key, (label, img, ev)) in zip(ax.ravel()[1:], model_imgs.items()):
    row = next(r for r in model_rows if r["model"] == key)
    a.imshow(show(img), vmin=-22, vmax=3)
    a.set_title(f"{label}:  r = {row['pearson_r']:.2f}, SSIM = {row['ssim']:.2f}")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("E1 · Simulated images for five backscatter models vs the real image (radar geometry, north up)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p1_models_images")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
labels = [r["label"] for r in model_rows]
for a, key, ttl in zip(ax, ("pearson_r", "ssim", "rmse_db_bias_removed"),
                       ("Pearson r (dB)", "SSIM", "RMSE after level calibration [dB]")):
    vals = [r[key] for r in model_rows]
    bars = a.bar(range(len(vals)), vals, color=["#bbbbbb", "#8fb8de", "#5a9bd4", "#2b6cb0", "#c0392b"])
    a.set_xticks(range(len(vals)), labels, rotation=20, ha="right")
    a.set_title(ttl)
    for b_, v_ in zip(bars, vals):
        a.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=8)
    a.set_ylim(0, max(vals) * 1.15)
fig.suptitle("E1 · Agreement with the real image per backscatter model (ascending scene)", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p1b_models_metrics")

# P2 fitted model
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
th = np.linspace(1, 89, 200)
for a, (cen, med, q25, q75, nm) in zip(ax, ((cenA, medA, q25A, q75A, "ascending"), (cenD, medD, q25D, q75D, "descending"))):
    a.fill_between(cen, q25, q75, color="k", alpha=0.12, label="real: interquartile range")
    a.plot(cen, med, "ko", ms=4, label="real: median")
    a.plot(th, db(rdsim.sigma0_model(np.cos(np.radians(th)), "cos", n_fit, k_fit)), "r", lw=2,
           label=f"fit (asc): {10*np.log10(k_fit):.1f} dB + {n_fit:.2f}·10log cos θ")
    mu = db(rdsim.sigma0_model(np.cos(np.radians(th)), "muhleman"))
    a.plot(th, mu + RESULTS["E1_models_ascending"][3]["level_offset_db"], "b--", lw=1, label="Muhleman (+level offset)")
    a.set(xlabel="local incidence angle θ_loc [°]", ylabel="σ⁰ VV [dB]", ylim=(-28, 5), title=f"{nm} scene")
    a.legend(fontsize=8)
fig.suptitle("E1/E2 · Real σ⁰ versus local incidence angle (layover & shadow excluded) and the fitted cosⁿ law",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p2_fitted_model")

# P3 ascending vs descending
fig, ax = plt.subplots(2, 3, figsize=(17, 11.5))
for i, (state, sim_, ev, met) in enumerate((("ascending", simA_fit, evA, metA), ("descending", simD, evD, metD))):
    f = ev["foot"]
    ov = np.zeros(f.shape + (4,))
    ov[(compare.apply_shift(sim_["radar_layover"].astype(float), ev["shift"]) > 0.5) & f] = [1, 0, 0, 0.55]
    ov[(compare.apply_shift(sim_["radar_shadow"].astype(float), ev["shift"]) > 0.5) & f] = [0, 0.4, 1, 0.85]
    ax[i, 0].imshow(oriented(np.where(f, db(ev["real_ml"]), np.nan), state), vmin=-22, vmax=3)
    ax[i, 0].set_title(f"REAL — {state} ({scenes[state]['meta']['platform']}, {scenes[state]['meta']['datetime'][:10]})")
    ax[i, 1].imshow(oriented(np.where(f, db(ev["sim_reg"]) + ev["bias"], np.nan), state), vmin=-22, vmax=3)
    ax[i, 1].set_title(f"SIMULATED — r = {met['pearson_r']:.2f}, SSIM = {met['ssim']:.2f}")
    ax[i, 2].imshow(oriented(np.where(f, db(ev["real_ml"]), np.nan), state), vmin=-22, vmax=3)
    ax[i, 2].imshow(oriented(ov, state))
    ax[i, 2].set_title("predicted layover (red) / shadow (blue) on REAL")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("E2 · Two viewing geometries (all panels north up): looking EAST (ascending) and WEST (descending)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p3_ascending_descending")

# P4 geometric validation
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for state, gc in grid_checks.items():
    ax[0].plot(gc["_pixel"], gc["_dpixel"], "o", ms=3, label=f"{state}: range [px]")
    ax[0].plot(gc["_pixel"], gc["_dline"], "x", ms=3, label=f"{state}: azimuth [lines]")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set(xlabel="GRD pixel", ylabel="ours − ESA", ylim=(-0.1, 0.1), title="(a) Geocoding vs ESA geolocation grid")
ax[0].legend(fontsize=7)
for a, state in zip(ax[1:], ("ascending", "descending")):
    d = class_stats[state]
    a.boxplot([v[np.isfinite(v)] for v in d.values()], showfliers=False, widths=0.6)
    a.set_xticks(range(1, len(d) + 1), [f"{k}\n(n={v.size/1e3:.0f}k)" for k, v in d.items()], fontsize=8)
    a.set(ylabel="real σ⁰ [dB]", title=f"({'b' if state == 'ascending' else 'c'}) Real σ⁰ in predicted classes — {state}")
fig.suptitle("E3 · Geometric validation", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p4_geometric_validation")

# P5 sensitivity
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
names = ["baseline", "no geoid", "no oversampling", "90 m DEM"]
ax[0].bar(names, [r["pearson_r"] for r in sens], color="#2b6cb0")
ax[0].set(ylim=(0.5, 0.8), title="(a) Pearson r", ylabel="r")
off = [np.hypot(r["shift_lines"] * 10, r["shift_pixels"] * 10) for r in sens]
ax[1].bar(names, off, color="#c0392b")
ax[1].set(title="(b) Residual misregistration [m]", ylabel="m")
crop = (slice(1700, 2300), slice(500, 1300))          # Nordkette (ascending window, rows = lines)
ax[2].imshow(holes_img[crop], cmap="Reds", origin="lower", interpolation="nearest")
ax[2].set(title="(c) Empty GRD pixels without oversampling (detail)", xticks=[], yticks=[])
for a in ax[:2]:
    a.tick_params(axis="x", labelrotation=15)
fig.suptitle("E4 · Sensitivity of the simulation to processing choices", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p5_sensitivity")

# P6 land cover
codes = sorted(data.WORLDCOVER_CLASSES)
cmap_lc = ListedColormap([data.WORLDCOVER_CLASSES[c][1] for c in codes])
norm_lc = BoundaryNorm(codes + [codes[-1] + 5], cmap_lc.N)
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax[0].imshow(lc, extent=ext, cmap=cmap_lc, norm=norm_lc, aspect=asp, interpolation="nearest")
from matplotlib.patches import Patch  # noqa: E402
ax[0].legend(handles=[Patch(color=r["color"], label=r["class"]) for r in lc_rows], fontsize=7, loc="lower left")
ax[0].set(title="(a) ESA WorldCover 2021 (10 m)", xlabel="longitude [°]", ylabel="latitude [°]")
nm_ = [r["class"] for r in lc_rows]
ax[1].barh(nm_, [r["mean_residual_db"] for r in lc_rows], xerr=[r["std_residual_db"] for r in lc_rows],
           color=[r["color"] for r in lc_rows], edgecolor="k")
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set(xlabel="real − simulated [dB]", title="(b) Mean residual per class (±1σ)")
ax[2].barh(nm_, [r["pearson_r"] for r in lc_rows], color=[r["color"] for r in lc_rows], edgecolor="k")
ax[2].set(xlabel="Pearson r (map geometry)", title="(c) Agreement within each class")
for i_, r in enumerate(lc_rows):
    ax[2].text(r["pearson_r"] + 0.01, i_, f"{r['share_pct']:.0f}% of area", va="center", fontsize=7)
ax[2].set_xlim(0, 1)
fig.suptitle("E5 · Land cover explains what the DEM cannot", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p6_landcover")

# P7 residual maps (map geometry)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].imshow(hs, extent=ext, aspect=asp)
ax[0].set(title="(a) DEM hillshade", xlabel="longitude [°]", ylabel="latitude [°]")
for a, (state, ev, sim_) in zip(ax[1:], (("ascending", evA, simA_fit), ("descending", evD, simD))):
    rm, sm, gd = to_map(ev, sim_)
    res_ = np.where(np.isfinite(rm) & (rm > 0) & (sm > 0), db(rm) - db(sm) - ev["bias"], np.nan)
    a.imshow(hs, extent=ext, aspect=asp)
    im = a.imshow(res_, extent=ext, cmap="RdBu_r", vmin=-8, vmax=8, alpha=0.85, aspect=asp)
    a.set(title=f"({'b' if state == 'ascending' else 'c'}) real − simulated [dB], {state}", xlabel="longitude [°]")
plt.colorbar(im, ax=ax[1:], label="dB", shrink=0.8)
fig.suptitle("E5 · Residual maps in map geometry: red = brighter than the DEM predicts (city, rock), blue = darker",
             fontsize=12, fontweight="bold")
save(fig, "p7_residual_maps")

# P8 scatter
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, (state, ev, met) in zip(ax, (("ascending", evA, metA), ("descending", evD, metD))):
    v = ev["valid"]
    a.hist2d(db(ev["sim_reg"])[v] + ev["bias"], db(ev["real_ml"])[v], bins=120, range=[[-25, 5], [-25, 5]],
             cmap="magma_r", cmin=1)
    a.plot([-25, 5], [-25, 5], "c--")
    a.set(xlabel="simulated σ⁰ [dB]", ylabel="real σ⁰ [dB]",
          title=f"{state}: r = {met['pearson_r']:.2f}, ρ = {met['spearman_rho']:.2f}, RMSE = {met['rmse_db_bias_removed']:.1f} dB")
fig.suptitle("Pixel-by-pixel agreement (5×5 multilook, fitted cosⁿ model)", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "p8_scatter")

# ============================================================================ SAVE
with open(RES / "metrics.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=float)
for name, rows in (("e1_models", model_rows), ("e4_sensitivity", sens), ("e5_landcover", lc_rows)):
    with open(RES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
log(f"done -> {RES} , {FIG}")
