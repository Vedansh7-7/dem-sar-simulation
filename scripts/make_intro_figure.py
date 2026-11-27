"""High-quality schematic of SAR geometric distortions (foreshortening, layover, shadow)
for the introduction of the report and the A3 poster. Vector PDF + 300 dpi PNG.

Usage: uv run python scripts/make_intro_figure.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

from sarsim.rdsim import shadow_layover  # noqa: E402

OUT = ROOT / "figures" / "intro"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

# ---------------------------------------------------------------- terrain profile
H, look_deg = 693e3, 35.0
look = np.radians(look_deg)
y = np.linspace(-3400, 2800, 12401)                          # ground range [m]
hill = 320 * np.exp(-((y + 2550) / 720) ** 2)                # gentle hill: max slope ~20 deg
peak, fore, back = 1450.0, 58.0, 66.0
mount = np.where(y < 0, peak - np.abs(y) * np.tan(np.radians(fore)), peak - y * np.tan(np.radians(back)))
z = np.maximum(np.maximum(mount, 0), hill)
z = np.convolve(np.pad(z, 25, mode="edge"), np.ones(51) / 51, mode="valid")
yc = H * np.tan(look)
Y = yc + y
alpha = np.arctan2(Y, H - z)
R = np.hypot(Y, H - z)
sh, lo = shadow_layover(np.zeros_like(y), Y, alpha, R, tol_alpha=1e-7, tol_R=0.05)
slope = np.degrees(np.arctan(np.gradient(z, y)))
cls = np.zeros(len(y), int)                                  # 0 undistorted
cls[(slope > 4) & ~lo] = 1                                   # foreshortening (slope facing the radar)
cls[lo] = 2                                                  # layover
cls[sh] = 3                                                  # shadow
COL = {0: "#8C8C8C", 1: "#2E7D32", 2: "#C62828", 3: "#1F3A93"}
NAME = {0: "Undistorted", 1: "Foreshortening", 2: "Layover", 3: "Radar shadow"}

fig = plt.figure(figsize=(8.6, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1], hspace=0.42)
ax = fig.add_subplot(gs[0])
yk, zk = y / 1e3, z / 1e3
ax.fill_between(yk, -0.15, zk, color="#EFE6D2", zorder=1)
for k in range(4):                                           # coloured surface line
    m = (cls == k).astype(float)
    m[m == 0] = np.nan
    ax.plot(yk, zk * m, color=COL[k], lw=3.2, solid_capstyle="butt", zorder=4)

# parallel radar rays (the sensor is 485 km to the left)
d = np.array([np.sin(look), -np.cos(look)])                  # propagation direction in (y, z)
for y0 in np.arange(-3300, 2700, 420):
    z0 = np.interp(y0, y, z)
    if sh[np.argmin(np.abs(y - y0))]:
        continue
    start = np.array([y0, z0]) - 1250 * d
    ax.add_patch(FancyArrowPatch(start / 1e3, np.array([y0, z0]) / 1e3, arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color="#E08A1E", zorder=3, shrinkA=0, shrinkB=1.5))
ip = np.argmax(z)                                            # ray grazing the summit
p0 = np.array([y[ip], z[ip]])
ax.plot([(p0 - 1500 * d)[0] / 1e3, (p0 + 2050 * d)[0] / 1e3], [(p0 - 1500 * d)[1] / 1e3, (p0 + 2050 * d)[1] / 1e3],
        ls="--", lw=1.0, color="#1F3A93", zorder=3)
# iso-range (wavefront) through the summit: the summit is closer to the radar than the foot
w = np.array([np.cos(look), np.sin(look)])
ax.plot([(p0 - 900 * w)[0] / 1e3, (p0 + 300 * w)[0] / 1e3], [(p0 - 900 * w)[1] / 1e3, (p0 + 300 * w)[1] / 1e3],
        ls=":", lw=1.1, color="#C62828", zorder=3)
# labels
ax.annotate("Foreshortening", (-2.95, 0.20), (-3.35, 1.05), color=COL[1], fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COL[1], lw=0.9))
ax.annotate("Layover", (-0.45, 0.80), (-1.75, 1.55), color=COL[2], fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COL[2], lw=0.9))
ax.annotate("Radar shadow", (0.75, 0.05), (1.15, 1.05), color=COL[3], fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COL[3], lw=0.9))
ax.text(-1.32, 0.98, "iso-range line\nthrough the summit", color=COL[2], fontsize=7.5, ha="right", va="center")
ax.text(1.62, 0.55, "ray grazing\nthe summit", color=COL[3], fontsize=7.5, ha="left", va="center")
ax.text(-3.35, 1.78, f"Radar illumination from the left, look angle {look_deg:.0f}°", color="#B86A0C",
        fontsize=9, fontstyle="italic")
ax.set(xlim=(-3.45, 2.85), ylim=(-0.15, 1.95), xlabel="Ground range [km]", ylabel="Height [km]")
ax.set_title("(a) Terrain profile and radar rays", loc="left", fontsize=10.5, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_aspect("equal")

# ---------------------------------------------------------------- what the radar records
ax2 = fig.add_subplot(gs[1])
rr = (R - R.min()) / 1e3
bins = np.arange(-0.02, rr.max() + 0.04, 0.012)
dA = np.hypot(np.gradient(y), np.gradient(z))                # surface length of each sample
bottom = np.zeros(len(bins) - 1)
for k in (0, 1, 2):
    h, _ = np.histogram(rr[(cls == k) & ~sh], bins=bins, weights=dA[(cls == k) & ~sh])
    ax2.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge", color=COL[k], lw=0)
    bottom += h
lit = np.histogram(rr[~sh], bins=bins, weights=dA[~sh])[0]
ref = np.median(lit[lit > 0])
ax2.set_ylim(0, ref * 4.2)
g1 = rr[sh].max()                                            # empty interval: no lit facet echoes there
g0 = rr[~sh & (rr < g1)].max()
ax2.axvspan(g0, g1, color=COL[3], alpha=0.12, lw=0)
ax2.text((g0 + g1) / 2, ref * 2.2, "no echo\n(shadow)", ha="center", color=COL[3], fontsize=8.5)
lay = rr[lo]
ax2.annotate("folded slope adds to\nthe valley in front", (lay.mean(), ref * 2.45), (0.70, ref * 2.85),
             fontsize=8.5, color=COL[2], ha="right", va="center",
             arrowprops=dict(arrowstyle="-", color=COL[2], lw=0.8))
ax2.set(xlabel="Slant range relative to the nearest point [km]", ylabel="Echo energy\n(relative)", yticks=[])
ax2.set_title("(b) What the radar records along this range line", loc="left", fontsize=10.5, fontweight="bold")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in range(4)], ncol=4, frameon=False,
           loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=9)
fig.savefig(OUT / "sar_distortions.pdf", bbox_inches="tight")
fig.savefig(OUT / "sar_distortions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- poster variant: one panel, large type
fig, ax = plt.subplots(figsize=(4.3, 2.75))
ax.fill_between(yk, -0.15, zk, color="#EFE6D2", zorder=1)
for k in range(4):
    m = (cls == k).astype(float)
    m[m == 0] = np.nan
    ax.plot(yk, zk * m, color=COL[k], lw=3.6, solid_capstyle="butt", zorder=4)
for y0 in np.arange(-3300, 2700, 600):
    z0 = np.interp(y0, y, z)
    if sh[np.argmin(np.abs(y - y0))]:
        continue
    start = np.array([y0, z0]) - 1150 * d
    ax.add_patch(FancyArrowPatch(start / 1e3, np.array([y0, z0]) / 1e3, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#E08A1E", zorder=3, shrinkA=0, shrinkB=1.5))
ax.plot([(p0 - 1300 * d)[0] / 1e3, (p0 + 2050 * d)[0] / 1e3], [(p0 - 1300 * d)[1] / 1e3, (p0 + 2050 * d)[1] / 1e3],
        ls="--", lw=1.2, color=COL[3], zorder=3)
ax.text(-3.35, 1.05, "Foreshortening", color=COL[1], fontsize=11, fontweight="bold")
ax.text(-2.05, 1.30, "Layover", color=COL[2], fontsize=11, fontweight="bold")
ax.text(1.05, 1.05, "Shadow", color=COL[3], fontsize=11, fontweight="bold")
ax.text(-3.35, 1.80, f"radar from the left, {look_deg:.0f}°", color="#B86A0C", fontsize=9.5,
        fontstyle="italic")
ax.set(xlim=(-3.45, 2.85), ylim=(-0.15, 1.95))
ax.set_xlabel("Ground range [km]", fontsize=10)
ax.set_ylabel("Height [km]", fontsize=10)
ax.tick_params(labelsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "sar_distortions_poster.pdf", bbox_inches="tight")
fig.savefig(OUT / "sar_distortions_poster.png", dpi=300, bbox_inches="tight")
print("written", OUT / "sar_distortions.pdf", OUT / "sar_distortions_poster.pdf")
