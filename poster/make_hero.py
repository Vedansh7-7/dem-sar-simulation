"""Hero image for the poster: real | simulated Sentinel-1 (ascending, fitted model), no axes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes  # noqa: E402

import sarsim  # noqa: E402
from sarsim import compare, data, rdsim  # noqa: E402

R = json.loads((ROOT / "results" / "metrics.json").read_text())
fm = R["fitted_model"]
dem = data.load_dem(data.get_dem(sarsim.AOI_BBOX, sarsim.AOI_NAME))
item = data.find_s1(sarsim.AOI_BBOX, sarsim.S1_DATES, orbit_state="ascending")
prod, s0, meta = data.load_s1(data.get_s1(item, dem))
sim = rdsim.simulate_grd(prod, dem, meta["geoid_N"], meta["row0"], meta["col0"], s0.shape,
                         model="cos", n=fm["n"], k=fm["k"])
real_ml, sim_ml = compare.multilook(s0, 5), compare.multilook(sim["sigma0"], 5)
foot = binary_erosion(binary_fill_holes(binary_closing(sim["footprint"], iterations=3)), iterations=8)
shift, box = compare.coregister(compare.db(real_ml), compare.db(sim_ml), foot)
sim_reg = compare.apply_shift(sim_ml, shift)
bias = np.median(compare.db(real_ml)[box] - compare.db(sim_reg)[box])
box = compare.interior_box(foot, 0.999)
a = compare.db(real_ml)[box][::-1]          # ascending: flip so north is up
b = (compare.db(sim_reg) + bias)[box][::-1]
fig, ax = plt.subplots(1, 2, figsize=(14, 14 * a.shape[0] / (2 * a.shape[1])), gridspec_kw={"wspace": 0.015})
for x, img, lab in ((ax[0], a, "REAL"), (ax[1], b, "SIMULATED")):
    x.imshow(img, cmap="gray", vmin=-22, vmax=3, interpolation="lanczos")
    x.set_axis_off()
    x.text(0.02, 0.97, lab, transform=x.transAxes, color="white", fontsize=22, fontweight="bold",
           va="top", family="sans-serif", bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4))
fig.savefig(ROOT / "figures" / "pipeline" / "p0_hero.png", dpi=160, bbox_inches="tight", pad_inches=0, facecolor="black")
print("hero written", a.shape)
