"""Smoke test of the real-data chain: DEM + S1 download, simulation, offset/correlation."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.registration import phase_cross_correlation

import sarsim
from sarsim import data, rdsim

t = time.time()
dem_path = data.get_dem(sarsim.AOI_BBOX, sarsim.AOI_NAME)
dem = data.load_dem(dem_path)
print("DEM", dem["h_orth"].shape, dem["h_orth"].min(), dem["h_orth"].max(), f"{time.time()-t:.1f}s")
item = data.find_s1(sarsim.AOI_BBOX, sarsim.S1_DATES, orbit_state="ascending")
print("S1", item.id)
sd = data.get_s1(item, dem)
prod, s0, meta = data.load_s1(sd)
print("window", s0.shape, meta, "orbit vel rms", prod.orbit.vel_rms, f"{time.time()-t:.1f}s")
sim = rdsim.simulate_grd(prod, dem, meta["geoid_N"], meta["row0"], meta["col0"], s0.shape)
print("sim done", f"{time.time()-t:.1f}s", "shadow %", sim["map"]["shadow"].mean()*100, "layover %", sim["map"]["layover"].mean()*100)

db = lambda x: 10*np.log10(np.maximum(x, 1e-5))
real = db(uniform_filter(s0, 5)); simi = db(uniform_filter(sim["sigma0"], 5))
valid = sim["area_norm"] > 0
shift, err, _ = phase_cross_correlation(real*valid, simi*valid, upsample_factor=10)
print("offset (rows, cols) real vs sim:", shift)
m = valid & np.isfinite(real)
print("corr dB:", np.corrcoef(real[m], simi[m])[0, 1])
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(real, cmap="gray", vmin=-20, vmax=5); ax[0].set_title("real S1 sigma0 VV dB")
ax[1].imshow(simi, cmap="gray", vmin=-20, vmax=5); ax[1].set_title("simulated")
plt.savefig(r"C:\Users\HP\.claude\jobs\774a883c\tmp\quick.png", dpi=70)
