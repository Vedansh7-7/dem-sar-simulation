# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DEM-based SAR simulation — a step-by-step walkthrough
#
# This notebook builds a Synthetic Aperture Radar (SAR) simulator from the ground up and
# ends with a simulation driven by a real DEM that is compared, side by side, with a real
# Sentinel-1 image.
#
# | § | Step | What is new |
# |---|------|-------------|
# | 1 | **A single point, a single pulse** | chirp, echo, range compression → range resolution |
# | 2 | **A line of points** | overlapping echoes, resolving neighbours |
# | 3 | **A moving satellite** | range history, Doppler, raw data → range-compressed → focused SAR image |
# | 4 | **A flat surface** | distributed scatterers → speckle, multilooking |
# | 5 | **A sine-wave surface** | topography → foreshortening; fast image-domain simulator |
# | 6 | **A real DEM** | raw-signal vs image-domain simulation of Copernicus GLO-30 terrain |
# | 7 | **Geometric complications** | foreshortening, layover, shadow, look-angle dependence |
# | 8 | **Radiometry** | local incidence angle, backscatter models, area normalisation, speckle |
# | 9 | **Real comparison** | real orbit + DEM → simulated Sentinel-1 GRD vs actual Sentinel-1 GRD |
#
# Every step is shown as a **sequence of sub-plots** that follows the processing chain.
# Sections 1–8 use a Sentinel-1-like C-band sensor flying a straight line over a flat Earth
# (so that the physics stays transparent); section 9 uses the real Sentinel-1 orbit, the
# WGS84 ellipsoid and ESA's product annotation.

# %%
import sys
import time
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.grid": False, "image.cmap": "gray",
})
FIG = ROOT / "figures" / "notebook"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")


def db(x, floor=1e-6):
    return 10 * np.log10(np.maximum(np.asarray(x, float), floor))


rng = np.random.default_rng(42)
T_START = time.time()

# %% [markdown]
# ## 0. The sensor
#
# A C-band sensor with Sentinel-1-like parameters. In sections 1–8 the satellite flies along
# the $x$ axis at height $H$ with speed $v$; the ground range axis $y$ is measured from the
# ground track and the radar looks to the right at a look angle of 35°.

# %%
c = 299_792_458.0          # speed of light [m/s]
f0 = 5.405e9               # carrier (Sentinel-1) [Hz]
lam = c / f0               # wavelength [m]
H = 693e3                  # orbit height [m]
v = 7600.0                 # platform speed [m/s]
look = np.deg2rad(35.0)    # look angle at scene centre
y_c = H * np.tan(look)     # ground range of scene centre [m]
R_c = H / np.cos(look)     # slant range of scene centre [m]

B = 50e6                   # chirp bandwidth [Hz]
Tp = 10e-6                 # pulse length [s]
Kr = B / Tp                # chirp rate [Hz/s]
fs = 60e6                  # range sampling rate [Hz]
L_ant = 12.3               # antenna length (azimuth) [m]
W_ant = 0.821              # antenna height (elevation) [m]
PRF = 1700.0               # pulse repetition frequency [Hz]

dr = c / (2 * fs)                       # slant-range sample spacing
rho_r = c / (2 * B)                     # slant-range resolution (nominal)
rho_g = rho_r / np.sin(look)            # ground-range resolution
L_sa = lam * R_c / L_ant                # synthetic aperture length (first-null beam width)
rho_az = L_ant / 2                      # azimuth resolution (nominal)
APERTURE = 0.6                          # we process +-0.6 L_sa (Doppler band < PRF)

rows = [("wavelength", f"{lam*100:.2f} cm"), ("orbit height", f"{H/1e3:.0f} km"),
        ("look angle", "35°"), ("slant range (centre)", f"{R_c/1e3:.1f} km"),
        ("chirp bandwidth / length", f"{B/1e6:.0f} MHz / {Tp*1e6:.0f} µs"),
        ("range sampling", f"{fs/1e6:.0f} MHz  ({dr:.2f} m)"),
        ("PRF / pulse spacing", f"{PRF:.0f} Hz / {v/PRF:.2f} m"),
        ("antenna", f"{L_ant} m x {W_ant} m"),
        ("synthetic aperture", f"{L_sa/1e3:.2f} km"),
        ("slant / ground range resolution", f"{rho_r:.2f} m / {rho_g:.2f} m"),
        ("azimuth resolution (L/2)", f"{rho_az:.2f} m")]
for k_, v_ in rows:
    print(f"{k_:>34s} : {v_}")

# %% [markdown]
# ## 1. A single point, a single pulse
#
# The radar transmits a **linear-FM chirp** $s(t)=\exp(j\pi K_r t^2)$ of length $T_p$ that
# sweeps the bandwidth $B$. A point at slant range $R$ returns a copy delayed by
# $\tau=2R/c$ and phase-shifted by $-4\pi R/\lambda$. **Range compression** (matched filtering,
# i.e. correlation with the transmitted chirp) squeezes the 10 µs echo into a sinc-shaped
# peak whose width is the range resolution $\rho_r\approx c/2B$.

# %%
def chirp(t):
    """Baseband LFM pulse; instantaneous frequency sweeps -B/2 -> +B/2 over [0, Tp)."""
    t = np.asarray(t, float)
    return np.where((t >= 0) & (t < Tp), np.exp(1j * np.pi * Kr * (t - Tp / 2) ** 2), 0)


ref = chirp(np.arange(int(round(Tp * fs))) / fs)       # replica of the transmitted pulse


def range_compress(sig):
    """Matched filter along the last axis (correlation with the chirp replica)."""
    n = sig.shape[-1]
    return np.fft.ifft(np.fft.fft(sig, axis=-1) * np.conj(np.fft.fft(ref, n)), axis=-1)


def upsample1d(x, k=16):
    n = len(x)
    F = np.fft.fft(x)
    return np.fft.ifft(np.concatenate([F[: n // 2], np.zeros((k - 1) * n, complex), F[n // 2:]])) * k


def width_3db(axis, amp):
    """-3 dB width of |amp|^2 around its maximum (linear interpolation)."""
    p = np.abs(amp) ** 2
    p = p / p.max()
    i = int(np.argmax(p))
    lft = i
    while lft > 0 and p[lft] > 0.5:
        lft -= 1
    rgt = i
    while rgt < len(p) - 1 and p[rgt] > 0.5:
        rgt += 1
    xl = np.interp(0.5, [p[lft], p[lft + 1]], [axis[lft], axis[lft + 1]])
    xr = np.interp(0.5, [p[rgt], p[rgt - 1]], [axis[rgt], axis[rgt - 1]])
    return xr - xl


# one target exactly at the scene centre
tau0 = 2 * (R_c - 150) / c                     # start of the receive window
nr = 1024
tau = tau0 + np.arange(nr) / fs                # fast time
srange = c * tau / 2 - R_c                     # slant range relative to the target [m]
echo = chirp(tau - 2 * R_c / c) * np.exp(-4j * np.pi * R_c / lam)
rc = range_compress(echo)
k_up = 16
rc_up = upsample1d(rc, k_up)
srange_up = srange[0] + np.arange(nr * k_up) * dr / k_up
res_meas = width_3db(srange_up, rc_up)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
# (a) geometry
a = ax[0, 0]
a.plot([-50, 700], [0, 0], color="saddlebrown", lw=3)
a.plot(0, H / 1e3, marker="^", ms=12, color="tab:blue")
bw = lam / W_ant / 2
for s_, ls in ((-bw, ":"), (bw, ":"), (0, "-")):
    ang = look + s_
    a.plot([0, H * np.tan(ang) / 1e3], [H / 1e3, 0], ls, color="tab:orange")
a.plot(y_c / 1e3, 0, "r*", ms=14)
a.annotate("satellite", (0, H / 1e3), (40, H / 1e3 + 20))
a.annotate(f"R = {R_c/1e3:.0f} km", (y_c / 2e3, H / 2e3), (y_c / 2e3 + 30, H / 2e3 + 30))
a.annotate("point target", (y_c / 1e3, 0), (y_c / 1e3 - 160, 60), arrowprops=dict(arrowstyle="->"))
a.text(10, H / 1e3 - 120, "look\n35°")
a.set(xlim=(-50, 700), ylim=(-30, 760), xlabel="ground range y [km]", ylabel="height [km]",
      title="(a) Geometry (side view)")
a.set_aspect("equal")
# (b) transmitted chirp
t_us = np.arange(len(ref)) / fs * 1e6
a = ax[0, 1]
a.plot(t_us, ref.real, lw=0.6)
a.set(xlabel="time [µs]", ylabel="Re s(t)", title="(b) Transmitted chirp (real part)")
ins = a.inset_axes([0.62, 0.62, 0.36, 0.34])
ins.plot(t_us[:120], ref.real[:120], lw=0.8)
ins.set_xticks([]); ins.set_yticks([])
# (c) instantaneous frequency + spectrum
a = ax[0, 2]
a.plot(t_us, Kr * (t_us * 1e-6 - Tp / 2) / 1e6, color="tab:green")
a.set(xlabel="time [µs]", ylabel="instantaneous frequency [MHz]", title="(c) Frequency sweep & spectrum")
a2 = a.twinx()
F = np.fft.fftshift(np.fft.fft(ref, 4096))
fax = np.fft.fftshift(np.fft.fftfreq(4096, 1 / fs)) / 1e6
a2.plot(np.full_like(fax, np.nan), fax)
ins = a.inset_axes([0.55, 0.08, 0.42, 0.35])
ins.plot(fax, db(np.abs(F) ** 2 / np.abs(F).max() ** 2), lw=0.7, color="k")
ins.set(ylim=(-40, 3)); ins.set_title("spectrum [dB] vs MHz", fontsize=7); ins.tick_params(labelsize=6)
a2.set_yticks([])
# (d) received echo
a = ax[1, 0]
a.plot(srange, echo.real, lw=0.5)
a.set(xlabel="slant range − R [m]", ylabel="Re echo", title="(d) Received echo (one pulse)")
# (e) range compressed
a = ax[1, 1]
a.plot(srange, db(np.abs(rc) ** 2 / np.abs(rc).max() ** 2), color="k")
a.set(xlabel="slant range − R [m]", ylabel="power [dB]", ylim=(-50, 3), xlim=(-150, 600),
      title="(e) After range compression")
# (f) zoom: resolution
a = ax[1, 2]
p = np.abs(rc_up) ** 2 / np.abs(rc_up).max() ** 2
a.plot(srange_up, p, color="k")
a.axhline(0.5, color="r", ls="--", lw=0.8)
a.axvspan(-res_meas / 2, res_meas / 2, color="r", alpha=0.15)
a.set(xlim=(-12, 12), xlabel="slant range − R [m]", ylabel="power (linear)",
      title=f"(f) Zoom: −3 dB width = {res_meas:.2f} m  (0.886·c/2B = {0.886*rho_r:.2f} m)")
fig.suptitle("1 · A single point target and a single pulse", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "01_point_single_pulse")
plt.show()

# %% [markdown]
# The 10 µs echo is ~1.5 km long in slant range, yet after matched filtering the target is
# a ~2.7 m wide peak: this is **pulse compression**. The resolution depends only on the
# bandwidth $B$, not on the pulse length.
#
# ## 2. A line of points
#
# Now five targets lie on a line in ground range. Their echoes **overlap** in the raw signal
# (each is 1.5 km long), but after range compression each target becomes its own peak at
# its own slant range. Two targets can only be separated if they are further apart than the
# resolution.

# %%
ys_line = np.array([-90.0, -45.0, 0.0, 30.0, 70.0])      # ground-range offsets [m]
amp_line = np.array([1.0, 0.7, 1.0, 0.5, 0.8])
R_line = np.hypot(H, y_c + ys_line)
echo_l = sum(a_ * chirp(tau - 2 * r_ / c) * np.exp(-4j * np.pi * r_ / lam) for a_, r_ in zip(amp_line, R_line))
rc_l = range_compress(echo_l)

fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
a = ax[0]
yy = np.linspace(-130, 110, 2)
a.plot(yy, [0, 0], color="saddlebrown", lw=3)
for y_, a_ in zip(ys_line, amp_line):
    a.plot(y_, 0, "r*", ms=8 + 8 * a_)
    # iso-range wavefront through the target (perpendicular to the look direction)
    d = np.array([np.cos(look), np.sin(look)])  # direction along wavefront (y, z)
    a.plot([y_ - 40 * d[0], y_ + 40 * d[0]], [0 + 40 * d[1], 0 - 40 * d[1]], color="tab:blue", lw=0.6, alpha=0.6)
    a.annotate("", (y_, 0), (y_ - 60 * np.sin(look), 60 * np.cos(look)),
               arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.8))
a.set(xlim=(-140, 120), ylim=(-40, 70), xlabel="ground range offset [m]", ylabel="height [m]",
      title="(a) Targets; rays (orange) & iso-range lines")
a.set_aspect("equal")
ax[1].plot(srange, echo_l.real, lw=0.4)
ax[1].set(xlabel="slant range − R_c [m]", title="(b) Raw echo: chirps overlap")
ax[2].plot(srange, np.abs(echo_l), lw=0.6, color="tab:purple")
ax[2].set(xlabel="slant range − R_c [m]", title="(c) |echo| — targets indistinguishable")
ax[3].plot(srange, np.abs(rc_l) / np.abs(rc_l).max(), color="k")
for r_ in R_line - R_c:
    ax[3].axvline(r_, color="r", ls=":", lw=0.8)
ax[3].set(xlim=(-80, 60), xlabel="slant range − R_c [m]", title="(d) Range compressed: 5 peaks")
fig.suptitle("2 · A line of points: one pulse", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "02_line_of_points")
plt.show()

# resolution test: pairs of equal targets at decreasing separation
fig, ax = plt.subplots(1, 4, figsize=(17, 3.4), sharey=True)
for a, s_ in zip(ax, [2.0, 1.2, 1.0, 0.6]):
    sep = s_ * rho_g
    Rp = np.hypot(H, y_c + np.array([-sep / 2, sep / 2]))
    e = sum(chirp(tau - 2 * r_ / c) * np.exp(-4j * np.pi * r_ / lam) for r_ in Rp)
    r_up = upsample1d(range_compress(e), k_up)
    a.plot(srange_up, np.abs(r_up) / np.abs(r_up).max(), color="k")
    for r_ in Rp - R_c:
        a.axvline(r_, color="r", ls=":", lw=0.8)
    a.set(xlim=(-15, 15), xlabel="slant range − R_c [m]",
          title=f"separation {s_:.1f}·ρg = {sep:.1f} m (ground)")
ax[0].set_ylabel("|range compressed|")
fig.suptitle("2b · Range resolution: when do two points merge?", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "02b_range_resolution")
plt.show()

# %% [markdown]
# ## 3. A moving satellite — forming the synthetic aperture
#
# A single pulse cannot separate targets in **azimuth** (along track): the real antenna beam is
# kilometres wide. As the satellite flies past, every target stays in the beam for
# ~$L_{sa}$ = 3.8 km of flight, i.e. ~900 pulses. During that time its range follows a
# hyperbola $R(\eta)=\sqrt{R_0^2+(v\eta)^2}$, which produces a **Doppler** chirp in azimuth.
# Correlating with that azimuth chirp (here done exactly by **time-domain back-projection**)
# gives the azimuth resolution $\approx L/2$, independent of range.
#
# We first write two building blocks that are reused in all later sections:
#
# * `simulate_raw` — the raw echo of many point scatterers for every pulse. Each echo is placed
#   as a band-limited impulse at its exact (fractional) delay with its two-way phase
#   $e^{-j4\pi R/\lambda}$ and two-way antenna gain, and then convolved with the chirp.
# * `backproject` — focuses range-compressed data onto any 3-D grid of image points by summing,
#   pulse by pulse, the range-compressed sample at the exact range of each image point.

# %%
TAPS = 8
_k = np.arange(-TAPS, TAPS + 1)


def make_eta(x_min, x_max, aperture=APERTURE):
    """Slow-time axis covering a scene [x_min, x_max] with +-aperture*L_sa of flight."""
    x0, x1 = x_min - aperture * L_sa, x_max + aperture * L_sa
    return np.arange(x0 / v, x1 / v, 1 / PRF)


def antenna_gain(dx, R):
    """Two-way azimuth antenna power pattern (uniform aperture of length L_ant)."""
    return np.sinc(L_ant * (dx / R) / lam) ** 2


def simulate_raw(xs, ys, zs, amps, eta, chunk=24, margin=30.0):
    """Raw SAR echoes (n_pulses x n_range) of point scatterers.

    xs: along-track [m], ys: ground range from the track [m], zs: height [m],
    amps: complex reflectivity. The sensor is at (v*eta, 0, H).
    Returns raw, tau0 (fast-time of sample 0).
    """
    xs, ys, zs, amps = (np.ravel(np.asarray(q)) for q in (xs, ys, zs, amps))
    R_min = np.sqrt(ys**2 + (H - zs) ** 2).min() - margin
    dx_max = np.abs(xs).max() + np.abs(v * eta).max()          # worst-case along-track offset
    R_max = np.sqrt(dx_max**2 + (ys**2 + (H - zs) ** 2).max())
    tau0_ = 2 * R_min / c
    n = int(2 ** np.ceil(np.log2((2 * (R_max - R_min + margin) / c + Tp) * fs)))
    imp = np.zeros((len(eta), n), complex)
    win = lambda q: 0.5 + 0.5 * np.cos(np.pi * q / (TAPS + 1))  # noqa: E731
    for p0 in range(0, len(eta), chunk):
        pe = eta[p0: p0 + chunk]
        dx = xs[None, :] - v * pe[:, None]
        R = np.sqrt(dx**2 + ys[None, :] ** 2 + (H - zs[None, :]) ** 2)
        ph = amps[None, :] * antenna_gain(dx, R) * np.exp(-4j * np.pi * R / lam)
        d = (2 * R / c - tau0_) * fs
        di = np.floor(d).astype(np.int64)
        fr = d - di
        q = _k[None, None, :] - fr[..., None]
        w = np.sinc(q) * win(q)
        idx = di[..., None] + _k
        flat = (np.arange(len(pe))[:, None, None] * n + idx)
        ok = (idx >= 0) & (idx < n)
        vals = (ph[..., None] * w)[ok]
        f_ = flat[ok]
        blk = np.bincount(f_, vals.real, len(pe) * n) + 1j * np.bincount(f_, vals.imag, len(pe) * n)
        imp[p0: p0 + len(pe)] = blk.reshape(len(pe), n)
    raw = np.fft.ifft(np.fft.fft(imp, axis=1) * np.fft.fft(ref, n)[None, :], axis=1)
    return raw, tau0_


def backproject(rcd, eta, tau0_, gx, gy, gz=0.0, up=8):
    """Time-domain back-projection of range-compressed data onto image points (gx, gy, gz)."""
    n = rcd.shape[1]
    F = np.fft.fft(rcd, axis=1)
    Fz = np.concatenate([F[:, : n // 2], np.zeros((len(eta), (up - 1) * n), complex), F[:, n // 2:]], axis=1)
    rcu = (np.fft.ifft(Fz, axis=1) * up).astype(np.complex64)
    gx, gy = np.broadcast_arrays(np.asarray(gx, float), np.asarray(gy, float))
    gz = np.broadcast_to(np.asarray(gz, float), gx.shape)
    img = np.zeros(gx.shape, complex)
    ryz2 = gy**2 + (H - gz) ** 2
    nu = rcu.shape[1]
    for p, e in enumerate(eta):
        R = np.sqrt((gx - v * e) ** 2 + ryz2)
        d = (2 * R / c - tau0_) * fs * up
        i0 = np.floor(d).astype(np.int64)
        f_ = d - i0
        ok = (i0 >= 0) & (i0 < nu - 1)
        i0c = np.clip(i0, 0, nu - 2)
        val = (1 - f_) * rcu[p, i0c] + f_ * rcu[p, i0c + 1]
        img += np.where(ok, val * np.exp(4j * np.pi * R / lam), 0)
    return img


# three point targets (x along-track, y ground-range offset)
tx = np.array([0.0, 40.0, -30.0])
ty = np.array([0.0, 25.0, -35.0])
ta = np.array([1.0, 0.8, 0.6])
eta = make_eta(tx.min(), tx.max())
t0 = time.time()
raw3, tau0_3 = simulate_raw(tx, y_c + ty, np.zeros(3), ta, eta)
rc3 = range_compress(raw3)
print(f"raw data: {raw3.shape[0]} pulses x {raw3.shape[1]} range samples  ({time.time()-t0:.1f} s)")

# %%
# ---- 3a: the satellite flies past: snapshots of the beam footprint ----
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
fw = 0.886 * L_sa  # 3 dB two-way footprint width in azimuth
for a, e in zip(ax, [-0.45, -0.2, 0.0, 0.2]):
    xs_ = v * e
    a.add_patch(Rectangle(((xs_ - fw / 2) / 1e3, -3), fw / 1e3, 6, color="tab:orange", alpha=0.25))
    a.plot(xs_ / 1e3, -2.7, marker="^", ms=12, color="tab:blue")
    a.axhline(-2.7, color="tab:blue", lw=0.5, ls="--")
    a.plot(tx / 1e3, ty / 1e3, "r*", ms=9)
    gain = antenna_gain(0 - xs_, R_c)
    a.set(xlim=(-5, 5), ylim=(-3, 3), xlabel="azimuth x [km]",
          title=f"η = {e:+.2f} s,  gain on target = {db(gain):.1f} dB")
ax[0].set_ylabel("ground range offset [km]\n(ground track 485 km below)")
fig.suptitle("3a · The satellite moves: the beam footprint (orange) sweeps over the targets",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "03a_moving_satellite_snapshots")
plt.show()

# ---- 3b: what one target experiences, and the 2-D signal chain ----
R_hist = np.sqrt((v * eta) ** 2 + R_c**2)
f_dop = -2 * v**2 * eta / (lam * R_hist)
g_hist = antenna_gain(-v * eta, R_hist)
srange3 = c * (tau0_3 + np.arange(raw3.shape[1]) / fs) / 2 - R_c
ext = [srange3[0], srange3[-1], eta[-1], eta[0]]

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].plot(eta, R_hist - R_c, color="k")
ax[0, 0].axhline(rho_r, color="r", ls=":", lw=0.8)
ax[0, 0].text(eta[0], rho_r * 1.05, "one range resolution cell", color="r", fontsize=8)
ax[0, 0].set(xlabel="azimuth time η [s]", ylabel="R(η) − R₀ [m]",
             title="(a) Range history: hyperbola (range migration)")
ax[0, 1].plot(eta, f_dop, color="tab:green")
ax[0, 1].axhline(PRF / 2, color="r", ls="--", lw=0.8); ax[0, 1].axhline(-PRF / 2, color="r", ls="--", lw=0.8)
ax[0, 1].text(eta[0], PRF / 2 * 0.9, "±PRF/2", color="r", fontsize=8)
ax[0, 1].set(xlabel="azimuth time η [s]", ylabel="Doppler [Hz]", title="(b) Doppler history: linear FM in azimuth")
ax[0, 2].plot(eta, np.sqrt(g_hist) * np.cos(4 * np.pi * (R_hist - R_c) / lam), lw=0.4)
ax[0, 2].plot(eta, np.sqrt(g_hist), "r", lw=1)
ax[0, 2].set(xlabel="azimuth time η [s]", ylabel="Re", title="(c) Azimuth phase history × antenna pattern")
ax[1, 0].imshow(raw3.real, aspect="auto", extent=ext, cmap="RdBu", vmin=-1.5, vmax=1.5)
ax[1, 0].set(xlabel="slant range − R_c [m]", ylabel="azimuth time η [s]", title="(d) RAW data (real part)")
ax[1, 1].imshow(np.abs(rc3), aspect="auto", extent=ext, cmap="magma")
ax[1, 1].set(xlim=(-60, 60), xlabel="slant range − R_c [m]", ylabel="azimuth time η [s]",
             title="(e) Range compressed: 3 'azimuth chirps'")
# focus
gx3, gy3 = np.meshgrid(np.arange(-70, 70, 0.5), np.arange(-70, 70, 0.5), indexing="ij")
t0 = time.time()
img3 = backproject(rc3, eta, tau0_3, gx3, y_c + gy3)
print(f"back-projection {gx3.size} px x {len(eta)} pulses: {time.time()-t0:.1f} s")
p3 = np.abs(img3) ** 2
ax[1, 2].imshow(db(p3 / p3.max()).T, origin="lower", extent=[-70, 70, -70, 70], vmin=-40, vmax=0, cmap="magma")
ax[1, 2].plot(tx, ty, "c+", ms=14, mew=0.8)
ax[1, 2].set(xlabel="azimuth x [m]", ylabel="ground range offset [m]", title="(f) FOCUSED SAR image [dB]")
fig.suptitle("3b · From raw echoes to a focused image  (raw → range compression → azimuth compression)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "03b_signal_chain")
plt.show()
print(f"range migration over the processed aperture: {R_hist.max()-R_c:.2f} m "
      f"(range resolution cell {rho_r:.1f} m)")

# %%
# ---- 3c: impulse response of the centre target ----
fx = np.arange(-15, 15, 0.05)
gaz = backproject(rc3, eta, tau0_3, fx, np.full_like(fx, y_c))            # azimuth cut
grg = backproject(rc3, eta, tau0_3, np.zeros_like(fx), y_c + fx)          # ground-range cut
gxx, gyy = np.meshgrid(np.arange(-15, 15, 0.25), np.arange(-15, 15, 0.25), indexing="ij")
irf = np.abs(backproject(rc3, eta, tau0_3, gxx, y_c + gyy)) ** 2
res_az = width_3db(fx, gaz)
res_gr = width_3db(fx, grg)


def pslr(axis, amp):
    p = np.abs(amp) ** 2 / np.max(np.abs(amp) ** 2)
    i = np.argmax(p)
    j = i
    while j < len(p) - 1 and p[j + 1] < p[j]:
        j += 1
    return db(p[j:].max())


fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].imshow(db(irf / irf.max()).T, origin="lower", extent=[-15, 15, -15, 15], vmin=-35, vmax=0, cmap="magma")
ax[0].set(xlabel="azimuth [m]", ylabel="ground range [m]", title="(a) Point-target response [dB]")
ax[1].plot(fx, db(np.abs(gaz) ** 2 / np.abs(gaz).max() ** 2), "k")
ax[1].axhline(-3, color="r", ls=":")
ax[1].set(ylim=(-40, 2), xlabel="azimuth [m]", ylabel="dB",
          title=f"(b) Azimuth cut: ρ_az = {res_az:.2f} m (L/2 = {rho_az:.2f}), PSLR {pslr(fx, gaz):.1f} dB")
ax[2].plot(fx, db(np.abs(grg) ** 2 / np.abs(grg).max() ** 2), "k")
ax[2].axhline(-3, color="r", ls=":")
ax[2].set(ylim=(-40, 2), xlabel="ground range [m]", ylabel="dB",
          title=f"(c) Range cut: ρ_g = {res_gr:.2f} m (0.886·ρr/sinθ = {0.886*rho_g:.2f}), PSLR {pslr(fx, grg):.1f} dB")
fig.suptitle("3c · Impulse response function (IRF) — the 'pixel' of the SAR", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "03c_impulse_response")
plt.show()

# separable sinc^2 PSF (intensity) reproducing the measured IRF - used by the image-domain simulator
def psf_kernel(dx_grid, dy_grid, half=20.0):
    xk = np.arange(-half, half + 1e-9, dx_grid)
    yk = np.arange(-half, half + 1e-9, dy_grid)
    kx = np.sinc(0.886 * xk / res_az) ** 2
    ky = np.sinc(0.886 * yk / res_gr) ** 2
    k2 = np.outer(kx, ky)
    return k2 / k2.sum()

# %% [markdown]
# **Observations.** The range history (a) is a very flat hyperbola: for a spaceborne SAR with
# a 4.6 km processed aperture the range migration is only ~3 m, about one range cell — so the
# azimuth chirps in (e) are almost straight lines. Back-projection (f) collapses every chirp to
# a point. The measured resolutions (≈ 5.6 m azimuth, ≈ 4.6 m ground range) match theory.
#
# ## 4. A flat surface — distributed scatterers and speckle
#
# A natural surface is not a few bright points but a dense carpet of small scatterers. Each
# resolution cell then contains many scatterers with random phases; their coherent sum gives
# the granular **speckle** pattern. Its intensity is exponentially distributed (single look),
# and averaging $L$ independent looks (**multilooking**) reduces the variance by $1/L$.

# %%
sp = 2.0
fx_ = np.arange(-60, 60 + 1e-9, sp)
fy_ = np.arange(-80, 80 + 1e-9, sp)
FX, FY = np.meshgrid(fx_, fy_, indexing="ij")
sigma0_flat = 0.05
amp_flat = np.sqrt(sigma0_flat * sp**2 / 2) * (rng.standard_normal(FX.shape) + 1j * rng.standard_normal(FX.shape))
eta4 = make_eta(fx_.min(), fx_.max())
t0 = time.time()
raw4, tau0_4 = simulate_raw(FX, y_c + FY, np.zeros_like(FX), amp_flat, eta4)
rc4 = range_compress(raw4)
g4x, g4y = np.meshgrid(np.arange(-50, 50, 1.0), np.arange(-70, 70, 1.0), indexing="ij")
slc4 = backproject(rc4, eta4, tau0_4, g4x, y_c + g4y)
print(f"{FX.size} scatterers x {len(eta4)} pulses: {time.time()-t0:.1f} s")
I4 = np.abs(slc4) ** 2
I4 /= I4.mean()
ml = 15   # 15 x 15 one-metre pixels ~ 3 x 3 resolution cells
I4_ml = uniform_filter(I4, ml)
enl = I4_ml[ml:-ml, ml:-ml].mean() ** 2 / I4_ml[ml:-ml, ml:-ml].var()
srange4 = c * (tau0_4 + np.arange(raw4.shape[1]) / fs) / 2 - R_c

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
sc = ax[0, 0].scatter(FX, FY, c=np.abs(amp_flat), s=2, cmap="viridis")
ax[0, 0].set(xlabel="azimuth [m]", ylabel="ground range offset [m]", title=f"(a) {FX.size} random scatterers (|a|)")
ax[0, 0].set_aspect("equal")
ax[0, 1].imshow(raw4.real, aspect="auto", cmap="RdBu", extent=[srange4[0], srange4[-1], eta4[-1], eta4[0]])
ax[0, 1].set(xlabel="slant range − R_c [m]", ylabel="η [s]", title="(b) RAW data (real part)")
ax[0, 2].imshow(np.abs(rc4), aspect="auto", cmap="magma", extent=[srange4[0], srange4[-1], eta4[-1], eta4[0]])
ax[0, 2].set(xlim=(-80, 80), xlabel="slant range − R_c [m]", ylabel="η [s]", title="(c) Range compressed")
ext4 = [-70, 70, -50, 50]
ax[1, 0].imshow(db(I4), extent=ext4, origin="lower", vmin=-15, vmax=8)
ax[1, 0].set(xlabel="ground range offset [m]", ylabel="azimuth [m]", title="(d) Focused single-look image [dB]: SPECKLE")
ax[1, 1].imshow(db(I4_ml), extent=ext4, origin="lower", vmin=-15, vmax=8)
ax[1, 1].set(xlabel="ground range offset [m]", ylabel="azimuth [m]", title=f"(e) Multilooked {ml}×{ml} px (ENL ≈ {enl:.1f})")
bins = np.linspace(0, 6, 61)
ax[1, 2].hist(I4.ravel(), bins, density=True, alpha=0.5, label="single look")
ax[1, 2].plot(bins, np.exp(-bins), "b", label="exponential (theory)")
ax[1, 2].hist(I4_ml.ravel(), bins, density=True, alpha=0.5, color="tab:orange", label=f"multilook")
from scipy.stats import gamma as gamma_dist
ax[1, 2].plot(bins, gamma_dist.pdf(bins, enl, scale=1 / enl), color="tab:orange", label=f"Gamma(L={enl:.1f})")
ax[1, 2].set(xlabel="normalised intensity", ylabel="pdf", title="(f) Speckle statistics")
ax[1, 2].legend(fontsize=8)
fig.suptitle("4 · A flat surface: random scatterers → raw → range compressed → focused → multilooked",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "04_flat_surface_speckle")
plt.show()

# %% [markdown]
# ## 5. A sine-wave surface — topography changes the geometry
#
# Now the surface height varies, $z(y)=A\sin(2\pi y/\Lambda)$, while every square metre of
# ground still has the **same** reflectivity. The SAR measures *slant range*, so:
#
# * slopes **facing** the radar are compressed into fewer range cells → brighter (**foreshortening**),
# * slopes **facing away** are stretched → darker,
# * an elevated point is imaged **closer** to the sensor than its true ground position
#   (displacement ≈ $z\cot\theta$ when focused onto a flat reference plane).
#
# This motivates a much faster **image-domain simulator**: instead of generating and focusing
# raw echoes, project every scatterer directly to its zero-Doppler position
# (azimuth $x$, slant range $R$), accumulate its power, and blur with the IRF measured in §3.

# %%
from sarsim.rdsim import splat, shadow_layover

A_s, Lam_s = 12.0, 150.0
sp5 = 2.0
x5 = np.arange(-40, 40 + 1e-9, sp5)
y5 = np.arange(-150, 150 + 1e-9, sp5)
X5, Y5 = np.meshgrid(x5, y5, indexing="ij")
Z5 = A_s * np.sin(2 * np.pi * Y5 / Lam_s)
amp5 = np.sqrt(sigma0_flat * sp5**2 / 2) * (rng.standard_normal(X5.shape) + 1j * rng.standard_normal(X5.shape))


def image_domain(xs, ys, zs, w, gx_axis, gy_axis, psf=True):
    """Fast simulator: zero-Doppler projection onto the flat reference plane + PSF blur.

    xs, ys, zs: scatterer positions; w: backscattered power of each scatterer.
    Returns intensity on the (gx_axis, gy_axis) grid (azimuth x ground range of the reference plane).
    """
    R = np.sqrt(np.ravel(ys) ** 2 + (H - np.ravel(zs)) ** 2)
    y_ref = np.sqrt(R**2 - H**2)                       # where the SAR 'puts' the point on z = 0
    dgx, dgy = gx_axis[1] - gx_axis[0], gy_axis[1] - gy_axis[0]
    img = splat((np.ravel(xs) - gx_axis[0]) / dgx, (y_ref - gy_axis[0]) / dgy, np.ravel(w),
                (len(gx_axis), len(gy_axis)))
    img /= dgx * dgy
    if psf:
        img = fftconvolve(img, psf_kernel(dgx, dgy), mode="same")
    return img


eta5 = make_eta(x5.min(), x5.max())
t0 = time.time()
raw5, tau0_5 = simulate_raw(X5, y_c + Y5, Z5, amp5, eta5)
gx5 = np.arange(-30, 30, 1.0)
gy5 = np.arange(-140, 140, 1.0)
G5x, G5y = np.meshgrid(gx5, gy5, indexing="ij")
slc5 = backproject(range_compress(raw5), eta5, tau0_5, G5x, y_c + G5y)
print(f"sine surface raw simulation + focusing: {time.time()-t0:.1f} s")
I5 = np.abs(slc5) ** 2
I5_id = image_domain(X5, y_c + Y5, Z5, np.abs(amp5) ** 2, gx5, y_c + gy5)

# geometry along one profile
yp = np.linspace(-150, 150, 3001)
zp = A_s * np.sin(2 * np.pi * yp / Lam_s)
slope = np.degrees(np.arctan(np.gradient(zp, yp)))
Rp = np.sqrt((y_c + yp) ** 2 + (H - zp) ** 2)
th_loc = np.degrees(look) - slope        # local incidence angle (2-D profile)
y_refp = np.sqrt(Rp**2 - H**2) - y_c

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]
sc = a.scatter(yp, zp, c=th_loc, s=3, cmap="coolwarm", vmin=0, vmax=70)
for y0 in np.arange(-150, 160, 25):
    a.annotate("", (y0, A_s * np.sin(2 * np.pi * y0 / Lam_s)),
               (y0 - 30 * np.sin(look), A_s * np.sin(2 * np.pi * y0 / Lam_s) + 30 * np.cos(look)),
               arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.7))
plt.colorbar(sc, ax=a, label="local incidence [°]")
a.set(xlabel="ground range offset [m]", ylabel="height [m]", ylim=(-25, 45),
      title="(a) Surface & radar rays (from the left)")
a = ax[0, 1]
a.plot(yp, slope, label="terrain slope toward sensor [°]")
a.plot(yp, th_loc, label="local incidence angle [°]")
a.axhline(np.degrees(look), color="k", ls=":", lw=0.8)
a.set(xlabel="ground range offset [m]", ylabel="degrees", title="(b) Slope and local incidence")
a.legend(fontsize=8)
a = ax[0, 2]
a.plot(yp, Rp - R_c, "k", label="slant range of the surface")
a.plot(yp, np.hypot(y_c + yp, H) - R_c, "k:", lw=0.8, label="flat surface")
for rr in np.arange(-100, 100, 2 * rho_r * 2):
    a.axhline(rr, color="tab:blue", lw=0.3)
a.set(xlabel="ground range offset [m]", ylabel="R − R_c [m]",
      title="(c) Ground → slant range (lines: every 4th range cell)")
a.legend(fontsize=8)
a = ax[1, 0]
dens, edges = np.histogram(y_refp, bins=np.arange(-160, 160, 2.0))
a.bar(edges[:-1], dens / dens.mean(), width=2.0, color="tab:gray")
a.set(xlabel="imaged ground range [m]", ylabel="relative power",
      title="(d) Projected scatterer density = geometric brightness")
ext5 = [gy5[0], gy5[-1], gx5[0], gx5[-1]]
ax[1, 1].imshow(db(uniform_filter(I5 / I5.mean(), 5)), extent=ext5, origin="lower", aspect="auto", vmin=-8, vmax=7)
ax[1, 1].set(xlabel="imaged ground range [m]", ylabel="azimuth [m]", title="(e) RAW-signal simulation, focused (5×5 looks)")
prof_raw = I5.mean(axis=0)
prof_id = I5_id.mean(axis=0)
ax[1, 2].plot(gy5, prof_raw / prof_raw.mean(), lw=0.8, label="raw-signal sim (focused)")
ax[1, 2].plot(gy5, prof_id / prof_id.mean(), "r", lw=1.2, label="image-domain sim")
r5 = np.corrcoef(prof_raw, prof_id)[0, 1]
ax[1, 2].set(xlabel="imaged ground range [m]", ylabel="mean intensity (norm.)",
             title=f"(f) Azimuth-averaged profile: r = {r5:.2f}")
ax[1, 2].legend(fontsize=8)
fig.suptitle("5 · A sine-wave surface (equal reflectivity everywhere): brightness comes from geometry alone",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "05_sine_surface")
plt.show()

# %% [markdown]
# Both simulators agree closely (panel f) — the image-domain simulator reproduces the raw-signal
# result without generating a single echo, and it runs thousands of times faster. That is what
# makes full-scene simulation from a DEM possible.
#
# ## 6. A real DEM
#
# We now replace the analytic surface by the **Copernicus GLO-30 DEM** of the study area
# (Innsbruck, Tyrol: the Inn valley at ~580 m between the Nordkette (2,300 m) and the
# Patscherkofel). The radar flies north–south and looks **east** (like an ascending
# Sentinel-1 pass), so terrain rising towards the east faces the sensor.
#
# First a small 480 m × 480 m patch is simulated **both** ways — full raw signal
# (~15,000 scatterers × ~1,100 pulses) and image-domain — then the image-domain simulator is
# applied to a 9 km × 9 km area.

# %%
from matplotlib.colors import LightSource
from scipy.interpolate import RectBivariateSpline

import sarsim
from sarsim import data as sdata

dem_full = sdata.load_dem(sdata.get_dem(sarsim.AOI_BBOX, sarsim.AOI_NAME))
lat_d, lon_d, h_d = dem_full["lat"], dem_full["lon"], dem_full["h_orth"]
MLAT = 111_132.0
MLON = 111_320.0 * np.cos(np.radians(lat_d.mean()))
dlat_m = abs(lat_d[1] - lat_d[0]) * MLAT
dlon_m = abs(lon_d[1] - lon_d[0]) * MLON
_spl = RectBivariateSpline(lat_d[::-1], lon_d, h_d[::-1], kx=3, ky=3)
ls = LightSource(azdeg=315, altdeg=45)
print(f"DEM {h_d.shape}, pixel {dlat_m:.1f} m (N) x {dlon_m:.1f} m (E), heights {h_d.min():.0f}–{h_d.max():.0f} m")


def dem_patch(lat0, lon0, half_n, half_e, spacing):
    """Bicubic resampling of the DEM onto a local metric grid: x = north (azimuth), y = east (range)."""
    xn = np.arange(-half_n, half_n + 1e-9, spacing)
    ye = np.arange(-half_e, half_e + 1e-9, spacing)
    return xn, ye, _spl(lat0 + xn / MLAT, lon0 + ye / MLON)


def hill(z, spacing):
    """Hillshade for arrays whose row 0 is SOUTH (displayed with origin='lower')."""
    return ls.hillshade(z[::-1], dx=spacing, dy=spacing)[::-1]


def toy_masks(X, Y, Z):
    """Shadow / layover for the flat-Earth toy geometry (sensor at y = 0, height H)."""
    alpha = np.arctan2(Y, H - Z)               # off-nadir look angle
    R = np.sqrt(Y**2 + (H - Z) ** 2)
    line = np.broadcast_to(np.arange(X.shape[0])[:, None], X.shape)
    sh, lo = shadow_layover(line.ravel().astype(float), Y.ravel(), alpha.ravel(), R.ravel(),
                            tol_alpha=1e-6, tol_R=0.5)
    return sh.reshape(X.shape), lo.reshape(X.shape)


# choose the patch with the steepest east-west terrain
gE = np.gradient(h_d, axis=1) / dlon_m
score = uniform_filter(np.abs(gE), 17)
m_ = 40
i0, j0 = np.unravel_index(np.argmax(score[m_:-m_, m_:-m_]), score[m_:-m_, m_:-m_].shape)
i0, j0 = i0 + m_, j0 + m_
lat0, lon0 = lat_d[i0], lon_d[j0]
xn6, ye6, Z6 = dem_patch(lat0, lon0, 240, 240, 4.0)
Z6 = Z6 - Z6.mean()                              # reference plane = mean patch height
X6, Y6 = np.meshgrid(xn6, y_c + ye6, indexing="ij")
sh6, lo6 = toy_masks(X6, Y6, Z6)
amp6 = np.sqrt(sigma0_flat * 4.0**2 / 2) * (rng.standard_normal(X6.shape) + 1j * rng.standard_normal(X6.shape))
amp6[sh6] = 0                                    # shadowed scatterers receive no energy
print(f"patch at {lat0:.4f}N {lon0:.4f}E: relief {np.ptp(Z6):.0f} m, {X6.size} scatterers, "
      f"shadow {sh6.mean()*100:.1f} %, layover {lo6.mean()*100:.1f} %")

eta6 = make_eta(xn6.min(), xn6.max())
t0 = time.time()
raw6, tau0_6 = simulate_raw(X6, Y6, Z6, amp6, eta6)
R6 = np.sqrt(Y6**2 + (H - Z6) ** 2)
yref6 = np.sqrt(R6**2 - H**2)
gx6 = np.arange(-200, 200, 2.0)
gy6 = np.arange(yref6.min() - 10, yref6.max() + 10, 2.0)
G6x, G6y = np.meshgrid(gx6, gy6, indexing="ij")
slc6 = backproject(range_compress(raw6), eta6, tau0_6, G6x, G6y)
t_raw6 = time.time() - t0
t0 = time.time()
I6_id = image_domain(X6, Y6, Z6, np.abs(amp6) ** 2, gx6, gy6)
t_id6 = time.time() - t0
print(f"raw-signal simulation + focusing: {t_raw6:.1f} s ;  image-domain: {t_id6*1e3:.0f} ms")
I6 = np.abs(slc6) ** 2
L6 = 9
r_db6 = db(uniform_filter(I6 / I6.mean(), L6))
i_db6 = db(uniform_filter(I6_id / I6_id.mean(), L6))
valid6 = i_db6 > -20
r6 = np.corrcoef(r_db6[valid6], i_db6[valid6])[0, 1]

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
ext_full = [lon_d[0], lon_d[-1], lat_d[-1], lat_d[0]]
ax[0, 0].imshow(ls.hillshade(h_d, dx=dlon_m, dy=dlat_m), extent=ext_full, aspect=1 / np.cos(np.radians(lat0)))
ax[0, 0].imshow(h_d, extent=ext_full, cmap="terrain", alpha=0.45, aspect=1 / np.cos(np.radians(lat0)))
ax[0, 0].add_patch(Rectangle((lon0 - 240 / MLON, lat0 - 240 / MLAT), 480 / MLON, 480 / MLAT, fill=False, ec="r", lw=2))
ax[0, 0].set(xlabel="longitude [°]", ylabel="latitude [°]", title="(a) Copernicus GLO-30 DEM, study area (red: patch)")
ext6 = [ye6[0], ye6[-1], xn6[0], xn6[-1]]
im = ax[0, 1].imshow(Z6, extent=ext6, origin="lower", cmap="terrain")
ax[0, 1].contour(ye6, xn6, Z6, 12, colors="k", linewidths=0.4)
plt.colorbar(im, ax=ax[0, 1], label="height − mean [m]")
ax[0, 1].set(xlabel="east = ground range [m]  (sensor →)", ylabel="north = azimuth [m]", title="(b) Patch DEM (4 m resampling)")
ov = np.zeros(Z6.shape + (4,))
ov[lo6] = [1, 0, 0, 0.6]
ov[sh6] = [0, 0, 1, 0.7]
ax[0, 2].imshow(hill(Z6, 4.0), extent=ext6, origin="lower")
ax[0, 2].imshow(ov, extent=ext6, origin="lower")
ax[0, 2].set(xlabel="east [m]", ylabel="north [m]", title="(c) Layover (red) & shadow (blue) on the ground")
srange6 = c * (tau0_6 + np.arange(raw6.shape[1]) / fs) / 2 - R_c
ax[1, 0].imshow(np.abs(range_compress(raw6)), aspect="auto", cmap="magma",
                extent=[srange6[0], srange6[-1], eta6[-1], eta6[0]])
ax[1, 0].set(xlabel="slant range − R_c [m]", ylabel="η [s]", xlim=(srange6[0], (R6.max() - R_c) + 60),
             title="(d) Range-compressed raw data of the DEM patch")
ext6i = [gy6[0] - y_c, gy6[-1] - y_c, gx6[0], gx6[-1]]
ax[1, 1].imshow(r_db6, extent=ext6i, origin="lower", vmin=-12, vmax=8, aspect="auto")
ax[1, 1].set(xlabel="imaged ground range [m]", ylabel="azimuth [m]", title=f"(e) RAW-signal sim, focused ({t_raw6:.0f} s)")
ax[1, 2].imshow(i_db6, extent=ext6i, origin="lower", vmin=-12, vmax=8, aspect="auto")
ax[1, 2].set(xlabel="imaged ground range [m]", ylabel="azimuth [m]",
             title=f"(f) IMAGE-DOMAIN sim ({t_id6*1e3:.0f} ms) — r = {r6:.2f} vs (e)")
fig.suptitle("6 · Real terrain: raw-signal and image-domain simulation of a Copernicus DEM patch",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "06_dem_patch_raw_vs_image_domain")
plt.show()

# %% [markdown]
# The two simulations of the patch agree (panel e vs f); the remaining differences are speckle
# (the raw-signal simulation is coherent, the image-domain one is not). From here on we use the
# image-domain simulator on a 9 km × 9 km area around Innsbruck (30 m DEM resampled to 10 m).

# %%
LAT_C, LON_C = 47.285, 11.395                 # Innsbruck / Nordkette
SP = 10.0
xn9, ye9, Zb = dem_patch(LAT_C, LON_C, 4500, 4500, SP)
Zb = Zb - Zb.min()                           # reference plane: valley floor
Xb, Yb0 = np.meshgrid(xn9, ye9, indexing="ij")


def big_scene(look_deg):
    """Scatterer coordinates of the 9 km area for a given look angle at the scene centre."""
    yc = H * np.tan(np.deg2rad(look_deg))
    return Xb, yc + Yb0, yc


Xs, Ys, yc35 = big_scene(35)
sh_b, lo_b = toy_masks(Xs, Ys, Zb)
gxb = np.arange(xn9[0], xn9[-1], SP)
Rb = np.sqrt(Ys**2 + (H - Zb) ** 2)
yrefb = np.sqrt(Rb**2 - H**2)
gyb = np.arange(yrefb.min() - 50, yrefb.max() + 50, SP)
w_geom = np.where(sh_b, 0.0, sigma0_flat * SP**2)          # equal reflectivity per ground area
I_geom = image_domain(Xs, Ys, Zb, w_geom, gxb, gyb)

fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
extb = [ye9[0] / 1e3, ye9[-1] / 1e3, xn9[0] / 1e3, xn9[-1] / 1e3]
ax[0].imshow(hill(Zb, SP), extent=extb, origin="lower")
im = ax[0].imshow(Zb + 0, extent=extb, origin="lower", cmap="terrain", alpha=0.45)
ax[0].set(xlabel="east [km]  (sensor →)", ylabel="north [km]", title="(a) DEM 9 × 9 km (map geometry)")
extbi = [(gyb[0] - yc35) / 1e3, (gyb[-1] - yc35) / 1e3, gxb[0] / 1e3, gxb[-1] / 1e3]
ax[1].imshow(db(I_geom / sigma0_flat), extent=extbi, origin="lower", vmin=-6, vmax=8)
ax[1].set(xlabel="imaged ground range [km]", ylabel="azimuth [km]", title="(b) Simulated SAR image — geometry only [dB]")
ax[2].imshow(db(I_geom / sigma0_flat), extent=extbi, origin="lower", vmin=-6, vmax=8)
ax[2].contour(ye9 / 1e3, xn9 / 1e3, Zb, levels=np.arange(250, 2000, 250), colors="yellow", linewidths=0.5)
ax[2].set(xlabel="imaged ground range [km]", ylabel="azimuth [km]",
          title="(c) Same, with TRUE DEM contours: peaks are displaced toward the sensor")
fig.suptitle("6b · Image-domain simulation of the 9 km Innsbruck scene (look angle 35°, sensor in the west)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "06b_dem_area_geometric_sim")
plt.show()

# %% [markdown]
# ## 7. Geometric complications: foreshortening, layover, shadow
#
# Because a SAR sorts echoes by **range**, three distortions appear in mountains:
#
# * **Foreshortening** – a slope facing the sensor (but less steep than the look angle) is
#   compressed into few range cells → bright and short.
# * **Layover** – a slope facing the sensor *steeper* than the look angle: its top is closer to
#   the radar than its foot, so the image is **folded** over and mixed with the valley in front.
# * **Shadow** – a slope facing away steeper than $90°-\theta$, or terrain hidden behind a ridge:
#   no echo at all.
#
# First on a synthetic mountain profile, then on the real DEM.

# %%
yy7 = np.linspace(-3200, 2600, 20001)
peak, fs_deg, bs_deg = 1500.0, 55.0, 65.0
z7 = np.where(yy7 < 0, peak - np.abs(yy7) * np.tan(np.radians(fs_deg)), peak - yy7 * np.tan(np.radians(bs_deg)))
z7 = np.maximum(z7, 0)
hill2 = 300 * np.exp(-((yy7 + 2550) / 700) ** 2)     # a gentle hill (max slope ~20°): foreshortening only
z7 = np.maximum(z7, hill2)
z7 = np.convolve(np.pad(z7, 20, mode="edge"), np.ones(41) / 41, mode="valid")   # smooth the kinks
Y7 = y_c + yy7
alpha7 = np.arctan2(Y7, H - z7)
R7 = np.sqrt(Y7**2 + (H - z7) ** 2)
sh7, lo7 = shadow_layover(np.zeros_like(yy7), Y7, alpha7, R7, tol_alpha=1e-7, tol_R=0.05)
slope7 = np.gradient(z7, yy7)
cat = np.zeros(len(yy7), int)                  # 0 flat
cat[slope7 > 0.05] = 1                         # foreshortening (facing)
cat[slope7 < -0.05] = 2                        # facing away
cat[lo7] = 3                                   # layover
cat[sh7] = 4                                   # shadow
cols = ["#999999", "#2ca02c", "#9467bd", "#d62728", "#1f1f8f"]
names = ["flat", "foreshortening", "back-slope", "layover", "shadow"]

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
a = ax[0]
for k_ in range(5):
    m = cat == k_
    a.scatter(yy7[m], z7[m], s=2, color=cols[k_], label=names[k_])
d_ = np.array([np.sin(look), -np.cos(look)])
ip = np.argmax(z7)
a.plot([yy7[ip] - 2500 * d_[0], yy7[ip] + 2200 * d_[0]], [z7[ip] - 2500 * d_[1], z7[ip] + 2200 * d_[1]],
       "k--", lw=0.8, label="ray grazing the peak")
for y0 in np.arange(-3000, 2600, 400):
    a.annotate("", (y0, np.interp(y0, yy7, z7)), (y0 - 500 * d_[0], np.interp(y0, yy7, z7) - 500 * d_[1]),
               arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.6))
a.set(xlabel="ground range [m]", ylabel="height [m]", ylim=(-100, 2100), title="(a) Profile and radar rays (sensor at left)")
a.legend(fontsize=7, markerscale=4, loc="upper left")
a = ax[1]
for k_ in range(5):
    m = cat == k_
    a.scatter(yy7[m], R7[m] - R_c, s=2, color=cols[k_])
a.set(xlabel="ground range [m]", ylabel="slant range − R_c [m]", title="(b) Slant range of every ground point: the FOLD = layover")
a = ax[2]
w7 = np.where(sh7, 0, 1.0)
hb, eb = np.histogram(R7 - R_c, bins=np.arange((R7 - R_c).min() - 50, (R7 - R_c).max() + 50, rho_r), weights=w7)
a.bar(eb[:-1], hb / np.median(hb[hb > 0]), width=rho_r, color="k")
for k_, nm in ((3, "layover"), (4, "shadow")):
    rr = R7[cat == k_] - R_c
    if rr.size:
        a.axvspan(rr.min(), rr.max(), color=cols[k_], alpha=0.25, label=nm)
a.set(xlabel="slant range − R_c [m]", ylabel="echo power (relative)", yscale="log",
      title="(c) What the SAR records along this range line")
a.legend(fontsize=8)
fig.suptitle(f"7 · Geometric distortions on a synthetic mountain (fore-slope {fs_deg:.0f}°, back-slope {bs_deg:.0f}°, look 35°)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "07_geometric_distortions_profile")
plt.show()

# %%
# --- 7b: on the real DEM, and as a function of look angle ---
ovb = np.zeros(Zb.shape + (4,))
ovb[lo_b] = [1, 0, 0, 0.65]
ovb[sh_b] = [0, 0.2, 1, 0.8]
looks = np.arange(20, 51, 5)
stats = []
imgs = {}
for la in looks:
    Xl, Yl, ycl = big_scene(la)
    s_, l_ = toy_masks(Xl, Yl, Zb)
    stats.append((s_.mean() * 100, l_.mean() * 100))
    if la in (25, 45):
        Rl = np.sqrt(Yl**2 + (H - Zb) ** 2)
        yr = np.sqrt(Rl**2 - H**2)
        gyl = np.arange(yr.min() - 50, yr.max() + 50, SP)
        imgs[la] = (image_domain(Xl, Yl, Zb, np.where(s_, 0, sigma0_flat * SP**2), gxb, gyl), gyl - ycl)
stats = np.array(stats)

fig, ax = plt.subplots(1, 4, figsize=(19, 5))
ax[0].imshow(hill(Zb, SP), extent=extb, origin="lower")
ax[0].imshow(ovb, extent=extb, origin="lower")
ax[0].set(xlabel="east [km]", ylabel="north [km]", title="(a) Layover (red) / shadow (blue), look 35°")
ax[1].bar(looks - 1.2, stats[:, 1], 2.4, color="#d62728", label="layover")
ax[1].bar(looks + 1.2, stats[:, 0], 2.4, color="#1f1f8f", label="shadow")
ax[1].set(xlabel="look angle [°]", ylabel="% of area", title="(b) Steep look → more shadow; shallow → more layover")
ax[1].legend()
for a, la in zip(ax[2:], (25, 45)):
    im_, gy_rel = imgs[la]
    a.imshow(db(im_ / sigma0_flat), extent=[gy_rel[0] / 1e3, gy_rel[-1] / 1e3, gxb[0] / 1e3, gxb[-1] / 1e3],
             origin="lower", vmin=-6, vmax=8)
    a.set(xlabel="imaged ground range [km]", ylabel="azimuth [km]", title=f"({'c' if la == 25 else 'd'}) Simulated, look {la}°")
fig.suptitle("7b · Geometric distortions on the Innsbruck DEM and their dependence on the look angle",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "07b_geometric_distortions_dem")
plt.show()

# %% [markdown]
# ## 8. Radiometry: how bright is each facet?
#
# So far every square metre reflected the same power. Real backscatter depends on the
# **local incidence angle** $\theta_{loc}$ between the radar beam and the surface normal:
# slopes facing the radar reflect more. We compare three classic models
#
# * $\sigma^0 = \cos\theta_{loc}$ (simple),  $\sigma^0=\cos^2\theta_{loc}$ (Lambert),
# * Muhleman: $\sigma^0 = 0.0133\,\cos\theta/(\sin\theta + 0.1\cos\theta)^3$ (used in many SAR simulators),
#
# then build the simulated image as the sequence
# **local incidence → $\sigma^0$ map → × illuminated area / pixel area → speckle → multilook**.

# %%
from sarsim.rdsim import sigma0_model

zN, zE = np.gradient(Zb, SP, SP)                         # dz/dnorth, dz/deast
Nn = np.sqrt(1 + zN**2 + zE**2)
cos_loc = (Ys * zE + (H - Zb)) / (Rb * Nn)                # -u.n for the toy geometry
theta_loc = np.degrees(np.arccos(np.clip(cos_loc, -1, 1)))
area_true = SP**2 * Nn
s0_mu = sigma0_model(cos_loc, "muhleman")
s0_mu[sh_b] = 0
I_area = image_domain(Xs, Ys, Zb, np.where(sh_b, 0, area_true), gxb, gyb)       # illuminated area / pixel area
I_rad = image_domain(Xs, Ys, Zb, s0_mu * area_true, gxb, gyb)                    # sigma0-equivalent image
speck = rng.exponential(1.0, I_rad.shape)
I_sl = I_rad * speck
I_ml = uniform_filter(I_sl, 4)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
th = np.linspace(0, 89, 300)
ct = np.cos(np.radians(th))
for mdl, lab in (("cos", "cos θ"), ("lambert", "cos² θ (Lambert)"), ("muhleman", "Muhleman")):
    yv = sigma0_model(ct, mdl)
    ax[0].plot(th, db(yv / sigma0_model(np.cos(np.radians(35)), mdl)), label=lab)
ax[0].set(xlabel="local incidence angle [°]", ylabel="σ⁰ relative to θ = 35° [dB]", ylim=(-25, 15),
          title="(a) Backscatter models")
ax[0].legend()
ax[1].hist(theta_loc[~sh_b].ravel(), bins=90, range=(0, 90), color="tab:gray")
ax[1].axvline(35, color="r", ls=":")
ax[1].set(xlabel="local incidence angle [°]", ylabel="# DEM facets", title="(b) Local incidence angles in the 9 km scene")
fig.tight_layout()
save(fig, "08a_backscatter_models")
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
im = ax[0, 0].imshow(theta_loc, extent=extb, origin="lower", cmap="RdYlBu_r", vmin=0, vmax=90)
plt.colorbar(im, ax=ax[0, 0], label="θ_loc [°]")
ax[0, 0].set(title="(1) Local incidence angle (map)", xlabel="east [km]", ylabel="north [km]")
im = ax[0, 1].imshow(db(s0_mu), extent=extb, origin="lower", vmin=-25, vmax=5)
plt.colorbar(im, ax=ax[0, 1], label="σ⁰ [dB]")
ax[0, 1].set(title="(2) σ⁰(θ_loc), Muhleman (map)", xlabel="east [km]", ylabel="north [km]")
im = ax[0, 2].imshow(db(I_area), extent=extbi, origin="lower", vmin=-6, vmax=8)
plt.colorbar(im, ax=ax[0, 2], label="dB")
ax[0, 2].set(title="(3) Illuminated area / pixel area (radar)", xlabel="imaged ground range [km]", ylabel="azimuth [km]")
for a, img_, ttl in ((ax[1, 0], I_rad, "(4) σ⁰ × area → noise-free SAR image"),
                     (ax[1, 1], I_sl, "(5) + speckle (single look)"),
                     (ax[1, 2], I_ml, "(6) multilooked 4 × 4 → final simulated image")):
    im = a.imshow(db(img_), extent=extbi, origin="lower", vmin=-25, vmax=5)
    plt.colorbar(im, ax=a, label="σ⁰ [dB]")
    a.set(title=ttl, xlabel="imaged ground range [km]", ylabel="azimuth [km]")
fig.suptitle("8 · Radiometric simulation chain on the Innsbruck DEM", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "08b_radiometric_chain")
plt.show()

# %% [markdown]
# ## 9. Real orbit, real DEM, real image
#
# Finally the same physics is applied in the **real Sentinel-1 geometry**:
#
# 1. A Sentinel-1 IW GRD scene (VV) covering the study area is found on Microsoft Planetary
#    Computer; its product annotation provides the **orbit state vectors**, azimuth timing,
#    slant-to-ground-range polynomials and the radiometric calibration LUT.
# 2. Every (3× oversampled) DEM facet is converted to ECEF (DEM heights + EGM2008 geoid →
#    ellipsoidal heights) and **range-Doppler geocoded**: the zero-Doppler time gives the GRD
#    line, the slant range (→ SRGR polynomial) the GRD pixel.
# 3. Local incidence angle, shadow and layover are computed per facet, and
#    $\sigma^0_{Muhleman}(\theta_{loc})\times A_{facet}$ is accumulated into the GRD grid and
#    divided by the ellipsoid pixel area — exactly what ESA's $\sigma^0$ calibration refers to.
# 4. The real DN are calibrated to $\sigma^0$, both images are multilooked 5 × 5, co-registered
#    and compared.
#
# Implementation: `sarsim/s1.py`, `sarsim/rdsim.py`, `sarsim/compare.py`.

# %%
from scipy.ndimage import binary_erosion, binary_fill_holes
from shapely.geometry import shape as shp

from sarsim import compare, rdsim

item = sdata.find_s1(sarsim.AOI_BBOX, sarsim.S1_DATES, orbit_state="ascending")
scene_dir = sdata.get_s1(item, dem_full)
prod, s0_real, meta = sdata.load_s1(scene_dir)
print(f"scene      : {item.id}")
print(f"pass       : {prod.pass_direction}, {prod.mission}, {meta['datetime']}")
print(f"GRD window : {s0_real.shape} px at line {meta['row0']}, pixel {meta['col0']} "
      f"(pixel {prod.dgr:.0f} m x {prod.daz:.0f} m), incidence mid-swath {prod.inc_mid:.1f}°")
print(f"geoid N    : {meta['geoid_N']:.1f} m (EGM2008)   orbit fit velocity RMS {prod.orbit.vel_rms*1e3:.2f} mm/s")

t0 = time.time()
sim = rdsim.simulate_grd(prod, dem_full, meta["geoid_N"], meta["row0"], meta["col0"], s0_real.shape,
                         oversample=3, model="muhleman")
print(f"simulation : {time.time()-t0:.0f} s")

# check our geocoding against ESA's own geolocation grid
from sarsim.geodesy import llh2ecef

gl = prod.grid
inside = (gl["latitude"] > 46.5) & (gl["latitude"] < 48) & (gl["longitude"] > 10.5) & (gl["longitude"] < 12.5)
gg = prod.geocode(llh2ecef(gl["latitude"][inside], gl["longitude"][inside], gl["height"][inside]))
print(f"geocoding vs ESA geolocation grid ({inside.sum()} points): "
      f"line {np.mean(gg['line']-gl['line'][inside]):+.3f} ± {np.std(gg['line']-gl['line'][inside]):.3f}, "
      f"pixel {np.mean(gg['pixel']-gl['pixel'][inside]):+.2f} ± {np.std(gg['pixel']-gl['pixel'][inside]):.2f}")

# %%
ML = 5
real_ml = compare.multilook(s0_real, ML)
sim_ml = compare.multilook(sim["sigma0"], ML)
foot = binary_erosion(binary_fill_holes(sim["footprint"]), iterations=8)   # fill gaps on steep back-slopes
shift, box = compare.coregister(compare.db(real_ml), compare.db(sim_ml), foot)
sim_reg = compare.apply_shift(sim_ml, shift)
foot_reg = compare.apply_shift(foot.astype(float), shift) > 0.99
lay_reg = compare.apply_shift(sim["radar_layover"].astype(float), shift) > 0.5
sh_reg = compare.apply_shift(sim["radar_shadow"].astype(float), shift) > 0.5
valid = foot_reg & (real_ml > 0) & (sim_reg > 0)
bias = np.median(compare.db(real_ml)[valid] - compare.db(sim_reg)[valid])
real_db = compare.db(real_ml)
sim_db = compare.db(sim_reg) + bias               # absolute level calibrated by one constant
met = compare.metrics(real_db, sim_db, valid, box)
print(f"residual offset real − sim: {shift[0]:+.2f} lines, {shift[1]:+.2f} pixels "
      f"(= {shift[0]*prod.daz:+.0f} m az, {shift[1]*prod.dgr:+.0f} m rg)")
print(f"model bias (real − Muhleman): {bias:+.2f} dB")
for k_, v_ in met.items():
    print(f"  {k_:>22s}: {v_:.3f}" if isinstance(v_, float) else f"  {k_:>22s}: {v_}")

# %%
fp = shp(item.geometry)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
xx_, yy_ = fp.exterior.xy
ax[0].fill(xx_, yy_, color="tab:blue", alpha=0.15, label="Sentinel-1 scene")
ax[0].plot(xx_, yy_, color="tab:blue")
b = sarsim.AOI_BBOX
ax[0].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color="r", label="study area (DEM)"))
ax[0].set(xlabel="longitude [°]", ylabel="latitude [°]", title=f"(a) {prod.mission} {prod.pass_direction} scene footprint")
ax[0].legend(fontsize=8)
ax[0].set_aspect(1 / np.cos(np.radians(47)))
ax[1].imshow(ls.hillshade(h_d, dx=dlon_m, dy=dlat_m), extent=ext_full, aspect=1 / np.cos(np.radians(47)))
ax[1].imshow(h_d, extent=ext_full, cmap="terrain", alpha=0.45, aspect=1 / np.cos(np.radians(47)))
ax[1].set(xlabel="longitude [°]", ylabel="latitude [°]", title="(b) Copernicus GLO-30 DEM of the study area")
ax[2].imshow(real_db, origin="lower", cmap="gray", vmin=-22, vmax=3)
ax[2].set(xlabel="GRD pixel (ground range, east →)", ylabel="GRD line (azimuth, north ↑)",
          title="(c) Real Sentinel-1 σ⁰ VV [dB] (radar geometry)")
fig.suptitle("9a · Study area and data", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "09a_study_area_data")
plt.show()

# %%
shown = np.where(foot_reg, 1.0, np.nan)
cb = ((np.arange(real_db.shape[0])[:, None] // 250 + np.arange(real_db.shape[1])[None, :] // 250) % 2).astype(bool)
fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
ax[0].imshow(real_db * shown, origin="lower", cmap="gray", vmin=-22, vmax=3)
ax[0].set(title="(a) REAL Sentinel-1 σ⁰ VV [dB]", xlabel="GRD pixel", ylabel="GRD line")
ax[1].imshow(sim_db * shown, origin="lower", cmap="gray", vmin=-22, vmax=3)
ax[1].set(title="(b) SIMULATED from DEM + orbit [dB]", xlabel="GRD pixel", ylabel="GRD line")
ax[2].imshow(np.where(cb, real_db, sim_db) * shown, origin="lower", cmap="gray", vmin=-22, vmax=3)
ax[2].set(title="(c) Checkerboard real / simulated", xlabel="GRD pixel", ylabel="GRD line")
fig.suptitle(f"9b · Simulated vs real Sentinel-1 — Pearson r = {met['pearson_r']:.2f}, SSIM = {met['ssim']:.2f}, "
             f"residual offset {shift[0]:+.1f} / {shift[1]:+.1f} px", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "09b_real_vs_simulated")
plt.show()

# %%
# map geometry: sample both radar images at every DEM pixel (back-geocoding)
mrow, mcol = sim["map"]["row"], sim["map"]["col"]
real_map = compare.sample_radar_to_map(real_ml, mrow + shift[0], mcol + shift[1])
sim_map = compare.sample_radar_to_map(sim_ml, mrow, mcol)
tl = sim["map"]["theta_loc"]
good = (~sim["map"]["shadow"]) & (~sim["map"]["layover"]) & np.isfinite(real_map) & (real_map > 0)
bins_t = np.arange(0, 91, 3)
cen = (bins_t[:-1] + bins_t[1:]) / 2
med = np.array([np.median(compare.db(real_map[good & (tl >= b0) & (tl < b1)]))
                if np.sum(good & (tl >= b0) & (tl < b1)) > 200 else np.nan for b0, b1 in zip(bins_t[:-1], bins_t[1:])])

fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
ov9 = np.zeros(real_db.shape + (4,))
ov9[lay_reg & foot_reg] = [1, 0, 0, 0.55]
ov9[sh_reg & foot_reg] = [0, 0.4, 1, 0.8]
ax[0].imshow(real_db * shown, origin="lower", cmap="gray", vmin=-22, vmax=3)
ax[0].imshow(ov9, origin="lower")
ax[0].set(title="(a) Predicted layover (red) / shadow (blue) on the REAL image", xlabel="GRD pixel", ylabel="GRD line")
hh = ax[1].hist2d(sim_db[valid], real_db[valid], bins=120, range=[[-25, 5], [-25, 5]], cmap="magma_r", cmin=1)
ax[1].plot([-25, 5], [-25, 5], "c--", lw=1)
ax[1].set(xlabel="simulated σ⁰ [dB]", ylabel="real σ⁰ [dB]", title=f"(b) Pixel-by-pixel (r = {met['pearson_r']:.2f})")
ax[2].plot(cen, med, "ko-", ms=4, label="real (median per bin)")
for mdl, lab in (("muhleman", "Muhleman"), ("cos", "cos θ"), ("lambert", "cos² θ")):
    mv = db(sigma0_model(np.cos(np.radians(cen)), mdl))
    off = np.nanmedian(med - mv)
    ax[2].plot(cen, mv + off, label=f"{lab} ({off:+.1f} dB)")
ax[2].set(xlabel="local incidence angle [°]", ylabel="σ⁰ [dB]", ylim=(-25, 5),
          title="(c) Real σ⁰ vs local incidence angle (no layover/shadow)")
ax[2].legend(fontsize=8)
fig.suptitle("9c · Where and why simulation and reality agree", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "09c_analysis")
plt.show()

# %% [markdown]
# **Reading 9c.** (a) The predicted layover bands sit on the bright, sensor-facing ridges of the real
# image and the predicted shadows on its black patches — the geometry is right. (b) The diagonal
# ridge of the 2-D histogram is the terrain-driven agreement; the **vertical streak** at ≈ −8 dB is
# the flat Inn valley floor: the DEM gives one value there, while the real image varies with land
# cover (city, fields, river, forest) — information a DEM simply does not contain. (c) The real
# backscatter falls off with local incidence angle more gently than the Muhleman law at small
# angles; a cos²θ law fits 20–60° well. Fitting the model to the data is done in the pipeline
# (phase 3).

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
asp = 1 / np.cos(np.radians(47))
ax[0].imshow(ls.hillshade(h_d, dx=dlon_m, dy=dlat_m), extent=ext_full, aspect=asp)
ax[0].set(title="(a) DEM hillshade", xlabel="longitude [°]", ylabel="latitude [°]")
ax[1].imshow(compare.db(real_map), extent=ext_full, cmap="gray", vmin=-22, vmax=3, aspect=asp)
ax[1].set(title="(b) REAL σ⁰ back-geocoded to the DEM grid [dB]", xlabel="longitude [°]")
ax[2].imshow(compare.db(sim_map) + bias, extent=ext_full, cmap="gray", vmin=-22, vmax=3, aspect=asp)
ax[2].set(title="(c) SIMULATED σ⁰ in map geometry [dB]", xlabel="longitude [°]")
fig.suptitle("9d · The same comparison in map geometry (north up)", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "09d_map_geometry")
plt.show()
print(f"total notebook run time: {(time.time()-T_START)/60:.1f} min")

# %% [markdown]
# ## Summary
#
# * A point target, a line of targets and a moving satellite show how pulse compression gives
#   range resolution and the synthetic aperture gives azimuth resolution (§1–3).
# * Distributed surfaces create speckle (§4); topography moves and re-weights the energy
#   (foreshortening, §5) — and an image-domain simulator reproduces the raw-signal result (§5–6).
# * Layover and shadow are purely geometric and depend strongly on the look angle (§7); the local
#   incidence angle controls the brightness of each facet (§8).
# * Driven by the real Sentinel-1 orbit and the Copernicus DEM, the simulator places mountains,
#   ridges, layover and shadow at the right GRD pixels (residual offset printed above) and
#   reproduces the real image's structure; the remaining differences are land cover (forest,
#   settlements, water, snow), which a DEM alone cannot know.
