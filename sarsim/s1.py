"""Sentinel-1 GRD product annotation: orbit, timing, slant->ground range, calibration,
and range-Doppler (zero-Doppler) geocoding of ground points into GRD (line, pixel)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import RegularGridInterpolator

from .geodesy import C


def _dt(s: str) -> np.datetime64:
    return np.datetime64(s, "ns")


class Orbit:
    """Polynomial fit of the annotated ECEF state vectors (smooth over a few minutes)."""

    def __init__(self, t, pos, vel, deg: int = 7):
        self.tc = t.mean()
        self.ts = (t.max() - t.min()) / 2
        tt = (t - self.tc) / self.ts
        deg = min(deg, len(t) - 2)
        self.cp = [P.polyfit(tt, pos[:, k], deg) for k in range(3)]
        self.cv = [P.polyder(c) / self.ts for c in self.cp]
        self.ca = [P.polyder(c) / self.ts for c in self.cv]
        # sanity: fitted velocity vs annotated velocity
        self.vel_rms = float(np.sqrt(np.mean((self.vel(t) - vel) ** 2)))

    def _ev(self, cs, t):
        tt = (np.asarray(t, float) - self.tc) / self.ts
        return np.stack([P.polyval(tt, c) for c in cs], axis=-1)

    def pos(self, t):
        return self._ev(self.cp, t)

    def vel(self, t):
        return self._ev(self.cv, t)

    def acc(self, t):
        return self._ev(self.ca, t)


class S1Product:
    """Parsed GRD product annotation (one polarisation).

    All times are seconds relative to ``t0`` = productFirstLineUtcTime, so that
    ``line = t / azimuth_time_interval``.
    """

    def __init__(self, ann_xml: str, cal_xml: str | None = None):
        r = ET.fromstring(ann_xml)
        ii = r.find("imageAnnotation/imageInformation")
        pi = r.find("generalAnnotation/productInformation")
        self.t0 = _dt(ii.find("productFirstLineUtcTime").text)
        self.t_last = self.sec(_dt(ii.find("productLastLineUtcTime").text))
        self.dt_az = float(ii.find("azimuthTimeInterval").text)
        self.dgr = float(ii.find("rangePixelSpacing").text)
        self.daz = float(ii.find("azimuthPixelSpacing").text)
        self.n_lines = int(ii.find("numberOfLines").text)
        self.n_samples = int(ii.find("numberOfSamples").text)
        self.near_slant_range_time = float(ii.find("slantRangeTime").text)
        self.inc_mid = float(ii.find("incidenceAngleMidSwath").text)
        self.f0 = float(pi.find("radarFrequency").text)
        self.wavelength = C / self.f0
        self.pass_direction = pi.find("pass").text
        self.mission = r.find("adsHeader/missionId").text

        orbs = r.findall("generalAnnotation/orbitList/orbit")
        t = np.array([self.sec(_dt(o.find("time").text)) for o in orbs])
        pos = np.array([[float(o.find(f"position/{k}").text) for k in "xyz"] for o in orbs])
        vel = np.array([[float(o.find(f"velocity/{k}").text) for k in "xyz"] for o in orbs])
        keep = (t > -90) & (t < self.t_last + 90)
        self.orbit = Orbit(t[keep], pos[keep], vel[keep])

        cc = r.findall("coordinateConversion/coordinateConversionList/coordinateConversion")
        self.cc_t = np.array([self.sec(_dt(c.find("azimuthTime").text)) for c in cc])
        self.cc_sr0 = np.array([float(c.find("sr0").text) for c in cc])
        self.cc_srgr = np.array([[float(v) for v in c.find("srgrCoefficients").text.split()] for c in cc])

        g = r.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
        f = lambda k: np.array([float(p.find(k).text) for p in g])  # noqa: E731
        self.grid = {k: f(k) for k in ("line", "pixel", "latitude", "longitude", "height", "incidenceAngle")}

        self._cal = _calibration_lut(cal_xml) if cal_xml else None

    # ---------------------------------------------------------------- timing
    def sec(self, t: np.datetime64) -> float:
        return float((t - self.t0) / np.timedelta64(1, "ns")) * 1e-9

    # ------------------------------------------------------------- geocoding
    def zero_doppler(self, xyz, n_iter: int = 8):
        """Solve (X - S(t)) . V(t) = 0 for each ground point X (N,3).

        Returns azimuth time t [s rel. t0], slant range R [m], sensor position S (N,3).
        """
        X = np.asarray(xyz, float)
        t = np.full(X.shape[0], self.t_last / 2)
        for _ in range(n_iter):
            S, V, A = self.orbit.pos(t), self.orbit.vel(t), self.orbit.acc(t)
            d = X - S
            f = np.einsum("ij,ij->i", d, V)
            fp = -np.einsum("ij,ij->i", V, V) + np.einsum("ij,ij->i", d, A)
            t = t - f / fp
        S = self.orbit.pos(t)
        R = np.linalg.norm(X - S, axis=1)
        return t, R, S

    def ground_range(self, t, R):
        """Slant range -> GRD ground range [m] via the annotated SRGR polynomials.

        Each polynomial set is valid from its azimuth time until the next one. (Linear
        blending of neighbouring sets is wrong: consecutive sets can differ by tens of
        pixels; with the 'valid-from' rule ESA's own geolocation grid is reproduced to
        0.02 px.)
        """
        t = np.asarray(t, float)
        j = np.clip(np.searchsorted(self.cc_t, t, side="right") - 1, 0, len(self.cc_t) - 1)
        x = R - self.cc_sr0[j]
        out = np.zeros_like(x)
        for k in range(self.cc_srgr.shape[1] - 1, -1, -1):  # Horner
            out = out * x + self.cc_srgr[j, k]
        return out

    def geocode(self, xyz):
        """Ground points (N,3 ECEF) -> dict(line, pixel, t, R, S)."""
        t, R, S = self.zero_doppler(xyz)
        return {"line": t / self.dt_az, "pixel": self.ground_range(t, R) / self.dgr, "t": t, "R": R, "S": S}

    # ----------------------------------------------------------- calibration
    def sigma0(self, dn, row0: int = 0, col0: int = 0):
        """Calibrate DN (window starting at row0/col0) to sigma0 (linear)."""
        if self._cal is None:
            raise ValueError("no calibration XML given")
        rows = np.arange(dn.shape[0]) + row0
        cols = np.arange(dn.shape[1]) + col0
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        a = self._cal((rr.ravel(), cc.ravel())).reshape(dn.shape)
        return dn.astype(np.float64) ** 2 / a**2


def _calibration_lut(cal_xml: str):
    r = ET.fromstring(cal_xml)
    vecs = r.findall("calibrationVectorList/calibrationVector")
    lines = np.array([int(v.find("line").text) for v in vecs])
    pix = [np.array(v.find("pixel").text.split(), int) for v in vecs]
    sig = [np.array(v.find("sigmaNought").text.split(), float) for v in vecs]
    n = min(len(p) for p in pix)
    return RegularGridInterpolator(
        (lines, pix[0][:n]), np.array([s[:n] for s in sig]), bounds_error=False, fill_value=None
    )
