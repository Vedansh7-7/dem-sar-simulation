# DEM-based SAR simulation and comparison with real Sentinel-1 imagery

Simulate what a Synthetic Aperture Radar *should* see from a Digital Elevation Model and the
real satellite orbit, then compare it pixel by pixel with an actual Sentinel-1 image.

* **Study area:** Innsbruck, Tyrol (Inn valley, Nordkette, Patscherkofel), 0.3° × 0.2°
* **DEM:** Copernicus GLO-30 · **SAR:** Sentinel-1 IW GRDH (VV), ascending + descending ·
  **Land cover:** ESA WorldCover — all streamed anonymously from Microsoft Planetary Computer.

## What is inside

| Path | Content |
|---|---|
| `notebooks/01_sar_simulation_walkthrough.ipynb` | Step-by-step tutorial: point → line → moving satellite (raw → range-compressed → focused) → flat surface (speckle) → sine surface → DEM → layover/shadow → radiometry → real Sentinel-1 comparison. Source: the jupytext `.py` next to it. |
| `sarsim/s1.py` | Sentinel-1 annotation parser, orbit interpolation, zero-Doppler range-Doppler geocoding, SRGR conversion, σ⁰ calibration |
| `sarsim/rdsim.py` | DEM facet geometry, local incidence, shadow/layover, backscatter models, rendering into GRD geometry |
| `sarsim/compare.py` | multilooking, sub-pixel co-registration, similarity metrics, back-geocoding |
| `sarsim/data.py` | cached data access (DEM, Sentinel-1 window, WorldCover) |
| `run_all.py` | the full experiment pipeline (E1–E5) → `results/`, `figures/pipeline/` |
| `tests/` | unit tests (geometry, models, geocoding vs ESA's geolocation grid) |
| `report/` | LaTeX report (`main.pdf`, 23 pages) |
| `poster/a3/` | A3 academic poster in LaTeX/TikZ, with data-flow and design-evolution diagrams |
| `poster/` | A1 newspaper-style poster (HTML → PDF/PNG) |
| `poster/canva/` | Canva import package: poster PDF, all figures, all poster text |

## Run it

```bash
uv sync                                   # Python 3.12 environment from uv.lock
uv run pytest -q tests                    # unit tests
uv run python run_all.py                  # downloads ~60 MB on first run, ~15 min
uv run jupyter lab notebooks/             # the walkthrough
```

To change the study area or dates, edit `sarsim/__init__.py` (`AOI_BBOX`, `S1_DATES`).

## Method in one paragraph

Each DEM cell (3× oversampled, heights converted from EGM2008 to WGS84) becomes an ECEF facet
with a normal and an area. Its zero-Doppler time on the interpolated Sentinel-1 orbit gives the
GRD line; its slant range, through the annotated slant-to-ground-range polynomial, gives the GRD
pixel. Facets are ordered by ground position along each azimuth line to flag radar shadow and
layover. Each lit facet contributes σ⁰(θ_loc)·A to its GRD pixel (bilinear splatting), divided by
the ellipsoid pixel area so that the result is directly comparable with ESA-calibrated σ⁰.
