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
| `report/` | LaTeX report |
| `poster/` | newspaper-style research poster |

## Run it

```bash
uv sync                                   # Python 3.12 environment from uv.lock
uv run pytest -q tests                    # unit tests
uv run python run_all.py                  # downloads ~60 MB on first run, ~15 min
uv run jupyter lab notebooks/             # the walkthrough
```

## GUI

```bash
uv run streamlit run app/streamlit_app.py
```

Opens a browser app (wide layout, inputs in the sidebar, single *Run simulation* button):

* **Terrain:** a preset, a bounding box (Copernicus GLO-30 is downloaded and cached), or an
  uploaded DEM GeoTIFF in any projection. Large areas are downsampled to stay below about
  4 million facets, and the app says so.
* **Geometry:** *Generic Sentinel-1-like* (offline, anywhere: pass, look angle, heading; flat
  Earth, straight orbit) or *Real Sentinel-1* (finds a covering IW GRD scene for a date range,
  simulates in its exact geometry and compares with the real image: r, SSIM, RMSE, offset).
* **Backscatter:** fitted cos^1.98 law (default), Muhleman, Lambert, cosine or constant, with
  optional speckle and multilook.
* **Outputs:** *Overview* (hillshade, area statistics, layover and shadow percentages, run time)
  and *Results* (sigma0 in radar and map geometry, layover and shadow overlay, local incidence
  map and histogram, sigma0 versus local incidence). The ZIP download holds georeferenced
  GeoTIFFs, the radar-geometry image (GeoTIFF and NPY) and `metadata.json`.

The info button in the top-right corner explains the method. Core logic lives in
`sarsim/pipeline.py` and `sarsim/generic.py`; `tests/test_app_core.py` runs it headless.

To change the study area or dates, edit `sarsim/__init__.py` (`AOI_BBOX`, `S1_DATES`).

## Method in one paragraph

Each DEM cell (3× oversampled, heights converted from EGM2008 to WGS84) becomes an ECEF facet
with a normal and an area. Its zero-Doppler time on the interpolated Sentinel-1 orbit gives the
GRD line; its slant range, through the annotated slant-to-ground-range polynomial, gives the GRD
pixel. Facets are ordered by ground position along each azimuth line to flag radar shadow and
layover. Each lit facet contributes σ⁰(θ_loc)·A to its GRD pixel (bilinear splatting), divided by
the ellipsoid pixel area so that the result is directly comparable with ESA-calibrated σ⁰.
