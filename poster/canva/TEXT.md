# Poster text for Canva

Fonts: Playfair Display 900 (masthead), Oswald 600-700 (headlines), Source Serif 4 (body). Paper colour #efe8d8, ink #1d1b18, accent #1f3a5f. Size A1 portrait (594 x 841 mm).

Research Edition | Remote Sensing Saturday, November 28, 2026 Price: one DEM & two orbits
## The Radar Gazette
Vol. I, No. 1 All the echoes that fit to print Innsbruck edition
Introduction Study Area & Data Methodology Results Conclusions
## Mountains simulated from space match real radar
DEM-based SAR simulation predicts Sentinel-1 images of the Alps, layover, shadow and all, with a correlation of r = 0.71.
By Vedansh Shrivastava (230003086) | Supervisor: Dr. Unmesh Khati, IIT Indore
A radar satellite does not take pictures: it sorts echoes by distance. Over mountains, that turns slopes into bright lines, folds peaks over valleys and hides whole hillsides in shadow. This study rebuilds a SAR from first principles, feeds it the Copernicus DEM and the real Sentinel-1 orbit, and asks: how much of a real Alpine radar image can terrain alone explain?
[IMAGE: Real versus simulated Sentinel-1 image of Innsbruck]
REAL (LEFT) | SIMULATED FROM DEM AND ORBIT (RIGHT) Sentinel-1 σ⁰ VV over Innsbruck, Austria (radar geometry, north up)
## Study at a glance
r Correlation (asc / desc) 0.71 / 0.72 5×5 multilook, dB, 4.9 M pixels
⌖ Geocoding vs ESA 0.016 px max error over 210 grid points
Δ Image vs simulation offset 0.3–3.5 px ≤ 35 m: the products' own geolocation
θ Fitted backscatter law cos 1.98 θ fit on ascending, holds on descending
▲ Terrain 552–2789 m Copernicus GLO-30, 23 × 22 km
€ Data cost 0 all open data, no login
## 1. Introduction
Why mountains confuse a radar
A side-looking SAR measures slant range : terrain height moves every point towards the sensor by h cot θ.
Slopes facing the radar are foreshortened or fall into layover ; steep slopes facing away vanish into shadow .
Simulating the image from a DEM is the standard way to explain, mask and correct these effects; this work builds such a simulator from scratch and tests it against reality.
## 2. Study Area & Data
Innsbruck: a city at the foot of a 2,300 m wall
[IMAGE: Study area]
Figure 1: Scene footprint, Copernicus DEM and calibrated Sentinel-1 σ⁰ of the study area.
## Scene report
DEM Copernicus GLO-30 (EGM2008 heights converted to WGS84, N = 49.6 m)
Ascending Sentinel-1C IW GRDH VV, 2025-09-27, θ 38.9°
Descending Sentinel-1C IW GRDH VV, 2025-09-26, θ 38.9°
Land cover ESA WorldCover 2021, 10 m
Access Microsoft Planetary Computer (anonymous)
## 3. Methodology
Part I: from a point to a mountain
A raw-signal simulator follows the radar physics step by step: chirp, echo, range compression, synthetic aperture and focusing by back-projection. Measured resolutions (5.5 m azimuth, 4.7 m ground range) match theory.
[IMAGE: signal chain]
Figure 2: Three point targets: range & Doppler history, raw data, range-compressed data and the focused image.
Distributed scatterers create speckle; topography re-weights the energy. A fast image-domain simulator reproduces the raw-signal result on a real DEM patch with r = 0.97 , in milliseconds instead of minutes.
[IMAGE: distortions]
Figure 3: Foreshortening, layover (the slant-range fold) and shadow on a synthetic mountain.
Part II: real orbit, real terrain
Geometry: 7 M DEM facets (3× oversampled) are range-Doppler geocoded with the annotated orbit (zero-Doppler Newton solve) and slant-to-ground-range polynomials.
Distortions: facets sorted along each azimuth line give shadow (look-angle test) and layover (slant-range test).
Radiometry: σ⁰(θ loc ) A facet splatted into GRD pixels and divided by the ellipsoid pixel area, directly comparable with ESA σ⁰.
Comparison: 5×5 multilook, sub-pixel co-registration, Pearson r, SSIM, RMSE; WorldCover breakdown.
“Polynomials are piece-wise: blending them cost us 3 pixels.” ESA's SRGR sets applied valid-from: 0.016 px
[IMAGE: radiometry]
Figure 4: Radiometric chain: local incidence, σ⁰ map, illuminated area, image, speckle, multilook.
## 4. Results
Geometry nailed; land cover is the rest
[IMAGE: asc desc]
Figure 5: Real vs simulated, ascending (looking east) and descending (looking west), with predicted layover (red) and shadow (blue).
Backscatter model r SSIM RMSE dB
Constant 0.53 0.38 3.8
cos θ 0.66 0.47 3.7
cos²θ (Lambert) 0.70 0.51 4.4
Muhleman 0.73 0.49 4.6
Fitted cos 1.98 θ 0.71 0.51 4.3
Geometry alone (constant σ⁰) already explains much of the image; the data-fitted law is cos 1.98 θ, close to Lambert, and transfers unchanged to the descending pass (r = 0.72). Real σ⁰ inside predicted shadow has a median of -21.0 dB, inside predicted layover -3.5 dB.
[IMAGE: land cover]
Figure 6: Residual (real − simulated) per WorldCover class. Bare / sparse is +3.2 dB brighter and water -3.8 dB darker than the DEM predicts.
[IMAGE: residual maps]
Figure 7: Real − simulated σ⁰ in map geometry: red = brighter than predicted (city, rock), blue = darker (river, fields).
## 5. Conclusions
Range-Doppler geocoding with the annotated orbit reproduces ESA's grid to 0.016 px ; simulated layover & shadow sit where the real image is brightest and darkest.
DEM-only simulation predicts Sentinel-1 σ⁰ of Alpine terrain with r = 0.71 / 0.72 (asc / desc); the fitted cos 1.98 law transfers between geometries.
The remaining disagreement is land cover : buildings, water and vegetation structure that no DEM contains. Next: class-specific backscatter, double bounce, terrain flattening.
## References
Cumming & Wong (2005) Digital Processing of SAR Data . Artech House.
Moreira et al. (2013) A tutorial on SAR. IEEE GRSM 1(1), 6–43.
Small (2011) Flattening gamma. IEEE TGRS 49(8), 3081–3093.
Muhleman (1964) Radar scattering from Venus and the Moon. AJ 69, 34–41.
Torres et al. (2012) GMES Sentinel-1 mission. RSE 120, 9–24.
## Acknowledgements
[IMAGE: QR code to the repository]
Copernicus DEM © ESA/Airbus; Sentinel-1 & WorldCover © ESA; data via Microsoft Planetary Computer. github.com/Vedansh7-7/dem-sar-simulation
“All models are wrong, some echo back.” ◉ Stay coherent, mind the layover
