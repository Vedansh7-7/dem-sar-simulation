"""Turn results/metrics.json into LaTeX macros and tables (report/generated/) and the QR code."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "results" / "metrics.json").read_text())
REPO_URL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/Vedansh7-7/dem-sar-simulation"


def f(x, d=2):
    return f"{x:.{d}f}"


asc, desc = R["E1_E2_summary"]["ascending_fitted"], R["E1_E2_summary"]["descending_fitted"]
gA, gD = R["E3_geocoding_vs_esa"]["ascending"], R["E3_geocoding_vs_esa"]["descending"]
sA, sD = R["scenes"]["ascending"], R["scenes"]["descending"]
mu = next(m for m in R["E1_models_ascending"] if m["model"] == "muhleman")
fm = R["fitted_model"]
sens = R["E4_sensitivity"]
mac = {
    "rAsc": f(asc["pearson_r"]), "rhoAsc": f(asc["spearman_rho"]), "ssimAsc": f(asc["ssim"]),
    "rmseAsc": f(asc["rmse_db_bias_removed"], 1), "rDesc": f(desc["pearson_r"]), "rhoDesc": f(desc["spearman_rho"]),
    "ssimDesc": f(desc["ssim"]), "rmseDesc": f(desc["rmse_db_bias_removed"], 1),
    "rMuAsc": f(mu["pearson_r"]), "ssimMuAsc": f(mu["ssim"]), "levelMu": f(mu["level_offset_db"], 1),
    "nFit": f(fm["n"]), "kFitDb": f(fm["k_db"], 1), "nRefitDesc": f(R["E2_descending"]["refit_on_descending"]["n"]),
    "rDescMu": f(R["E2_descending"]["muhleman"]["pearson_r"]),
    "shiftAscL": f(asc["shift_lines"], 1), "shiftAscP": f(asc["shift_pixels"], 1),
    "shiftDescL": f(desc["shift_lines"], 1), "shiftDescP": f(desc["shift_pixels"], 1),
    "geoDtMax": f(max(gA["dt_azimuth_ms_max"], gD["dt_azimuth_ms_max"]), 3),
    "geoDlineMax": f(max(gA["dline_equiv_max"], gD["dline_equiv_max"]), 3),
    "geoDpixMax": f(max(gA["dpixel_max"], gD["dpixel_max"]), 3),
    "geoDRmax": f"{max(gA['dslant_range_m_max'], gD['dslant_range_m_max'])*1e3:.2f}",
    "nPix": f"{asc['n_pixels']/1e6:.1f}", "nGrid": str(gA["n_points"]),
    "sceneAsc": sA["id"].replace("_", r"\_"), "sceneDesc": sD["id"].replace("_", r"\_"),
    "dateAsc": sA["datetime"][:10], "dateDesc": sD["datetime"][:10],
    "platAsc": sA["platform"].replace("sentinel-", "Sentinel-").upper().replace("SENTINEL", "Sentinel"),
    "platDesc": sD["platform"].replace("sentinel-", "Sentinel-").upper().replace("SENTINEL", "Sentinel"),
    "incAsc": f(sA["incidence_mid_deg"], 1), "incDesc": f(sD["incidence_mid_deg"], 1),
    "geoidN": f(sA["geoid_N_m"], 1),
    "rNoGeoid": f(sens[1]["pearson_r"]), "shiftNoGeoidP": f(sens[1]["shift_pixels"], 1),
    "shiftNoGeoidL": f(sens[1]["shift_lines"], 1),
    "rNoOver": f(sens[2]["pearson_r"]), "holesNoOver": f(100 * sens[2]["hole_fraction"], 1),
    "holesBase": f(100 * sens[0]["hole_fraction"], 2), "rDemNinety": f(sens[3]["pearson_r"]),
    "repoURL": REPO_URL,
}
(OUT / "macros.json").write_text(json.dumps(mac, indent=1), encoding="utf-8")
with open(OUT / "macros.tex", "w", encoding="utf-8") as fh:
    for k, v in mac.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- tables ----
TEX_LABEL = {"const": "Constant", "cos": r"$\cos\theta_{loc}$", "lambert": r"$\cos^2\theta_{loc}$ (Lambert)",
             "muhleman": "Muhleman", "fitted": rf"Fitted $k\cos^{{{fm['n']:.2f}}}\theta_{{loc}}$"}
rows = "\n".join(
    f"{TEX_LABEL[m['model']]} & "
    f"{f(m['pearson_r'],3)} & {f(m['spearman_rho'],3)} & {f(m['ssim'],3)} & {f(m['rmse_db_bias_removed'])} & {f(m['level_offset_db'],1)} \\\\"
    for m in R["E1_models_ascending"])
(OUT / "tab_models.tex").write_text(
    "\\begin{tabular}{lrrrrr}\\toprule\n Backscatter model & Pearson $r$ & Spearman $\\rho$ & SSIM & RMSE [dB] & level offset [dB]\\\\\\midrule\n"
    + rows + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

rows = "\n".join(
    f"{s['case']} & {f(s['pearson_r'],3)} & {f(s['ssim'],3)} & {f(s['shift_lines'],1)} & {f(s['shift_pixels'],1)} & {f(100*s['hole_fraction'],2)} \\\\"
    for s in sens)
(OUT / "tab_sensitivity.tex").write_text(
    "\\begin{tabular}{lrrrrr}\\toprule\n Configuration & $r$ & SSIM & $\\Delta$line & $\\Delta$pixel & empty px [\\%]\\\\\\midrule\n"
    + rows + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

rows = "\n".join(
    f"{c['class'].replace('&', '\\&')} & {f(c['share_pct'],1)} & {f(c['mean_residual_db'],2)} & {f(c['std_residual_db'],2)} & {f(c['pearson_r'],2)} \\\\"
    for c in R["E5_landcover_ascending"])
(OUT / "tab_landcover.tex").write_text(
    "\\begin{tabular}{lrrrr}\\toprule\n WorldCover class & area [\\%] & mean residual [dB] & std [dB] & $r$\\\\\\midrule\n"
    + rows + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

cls = R["E3_real_sigma0_by_predicted_class"]
rows = "\n".join(
    f"{k} & {cls['ascending'][k]['n']/1e3:.0f}k & {f(cls['ascending'][k]['median_db'],1)} & {cls['descending'][k]['n']/1e3:.0f}k & {f(cls['descending'][k]['median_db'],1)} \\\\"
    for k in cls["ascending"])
(OUT / "tab_classes.tex").write_text(
    "\\begin{tabular}{lrrrr}\\toprule\n & \\multicolumn{2}{c}{ascending} & \\multicolumn{2}{c}{descending}\\\\\n"
    "Predicted class & pixels & median $\\sigma^0$ [dB] & pixels & median $\\sigma^0$ [dB]\\\\\\midrule\n"
    + rows + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

import segno  # noqa: E402

segno.make(REPO_URL, error="m").save(OUT / "qr_repo.png", scale=12, border=2)
print("macros + tables written to", OUT)
