"""Fill the newspaper-style poster template with the pipeline results and render it
(A1 portrait) to PDF and PNG with headless Chrome.

Usage: uv run python poster/build_poster.py ["Author One, Author Two"] ["Affiliation"]
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "poster"
R = json.loads((ROOT / "results" / "metrics.json").read_text())
M = json.loads((ROOT / "report" / "generated" / "macros.json").read_text())
CHROME = next(p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") if Path(p).exists())

authors = sys.argv[1] if len(sys.argv) > 1 else "[Author name]"
affiliation = sys.argv[2] if len(sys.argv) > 2 else "[Affiliation]"

# model table rows (best value per column in bold via class)
rows = R["E1_models_ascending"]
best_r = max(rows, key=lambda r: r["pearson_r"])["model"]
html_rows = []
for r in rows:
    lab = r["label"].replace("^", "").replace("cos", "cos").replace("θ", "θ")
    if r["model"] == "fitted":
        lab = f"Fitted cos<sup>{R['fitted_model']['n']:.2f}</sup>θ"
    if r["model"] == "lambert":
        lab = "cos²θ (Lambert)"
    cls = ' class="best"' if r["model"] == best_r else ""
    html_rows.append(f"<tr{cls}><td>{lab}</td><td class='n'>{r['pearson_r']:.2f}</td>"
                     f"<td class='n'>{r['ssim']:.2f}</td><td class='n'>{r['rmse_db_bias_removed']:.1f}</td></tr>")

cls = R["E3_real_sigma0_by_predicted_class"]["ascending"]
class_sentence = (f"Real σ⁰ inside predicted shadow has a median of {cls['shadow']['median_db']:.1f} dB, "
                  f"inside predicted layover {cls['layover']['median_db']:.1f} dB.")
lc = R["E5_landcover_ascending"]
hi = max(lc, key=lambda c: c["mean_residual_db"])
lo = min(lc, key=lambda c: c["mean_residual_db"])
lc_sentence = (f"{hi['class']} is {hi['mean_residual_db']:+.1f} dB brighter and {lo['class'].lower()} "
               f"{lo['mean_residual_db']:+.1f} dB darker than the DEM predicts.")

repo = M["repoURL"]
subs = {**M, "authors": authors, "affiliation": affiliation, "model_rows": "\n".join(html_rows),
        "class_sentence": class_sentence, "lc_sentence": lc_sentence,
        "fig": "../figures", "qr": "../report/generated/qr_repo.png",
        "repoShort": repo.replace("https://", ""),
        "date_long": date.today().strftime("%A, %B %d, %Y").replace(" 0", " ")}
html = (P / "poster_template.html").read_text(encoding="utf-8")
for k, v in subs.items():
    html = html.replace("{{" + k + "}}", str(v))
assert "{{" not in html, [s for s in html.split("{{")[1:]][:3]
out = P / "poster.html"
out.write_text(html, encoding="utf-8")

url = out.resolve().as_uri()
import tempfile  # noqa: E402

profile = tempfile.mkdtemp(prefix="poster-chrome-")   # separate profile: don't hand off to a running Chrome
common = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
          f"--user-data-dir={profile}",
          "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw"]
subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={P / 'poster.pdf'}", url], check=True,
               capture_output=True)
# PNG from the PDF (Chrome's --screenshot can hang on large pages)
import pymupdf  # noqa: E402

pymupdf.open(P / "poster.pdf")[0].get_pixmap(dpi=110).save(P / "poster.png")
print("poster ->", P / "poster.pdf", P / "poster.png")
