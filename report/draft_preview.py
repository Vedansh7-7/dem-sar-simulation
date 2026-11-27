"""Draft previews of the report (PDF) and poster (HTML) BEFORE the pipeline has finished:
every number that comes from results/metrics.json is shown as a red [pending] marker."""
import re
import subprocess
from pathlib import Path

import segno

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
tex = (ROOT / "report" / "main.tex").read_text(encoding="utf-8")
names = sorted(set(re.findall(r"\\(rAsc|rhoAsc|ssimAsc|rmseAsc|rDesc|rhoDesc|ssimDesc|rmseDesc|rMuAsc|ssimMuAsc|levelMu|nFit|kFitDb|nRefitDesc|rDescMu|shift\w+|geo\w+|nPix|nGrid|scene\w+|date\w+|plat\w+|inc(?:Asc|Desc)|geoidN|rNoGeoid|rNoOver|holes\w+|rDemNinety|repoURL)\b", tex)))
pend = r"\textcolor{red}{[pending]}"
with open(GEN / "macros.tex", "w", encoding="utf-8") as f:
    for n in names:
        val = "https://github.com/Vedansh7-7/dem-sar-simulation" if n == "repoURL" else pend
        f.write(f"\\newcommand{{\\{n}}}{{{val}}}\n")
for t in ("tab_models", "tab_sensitivity", "tab_landcover", "tab_classes"):
    (GEN / f"{t}.tex").write_text("\\begin{tabular}{l}\\toprule \\textcolor{red}{[table pending: pipeline still running]}\\\\ \\bottomrule\\end{tabular}\n")
segno.make("https://github.com/Vedansh7-7/dem-sar-simulation", error="m").save(GEN / "qr_repo.png", scale=12, border=2)

# missing pipeline figures -> skip gracefully with a placeholder box
pipe = ROOT / "figures" / "pipeline"
draft = tex
for fig in re.findall(r"\\includegraphics\[[^\]]*\]\{(p\d\w*\.png)\}", tex):
    if not (pipe / fig).exists():
        draft = re.sub(r"\\includegraphics\[[^\]]*\]\{" + re.escape(fig) + r"\}",
                       r"\\fbox{\\parbox[c][4cm][c]{0.9\\textwidth}{\\centering\\textcolor{red}{[figure "
                       + fig.replace("_", r"\_") + r" pending]}}}", draft)
draft = draft.replace(r"\begin{document}", r"\begin{document}\noindent\colorbox{red!15}{\parbox{\dimexpr\textwidth-2\fboxsep}{\centering\bfseries DRAFT PREVIEW --- numbers and pipeline figures pending}}")
(ROOT / "report" / "draft.tex").write_text(draft, encoding="utf-8")
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "draft.tex"], cwd=ROOT / "report", capture_output=True)

# poster draft
html = (ROOT / "poster" / "poster_template.html").read_text(encoding="utf-8")
fill = {"fig": "../figures", "qr": "../report/generated/qr_repo.png", "authors": "[Author name]",
        "affiliation": "[Affiliation]", "date_long": "Draft preview", "repoShort": "github.com/Vedansh7-7/dem-sar-simulation",
        "model_rows": "<tr><td colspan='4' style='color:#8c1c13'>[model table pending]</td></tr>"}
for k, v in fill.items():
    html = html.replace("{{" + k + "}}", v)
html = re.sub(r"\{\{\w+\}\}", "<span style='color:#8c1c13'>[…]</span>", html)
(ROOT / "poster" / "poster_draft.html").write_text(html, encoding="utf-8")
print("draft.pdf and poster_draft.html written")
