"""Canva import package: poster PDF/PNG (Canva: Upload -> import PDF as an editable design),
every figure used on the poster as a separate PNG, and all poster text in TEXT.md."""
import html as htmlmod
import re
import shutil
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent
OUT = P / "canva"
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)

page = (P / "poster.html").read_text(encoding="utf-8")
for f in ("poster.pdf", "poster.png"):
    if (P / f).exists():
        shutil.copy(P / f, OUT / f)
for src in re.findall(r'src="([^"]+\.png)"', page):
    path = (P / src).resolve()
    if path.exists():
        shutil.copy(path, OUT / "figures" / path.name)

body = page.split("<body>", 1)[1]
body = re.sub(r"<(style|script)[\s\S]*?</\1>", "", body)
body = re.sub(r"<img[^>]*alt=\"([^\"]*)\"[^>]*>", r"\n[IMAGE: \1]\n", body)
body = re.sub(r"<h[1-4][^>]*>", "\n\n## ", body)
body = re.sub(r"</h[1-4]>", "\n", body)
body = re.sub(r"<(p|li|tr|figcaption|div class=\"stat\")[^>]*>", "\n", body)
body = re.sub(r"<li[^>]*>", "\n- ", body)
body = re.sub(r"<[^>]+>", " ", body)
body = htmlmod.unescape(body)
lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in body.splitlines()]
text = "\n".join(ln for ln in lines if ln)
(OUT / "TEXT.md").write_text(
    "# Poster text for Canva\n\nFonts: Playfair Display 900 (masthead), Oswald 600-700 (headlines), "
    "Source Serif 4 (body). Paper colour #efe8d8, ink #1d1b18, accent #1f3a5f. Size A1 portrait "
    "(594 x 841 mm).\n\n" + text + "\n", encoding="utf-8")
(OUT / "HOW_TO_IMPORT.txt").write_text(
    "Canva -> Create a design -> Import file (or Uploads -> Upload files) -> choose poster.pdf.\n"
    "Canva converts the PDF into an editable A1 design: text boxes stay editable, images become\n"
    "separate elements. If a figure looks soft, replace it with the PNG of the same name from\n"
    "figures/. All text is also in TEXT.md.\n", encoding="utf-8")
print("Canva package ->", OUT, len(list((OUT / "figures").iterdir())), "figures")
