"""Generate chapters 3-5 of the B.Tech. report from the article-style manuscript (report/main.tex),
so that the prose stays identical. Run once after editing the manuscript:
    uv run python report/btp/convert_from_article.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "main.tex").read_text(encoding="utf-8")


def between(a, b):
    i, j = src.index(a), src.index(b)
    return src[i + len(a):j]


def brace_block(s, i):
    """Return (content, end) of the balanced {...} starting at s[i] == '{'."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
    raise ValueError("unbalanced")


def short_captions(s):
    out, pos = [], 0
    while True:
        i = s.find(r"\caption{", pos)
        if i < 0:
            out.append(s[pos:])
            return "".join(out)
        body, end = brace_block(s, i + len(r"\caption"))
        first = re.split(r":\s|(?<=[a-z0-9\)\}])\.\s", body, maxsplit=1)[0].rstrip(".")
        first = re.sub(r"\\label\{[^}]*\}", "", first)
        first = re.sub(r"\s*\(\\textbf\{[^}]*\}\)", "", first).strip()
        if "[" in first or "]" in first:
            first = first.replace("[", "{[").replace("]", "]}")
        out.append(s[pos:i] + r"\caption[" + first + "]{" + body + "}")
        pos = end


def convert(text, label):
    text = text.replace(r"\subsubsection{", r"\subsection{").replace(r"\subsection{", r"\section{")
    text = text.replace(r"\figw", r"\textwidth")
    text = text.replace(r"\input{tikz/", r"\input{../tikz/")
    text = text.replace(r"\input{generated/", r"\input{../generated/")
    text = text.replace(r"\input{../tikz/design.tex}", r"\resizebox{\textwidth}{!}{\input{../tikz/design.tex}}")
    text = text.replace(r"\input{../tikz/dataflow.tex}", r"\resizebox{\textwidth}{!}{\input{../tikz/dataflow.tex}}")
    text = text.replace(r"\section{Comparison with Sentinel-1 and Choice of the Backscatter Law}",
                        r"\section{Comparison with Sentinel-1 and the Backscatter Law}")
    text = text.replace(r"{../figures/intro/sar_distortions.pdf}", r"{sar_distortions.pdf}")
    # article-style cross references to chapters
    text = text.replace(r"Section~\ref{sec:data}", r"Chapter~\ref{ch:data}")
    text = text.replace(r"Section~\ref{sec:method}", r"Chapter~\ref{ch:method}")
    text = text.replace(r"Section~\ref{sec:results}", r"Chapter~\ref{ch:results}")
    text = text.replace(r"Section~\ref{sec:conclusions}", r"Chapter~\ref{ch:conclusions}")
    text = text.replace("Figure~", "Fig.~").replace("Figures~", "Figs.~")
    return short_captions(text.strip())


data = between(r"\label{sec:data}", "% ================================================================= 3")
method = between(r"\label{sec:method}", "% ================================================================= 4")
results = between(r"\label{sec:results}", "% ================================================================= 5")

(HERE / "chapters" / "03_data.tex").write_text(
    "\\chapter{Study Area and Data}\n\\label{ch:data}\n" + convert(data, "data") + "\n", encoding="utf-8")
FINAL = r"""

\section{Finalized Design}
\label{sec:final-design}
The final processing chain is the result of nine development iterations, summarised in Fig.~\ref{fig:design}. The
verification of the raw-signal simulator on a point target, on the synthetic aperture and on distributed and periodic
surfaces (iterations 1 to 3), and its comparison with the image-domain simulator on a real DEM patch (iteration 4), led
to the adoption of the image-domain simulator for scene-scale simulations. The four iterations that failed their test
fixed the remaining design decisions: the slant-to-ground-range polynomials are applied piece-wise rather than
interpolated (iteration 5), the DEM is oversampled three times so that every GRD pixel receives facets (iteration 6),
terrain heights are converted to ellipsoidal heights with the EGM2008 geoid (iteration 7), and the facet geometry is
computed block-wise in single precision so that a full scene fits in memory (iteration 8). The backscatter law,
finally, is fitted to the real image and cross-validated on the second viewing geometry (iteration 9). The resulting
data flow is the one shown in Fig.~\ref{fig:dataflow}.
"""
(HERE / "chapters" / "04_methodology.tex").write_text(
    "\\chapter{Methodology}\n\\label{ch:method}\n" + convert(method, "method") + FINAL, encoding="utf-8")
(HERE / "chapters" / "05_results.tex").write_text(
    "\\chapter{Results and Discussion}\n\\label{ch:results}\n" + convert(results, "results") + "\n", encoding="utf-8")
print("chapters 3-5 written")
