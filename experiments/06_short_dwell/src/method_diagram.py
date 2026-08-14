"""Method figure: the ring Markov chain (TikZ automaton with transition
probabilities and the T_ij formula) above a real corpus excerpt whose tokens
are colored by the true state path.

The excerpt is document 0 of the s=1.5, p_stay=0.95 pilot; its state path is
reconstructed exactly from the pilot's deterministic RNG (verified against the
recorded transition indices in s15p95_sample.json).

Requires: tectonic, pdftoppm, transformers (tokenizer only).
Writes results/blog/fig_method_ring.png and copies it to ~/Desktop.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "blog"
WORK = ROOT / "results" / "blog" / "_tex"

K = 8
TAB10 = ["1F77B4", "FF7F0E", "2CA02C", "D62728",
         "9467BD", "8C564B", "E377C2", "7F7F7F"]


def state_path():
    """Regenerate doc 0's z path with the pilot's exact RNG; verify."""
    rng = np.random.default_rng([606, 950])
    n, length, p_stay = 64, 1011, 0.95
    z = np.empty((n, length), dtype=np.int8)
    z[:, 0] = rng.integers(0, K, n)
    u = rng.random((n, length - 1))
    step = rng.integers(0, 2, (n, length - 1)).astype(np.int8) * 2 - 1
    for t in range(1, length):
        move = u[:, t - 1] >= p_stay
        z[:, t] = np.where(move, (z[:, t - 1] + step[:, t - 1]) % K,
                           z[:, t - 1])
    rec = json.load(open(ROOT / "results/pilot/s15p95_sample.json"))
    trans = ((np.diff(z[0]) != 0).nonzero()[0] + 1).tolist()
    assert int(z[0, 0]) == rec["z0"] and trans == rec["transitions"]
    return z[0], rec["text"]


def tex_escape(s):
    s = (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
         .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
         .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
         .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))
    return (s.replace("“", "``").replace("”", "''")
            .replace("‘", "`").replace("’", "'")
            .replace("–", "--").replace("—", "---"))


def excerpt_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")
    z, text = state_path()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    toks = [tok.decode([i]) for i in ids]
    start = next(j for j in range(len(toks) - 1)
                 if "In survival" in toks[j] + toks[j + 1])
    acc, cut = "", None
    for j in range(128, 160):          # cut at end of 'imperative aim".'
        acc += toks[j]
        if "aim”." in acc:
            cut = j + 1
            break
    # text token j: j < 12 is the opener, else state z[j-12]
    return [(toks[j].replace("\n", " "), int(z[j - 12]))
            for j in range(start, cut)]


def chip(s):
    return (r"\tikz[baseline=-3.2pt]{\node[circle, fill=state%d, text=white, "
            r"inner sep=1.1pt, minimum size=8.5pt, "
            r"font=\tiny\bfseries]{%d};}" % (s, s))


def strip_tex(pairs):
    parts, prev = [], None
    for t, s in pairs:
        col = "gray!16" if s < 0 else f"state{s}!22"
        if s != prev and s >= 0:
            parts.append(chip(s) + r"\hspace{1pt}")
        parts.append(r"\colorbox{%s}{\strut{}%s}\allowbreak" % (col, tex_escape(t)))
        prev = s
    return "\n".join(parts)


def ring_tex():
    lines = [r"\begin{tikzpicture}[>={Stealth[length=4pt]}, line cap=round]"]
    r = 1.72
    for i in range(K):
        a = 90 - 45 * i
        lines.append(
            rf"\node[circle, fill=state{i}, draw=state{i}!60!black, "
            rf"line width=0.5pt, text=white, font=\footnotesize\bfseries, "
            rf"minimum size=6.4mm, inner sep=0pt] (n{i}) at ({a}:{r}cm) {{{i}}};")
    for i in range(K):                       # paired directional arcs
        j = (i + 1) % K
        lines.append(rf"\draw[->, gray!55!black, line width=0.55pt] "
                     rf"(n{i}) to[bend left=14] (n{j});")
        lines.append(rf"\draw[->, gray!55!black, line width=0.55pt] "
                     rf"(n{j}) to[bend left=14] (n{i});")
    for i in range(K):                       # radial self-loops
        a = 90 - 45 * i
        lines.append(
            rf"\path[->, gray!55!black, line width=0.55pt] (n{i}) "
            rf"edge[out={a + 24}, in={a - 24}, looseness=5.2, "
            rf"min distance=5.2mm] (n{i});")
    lines.append(r"\node[font=\scriptsize, gray!30!black] at (90:2.72cm) {$0.95$};")
    lines.append(r"\node[font=\scriptsize, gray!30!black] at (67.5:2.32cm) {$0.025$};")
    lines.append(r"\node[font=\small] at (0, 0) {$z_t$};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    color_defs = "\n".join(
        rf"\definecolor{{state{i}}}{{HTML}}{{{c}}}" for i, c in enumerate(TAB10))
    tex = r"""\documentclass[border=8pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\setlength{\fboxsep}{0pt}
""" + color_defs + r"""
\begin{document}
\begin{minipage}{14.2cm}
\centering
\begin{minipage}[c]{6.6cm}
\centering
""" + ring_tex() + r"""
\end{minipage}%
\begin{minipage}[c]{7.4cm}
\centering
$T_{ij} \;=\; \begin{cases}
  p_{\text{stay}} & j = i \\[3pt]
  \tfrac{1}{2}\,(1 - p_{\text{stay}}) & j = i \pm 1 \ (\mathrm{mod}\ 8) \\[3pt]
  0 & \text{otherwise}
\end{cases}$

\vspace{7pt}
{\small $p_{\text{stay}} = 0.95$ \; $\Rightarrow$ \; mean dwell $\approx 20$ tokens}
\end{minipage}

\vspace{5mm}
\begin{minipage}{14.2cm}
\raggedright\small\setlength{\parindent}{0pt}%
{\itshape\footnotesize A corpus excerpt, tokens colored by the active state $z_t$:}\\[3.5pt]
\ldots\,%
""" + strip_tex(excerpt_tokens()) + r"\,\ldots" + r"""
\end{minipage}
\end{minipage}
\end{document}
"""
    (WORK / "method_ring.tex").write_text(tex)
    subprocess.run(["tectonic", "method_ring.tex"], cwd=WORK, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    "method_ring.pdf", "fig_method_ring"], cwd=WORK, check=True)
    shutil.copy(WORK / "fig_method_ring.png", OUT / "fig_method_ring.png")
    shutil.copy(WORK / "fig_method_ring.png",
                Path.home() / "Desktop" / "fig_method_ring.png")
    print("wrote", OUT / "fig_method_ring.png", "and copied to Desktop")


if __name__ == "__main__":
    main()
