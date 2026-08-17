"""Method figure for control2 (exp07): the uniform-jump Markov chain as a
complete graph (TikZ automaton with transition probabilities and the T_ij
formula) above a real corpus excerpt whose tokens are colored by the true
state path — the exp06 fig_method_ring companion.

The excerpt is document 0 of the uniform pilot; its state path comes
straight from the recorded segments in uniform_sample.json (no RNG
reconstruction needed). The visible jumps are 2->4, 4->0, 0->2, 2->5 —
none of them ring neighbors.

Requires: tectonic, pdftoppm, transformers (tokenizer only).
Writes results/blog/fig_method_uniform.png and copies it to ~/Desktop.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "blog"
WORK = OUT / "_tex"

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
from method_diagram import K, strip_tex, TAB10  # noqa: E402

G_START, G_END = -12, 45     # generated-token window (opener = -12..-1, gray)


def excerpt_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")
    rec = json.load(open(ROOT / "results/pilot/uniform_sample.json"))
    ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
    toks = [tok.decode([i]) for i in ids]
    bounds = rec["transitions"] + [10 ** 9]

    def state(g):
        if g < 0:
            return -1                      # opener tokens: uncolored
        for k, b in enumerate(bounds):
            if g < b:
                return rec["states"][k]

    return [(toks[g + 12].replace("\n", " "), state(g))
            for g in range(G_START, G_END)]


def complete_tex():
    lines = [r"\begin{tikzpicture}[>={Stealth[length=3.5pt]}, line cap=round]"]
    r = 1.8
    for i in range(K):                       # chords first, nodes on top
        for j in range(i + 1, K):
            ai, aj = 90 - 45 * i, 90 - 45 * j
            lines.append(rf"\draw[<->, gray!50, line width=0.45pt, "
                         rf"shorten >=9.5pt, shorten <=9.5pt] "
                         rf"({ai}:{r}cm) -- ({aj}:{r}cm);")
    for i in range(K):
        a = 90 - 45 * i
        lines.append(
            rf"\node[circle, fill=state{i}, draw=state{i}!60!black, "
            rf"line width=0.5pt, text=white, font=\footnotesize\bfseries, "
            rf"minimum size=6.4mm, inner sep=0pt] (n{i}) at ({a}:{r}cm) {{{i}}};")
    for i in range(K):                       # radial self-loops
        a = 90 - 45 * i
        lines.append(
            rf"\path[->, gray!55!black, line width=0.55pt] (n{i}) "
            rf"edge[out={a + 24}, in={a - 24}, looseness=5.2, "
            rf"min distance=5.2mm] (n{i});")
    lines.append(r"\node[font=\scriptsize, gray!30!black] at (90:2.8cm) {$0.95$};")
    lines.append(r"\node[font=\scriptsize, gray!30!black, fill=white, "
                 r"inner sep=1pt] at (67.5:1.62cm) {$0.0071$};")
    lines.append(r"\node[font=\small, fill=white, inner sep=1.6pt] "
                 r"at (0, 0) {$z_t$};")
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
""" + complete_tex() + r"""
\end{minipage}%
\begin{minipage}[c]{7.4cm}
\centering
$T_{ij} \;=\; \begin{cases}
  p_{\text{stay}} & j = i \\[3pt]
  \tfrac{1}{7}\,(1 - p_{\text{stay}}) & j \neq i
\end{cases}$

\vspace{7pt}
{\small $p_{\text{stay}} = 0.95$ \; $\Rightarrow$ \; same dwell statistics
as the ring,}

{\small but all 7 jump destinations equally likely}
\end{minipage}

\vspace{5mm}
\begin{minipage}{14.2cm}
\raggedright\small\setlength{\parindent}{0pt}%
{\itshape\footnotesize A control2 corpus excerpt, tokens colored by the
active state $z_t$ (opener in gray):}\\[3.5pt]
""" + strip_tex(excerpt_tokens()) + r"\,\ldots" + r"""
\end{minipage}
\end{minipage}
\end{document}
"""
    (WORK / "method_uniform.tex").write_text(tex)
    subprocess.run(["tectonic", "method_uniform.tex"], cwd=WORK, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    "method_uniform.pdf", "fig_method_uniform"],
                   cwd=WORK, check=True)
    shutil.copy(WORK / "fig_method_uniform.png", OUT / "fig_method_uniform.png")
    shutil.copy(WORK / "fig_method_uniform.png",
                Path.home() / "Desktop" / "fig_method_uniform.png")
    print("wrote", OUT / "fig_method_uniform.png", "and copied to Desktop")


if __name__ == "__main__":
    main()
