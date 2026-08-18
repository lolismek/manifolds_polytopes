"""Intro diagram: belief geometry vs concept geometry, on a 6-color example.

Left: the belief view — one point per token, a probability tuple over the
6 colors; a confident-between-neighbors point on the ring and an uncertain
interior point. Right: the concept view — one vector per color, obtained
by averaging activations over that color's tokens; what shape do the 6
vectors form? Both read from the same residual stream (top).

Requires: tectonic, pdftoppm (same pipeline as method_diagram.py).
Writes results/blog/fig_belief_vs_concept.png and copies it to ~/Desktop.

Usage: python3 geometry_contrast_diagram.py
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "blog"
WORK = ROOT / "results" / "blog" / "_tex"

# hue order around the color wheel: red-orange-yellow-green-blue-purple
COLORS = [("red", "C0392B"), ("orange", "D68910"), ("yellow", "B7950B"),
          ("green", "1E8449"), ("blue", "2471A3"), ("purple", "7D3C98")]

ANG = [90 - 60 * i for i in range(6)]      # red on top, clockwise
R = 1.65


def belief_tuple(vals):
    """Probability tuple with each entry typeset in its state's color."""
    parts = [rf"\textcolor{{c{name}}}{{{v}}}"
             for (name, _), v in zip(COLORS, vals)]
    return r"$(" + ",\\,".join(parts) + r")$"


def left_panel():
    lines = [r"\begin{scope}[shift={(-5.2,-0.75)}]"]
    lines.append(r"\draw[gray!45, line width=0.6pt, dash pattern=on 2.2pt "
                 rf"off 2.6pt] (0,0) circle ({R}cm);")
    for (name, _), a in zip(COLORS, ANG):
        lines.append(
            rf"\node[circle, fill=c{name}, draw=c{name}!60!black, "
            rf"line width=0.5pt, minimum size=3.6mm, inner sep=0pt] "
            rf"at ({a}:{R}cm) {{}};")
        lines.append(
            rf"\node[font=\scriptsize, text=c{name}!85!black] "
            rf"at ({a}:{R + 0.42}cm) {{{name}}};")
    # belief A: on the arc between red and orange, only two nonzero coords
    lines.append(rf"\node[circle, fill=black, minimum size=2.4mm, "
                 rf"inner sep=0pt] (bA) at (60:{R}cm) {{}};")
    lines.append(r"\node[font=\scriptsize, anchor=west] at (1.35,2.3) "
                 rf"{{{belief_tuple(['0.5', '0.5', '0', '0', '0', '0'])}}};")
    lines.append(r"\draw[gray!60, line width=0.5pt] (1.42,2.18) -- "
                 r"($(bA)+(0.08,0.15)$);")
    # belief B: interior, spread over everything
    lines.append(r"\node[circle, fill=black, minimum size=2.4mm, "
                 r"inner sep=0pt] (bB) at (0,0.15) {};")
    lines.append(r"\node[font=\scriptsize, anchor=north] at (0,-0.12) "
                 rf"{{{belief_tuple(
                     ['0.3', '0.1', '0.1', '0.2', '0.1', '0.2'])}}};")
    # captions
    lines.append(r"\node[font=\footnotesize, align=center] at (0,-2.85) "
                 r"{a belief is a probability over the 6 colors;\\"
                 r"as tokens arrive, the point moves};")
    lines.append(r"\node[font=\footnotesize\itshape, gray!30!black] at "
                 r"(0,-3.75) {what shape does the set of beliefs trace?};")
    lines.append(r"\end{scope}")
    return "\n".join(lines)


def right_panel():
    lines = [r"\begin{scope}[shift={(5.2,-0.75)}]"]
    for (name, _), a in zip(COLORS, ANG):
        lines.append(
            rf"\draw[->, c{name}, line width=1.1pt] (0,0) -- ({a}:{R}cm) "
            rf"node[font=\scriptsize, text=c{name}!85!black, "
            rf"pos=1.26] {{$v_{{\mathrm{{{name}}}}}$}};")
    lines.append(r"\draw[gray!45, line width=0.6pt, dash pattern=on 2.2pt "
                 rf"off 2.6pt] (0,0) circle ({R}cm);")
    # sentences with red underlined, then the definition of v_red
    lines.append(r"\node[font=\footnotesize, align=left] at (0,-3.05) {"
                 + sentences() + r"};")
    lines.append(r"\node[font=\footnotesize] at (0,-4.0) "
                 r"{$v_{\mathrm{red}}$ = average activation at the "
                 r"\textcolor{cred}{\underline{red}} tokens};")
    lines.append(r"\node[font=\footnotesize\itshape, gray!30!black] at "
                 r"(0,-4.65) {what shape do the 6 vectors form?};")
    lines.append(r"\end{scope}")
    return "\n".join(lines)


def sentences():
    def red(w):
        return rf"\textcolor{{cred}}{{\underline{{\smash{{{w}}}}}}}"
    rows = [
        rf"``My favorite color is {red('red')}.''",
        rf"``Her lipstick was a deep {red('red')}.''",
        rf"``{red('Red')}, white and blue are nice colors.''",
    ]
    return (r"\begin{tabular}{@{}l@{}}" +
            r"\\[1.5pt]".join(rows) + r"\end{tabular}")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    color_defs = "\n".join(
        rf"\definecolor{{c{n}}}{{HTML}}{{{h}}}" for n, h in COLORS)
    tex = r"""\documentclass[border=10pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, calc}
""" + color_defs + r"""
\begin{document}
\begin{tikzpicture}[>={Stealth[length=4.5pt]}, line cap=round]

% ---- top: the shared residual stream ----
\node[draw=black, line width=0.8pt, font=\small, align=center,
      inner xsep=10pt, inner ysep=5pt] (rs) at (0, 4.0)
      {residual stream\\[-1pt]
       {\footnotesize\itshape\color{gray!30!black}
        one activation vector per token}};

% ---- arrows from the stream to the two readings ----
\draw[->, line width=0.7pt] (rs.west) -| (-5.2, 2.95);
\draw[->, line width=0.7pt] (rs.east) -| (5.2, 2.95);
\node[font=\itshape\footnotesize, anchor=south east] at (-2.55, 4.08)
      {read at one token};
\node[font=\itshape\footnotesize, anchor=south west] at (2.3, 4.08)
      {average each color's tokens};

% ---- headers ----
\node[font=\small\bfseries] at (-5.2, 2.6) {belief geometry};
\node[font=\itshape\footnotesize, gray!30!black] at (-5.2, 2.2)
      {one point = the model's guess at one token};
\node[font=\small\bfseries] at (5.2, 2.6) {concept geometry};
\node[font=\itshape\footnotesize, gray!30!black] at (5.2, 2.2)
      {one arrow = one color};

""" + left_panel() + "\n\n" + right_panel() + r"""

% ---- thought bubble for belief A, in the middle gap ----
\node[draw=gray!60, fill=gray!6, rounded corners=6pt, line width=0.5pt,
      font=\itshape\footnotesize, align=center, inner sep=5pt]
      (th) at (0.35, 0.45)
      {``the text is talking\\about something
       \textcolor{cred}{red}\,\ldots\\or maybe
       \textcolor{corange}{orange}''};
\fill[gray!55] (-3.5, 0.62) circle (1.0pt) (-2.9, 0.57) circle (1.4pt)
      (-2.3, 0.52) circle (1.8pt);

\end{tikzpicture}
\end{document}
"""
    (WORK / "belief_vs_concept.tex").write_text(tex)
    subprocess.run(["tectonic", "belief_vs_concept.tex"], cwd=WORK,
                   check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    "belief_vs_concept.pdf", "fig_belief_vs_concept"],
                   cwd=WORK, check=True)
    shutil.copy(WORK / "fig_belief_vs_concept.png",
                OUT / "fig_belief_vs_concept.png")
    shutil.copy(WORK / "fig_belief_vs_concept.png",
                Path.home() / "Desktop" / "fig_belief_vs_concept.png")
    print("wrote", OUT / "fig_belief_vs_concept.png",
          "and copied to Desktop")


if __name__ == "__main__":
    main()
