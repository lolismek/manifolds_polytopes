"""Intro diagram: belief state geometry vs concept geometry, 6-color example.

Left: the belief view — b = f_belief(h), a probability over the 6 colors;
a confident-between-neighbors point on the ring (vertical tuple) and an
uncertain interior point, plus a monocle reader thinking the belief out
loud. Right: the concept view — one vector per color, v_red = average h
over the tokens about red. Both read from the same residual stream (top).

Requires: tectonic, pdftoppm, Pillow (emoji). Same pipeline as
method_diagram.py. Writes results/blog/fig_belief_vs_concept.png and
copies it to ~/Desktop.

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


def emoji_png():
    """Render the monocle-face emoji to WORK/monocle.png (Apple bitmap)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc",
                              160)
    img = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((160, 160), "\U0001F9D0", font=font,
                             embedded_color=True, anchor="mm")
    img.crop(img.getbbox()).save(WORK / "monocle.png")


def belief_column(vals):
    """Vertical probability tuple, each entry in its state's color."""
    rows = "\\\\".join(rf"\textcolor{{c{name}}}{{{v}}}"
                       for (name, _), v in zip(COLORS, vals))
    return (r"$\left(\begin{smallmatrix}" + rows +
            r"\end{smallmatrix}\right)$")


def left_panel():
    lines = [r"\begin{scope}[shift={(-5.2,-0.85)}]"]
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
    lines.append(r"\node[font=\scriptsize, anchor=west] (tA) at (2.3,1.55) "
                 rf"{{{belief_column(['0.5', '0.5', '0', '0', '0', '0'])}}};")
    lines.append(r"\draw[gray!60, line width=0.5pt] (tA.west) -- "
                 r"($(bA)+(0.14,0.06)$);")
    # belief B: interior, spread over everything
    lines.append(r"\node[circle, fill=black, minimum size=2.4mm, "
                 r"inner sep=0pt] (bB) at (0.15,-0.15) {};")
    lines.append(r"\node[font=\scriptsize, anchor=east] (tB) at (-2.6,0.0) "
                 rf"{{{belief_column(
                     ['0.3', '0.1', '0.1', '0.2', '0.1', '0.2'])}}};")
    lines.append(r"\draw[gray!60, line width=0.5pt] (tB.east) -- "
                 r"($(bB)+(-0.13,0.0)$);")
    # monocle reader thinking the belief out loud, below the circle
    lines.append(r"\node[inner sep=0pt] at (-2.75,-3.05) "
                 r"{\includegraphics[width=8.5mm]{monocle.png}};")
    lines.append(r"\node[draw=gray!60, fill=gray!6, rounded corners=6pt, "
                 r"line width=0.5pt, font=\itshape\footnotesize, "
                 r"align=center, inner sep=5pt] (th) at (0.85,-3.05) "
                 r"{``the text is talking about something\\"
                 r"\textcolor{cred}{red}\,\ldots{} or maybe "
                 r"\textcolor{corange}{orange}''};")
    lines.append(r"\fill[gray!55] (-2.05,-2.95) circle (1.0pt) "
                 r"(-1.8,-2.98) circle (1.4pt);")
    lines.append(r"\end{scope}")
    return "\n".join(lines)


def right_panel():
    lines = [r"\begin{scope}[shift={(4.3,-0.85)}]"]
    for (name, _), a in zip(COLORS, ANG):
        lines.append(
            rf"\draw[->, c{name}, line width=1.1pt] (0,0) -- ({a}:{R}cm) "
            rf"node[font=\scriptsize, text=c{name}!85!black, "
            rf"pos=1.26] {{$v_{{\mathrm{{{name}}}}}$}};")
    lines.append(r"\draw[gray!45, line width=0.6pt, dash pattern=on 2.2pt "
                 rf"off 2.6pt] (0,0) circle ({R}cm);")
    lines.append(r"\node[font=\footnotesize] at (0,-2.55) "
                 r"{$v_{\mathrm{red}}$ = average $h$ over the tokens "
                 r"about \textcolor{cred}{red}};")
    lines.append(r"\end{scope}")
    return "\n".join(lines)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    emoji_png()
    color_defs = "\n".join(
        rf"\definecolor{{c{n}}}{{HTML}}{{{h}}}" for n, h in COLORS)
    tex = r"""\documentclass[border=10pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, calc}
""" + color_defs + r"""
\begin{document}
\begin{tikzpicture}[>={Stealth[length=4.5pt]}, line cap=round]

% ---- top: the shared residual stream ----
\node[draw=black, line width=0.8pt, font=\small, inner xsep=12pt,
      inner ysep=6pt] (rs) at (0, 4.0) {residual stream $h$};

\draw[->, line width=0.7pt] (rs.west) -| (-5.2, 2.95);
\draw[->, line width=0.7pt] (rs.east) -| (4.3, 2.95);

% ---- headers ----
\node[font=\small\bfseries] at (-5.2, 2.6) {belief state geometry};
\node[font=\footnotesize] at (-5.2, 2.15)
      {$b = f_{\mathrm{belief}}(h) \in \mathbb{R}^6$};
\node[font=\itshape\footnotesize, gray!30!black] at (-5.2, 1.78)
      {entries sum to 1: a probability over the 6 colors};
\node[font=\small\bfseries] at (4.3, 2.6) {concept geometry};
\node[font=\itshape\footnotesize, gray!30!black] at (4.3, 2.15)
      {one arrow = one color};

""" + left_panel() + "\n\n" + right_panel() + r"""

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
