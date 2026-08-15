"""Full method-pipeline figure, three panels:

  (a) a generic (non-ring) Markov chain over K states, the current state
      highlighted, plus the pre-sampled state path z_{1:T} as a colored strip;
  (b) the teacher transformer stack with the frozen SAE decoder — z_t selects
      direction d_{z_t}, injected into the residual stream across the steered
      layer band, with the autoregressive loop;
  (c) the real corpus excerpt (doc 0 of the s=1.5, p_stay=0.95 pilot), tokens
      colored by the true state path.

The excerpt path visits states 3 -> 4 -> 3 -> 2, all of which exist in the
schematic 5-state chain, so state colors are consistent across panels.

Requires: tectonic, pdftoppm, transformers (tokenizer only).
Writes results/blog/fig_method_pipeline.png and copies it to ~/Desktop.
"""
import shutil
import subprocess
from pathlib import Path

from method_diagram import TAB10, excerpt_tokens, strip_tex

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "blog"
WORK = ROOT / "results" / "blog" / "_tex"

GRAY = "gray!55!black"


def chain_panel():
    """5-state irregular chain + pre-sampled path strip."""
    pos = {0: (1.1, 5.9), 1: (3.3, 6.2), 2: (4.35, 4.5),
           3: (2.9, 3.2), 4: (0.6, 3.9)}
    cx = sum(p[0] for p in pos.values()) / 5
    cy = sum(p[1] for p in pos.values()) / 5
    L = []
    # nodes (active state 3 gets a dark halo)
    for i, (x, y) in pos.items():
        extra = ", line width=1.1pt, draw=black!75" if i == 3 else ""
        L.append(
            rf"\node[circle, fill=state{i}, draw=state{i}!60!black, "
            rf"line width=0.5pt, text=white, font=\footnotesize\bfseries, "
            rf"minimum size=6.2mm, inner sep=0pt{extra}] (n{i}) "
            rf"at ({x}, {y}) {{{i}}};")
    # directed edges: clearly not a ring (asymmetric + a chord)
    E = [(0, 1, 0), (1, 0, 14), (1, 2, 0), (2, 1, 14), (3, 4, 14),
         (4, 3, 14), (3, 2, 0), (0, 4, 0), (1, 3, 0)]
    for i, j, bend in E:
        b = f"[bend left={bend}]" if bend else ""
        L.append(rf"\draw[->, {GRAY}, line width=0.55pt] (n{i}) to{b} (n{j});")
    # radial self-loops
    import math
    for i, (x, y) in pos.items():
        a = math.degrees(math.atan2(y - cy, x - cx))
        L.append(
            rf"\path[->, {GRAY}, line width=0.55pt] (n{i}) "
            rf"edge[out={a + 24:.0f}, in={a - 24:.0f}, looseness=5.2, "
            rf"min distance=5mm] (n{i});")
    # probabilities around the active state (sum to 1)
    L.append(rf"\node[font=\tiny, gray!30!black] at (2.30, 2.30) {{$0.95$}};")
    L.append(rf"\node[font=\tiny, gray!30!black] at (1.55, 3.20) {{$0.03$}};")
    L.append(rf"\node[font=\tiny, gray!30!black] at (3.78, 3.58) {{$0.02$}};")
    return "\n".join(L)


def path_strip(segs, x0=0.35, x1=4.75, y=1.28, h=0.30):
    """Pre-sampled path z_{1:T} as a dwell-colored strip."""
    total = sum(n for _, n in segs)
    L = [rf"\node[font=\scriptsize, gray!25!black, anchor=west] "
         rf"at ({x0 - 0.05}, {y + h + 0.32}) "
         rf"{{the whole path $z_{{1:T}}$ is sampled before any text:}};"]
    x = x0
    for s, n in segs:
        w = (x1 - x0) * n / total
        L.append(rf"\fill[state{s}] ({x:.2f}, {y}) "
                 rf"rectangle ({x + w - 0.04:.2f}, {y + h});")
        if w > 0.55:
            L.append(rf"\node[font=\tiny\bfseries, white] at "
                     rf"({x + w / 2 - 0.02:.2f}, {y + h / 2}) {{{s}}};")
        x += w
    L.append(rf"\draw[->, {GRAY}, line width=0.5pt] "
             rf"({x0}, {y - 0.22}) -- ({x1}, {y - 0.22}) "
             rf"node[font=\tiny, anchor=north east, inner sep=1.5pt] "
             rf"{{token position $t$}};")
    return "\n".join(L)


def teacher_panel():
    """Transformer stack + SAE decoder + injection band + sampling loop."""
    L = []
    SX, BW, BH = 11.8, 2.7, 0.52          # stack center x, box width/height

    def lbox(yc, text, name):
        L.append(
            rf"\node[draw=black!60, fill=white, rounded corners=1.2pt, "
            rf"line width=0.6pt, minimum width={BW}cm, "
            rf"minimum height={BH}cm, font=\scriptsize, inner sep=1pt] "
            rf"({name}) at ({SX}, {yc}) {{{text}}};")

    # injection band background first (behind the layer boxes)
    L.append(rf"\fill[state3!9, rounded corners=2pt] "
             rf"(10.28, 1.86) rectangle (13.32, 4.95);")
    L.append(rf"\draw[state3!55, line width=0.6pt, rounded corners=2pt] "
             rf"(10.28, 1.86) rectangle (13.32, 4.95);")
    L.append(rf"\node[font=\tiny, state3!75!black, rotate=90] "
             rf"at (13.55, 3.40) {{steered layers $12$--$23$}};")

    lbox(1.28, r"embed $+$ layers $0$--$11$", "lo")
    lbox(2.28, r"layer $12$", "l12")
    L.append(rf"\node[font=\scriptsize, black!60] at ({SX}, 3.28) {{$\vdots$}};")
    lbox(4.28, r"layer $23$", "l23")
    lbox(5.52, r"layers $24$--$25$", "hi")
    # residual stream
    for y0, y1 in [(0.62, 1.02), (1.54, 2.02), (2.54, 3.02),
                   (3.54, 4.02), (4.54, 5.26), (5.78, 6.18)]:
        L.append(rf"\draw[->, black!70, line width=0.7pt] "
                 rf"({SX}, {y0}) -- ({SX}, {y1});")
    L.append(rf"\node[font=\scriptsize, anchor=north] at ({SX}, 0.58) "
             rf"{{text so far $x_{{<t}}$}};")
    L.append(rf"\node[font=\scriptsize, anchor=south] (samp) at ({SX}, 6.20) "
             rf"{{sample next token $x_t$}};")

    # SAE decoder box with a fan of K unit directions
    L.append(rf"\node[draw={GRAY}, fill=white, rounded corners=2pt, "
             rf"line width=0.6pt, minimum width=2.75cm, minimum height=2.5cm, "
             rf"anchor=south west] (sae) at (5.65, 2.45) {{}};")
    L.append(rf"\node[font=\scriptsize, anchor=north] at (7.03, 4.83) "
             rf"{{frozen SAE decoder}};")
    L.append(rf"\node[font=\tiny, gray!25!black, anchor=north] "
             rf"at (7.03, 4.48) {{$K$ unit directions}};")
    import math
    ox, oy = 6.95, 2.85
    for i, ang in enumerate([155, 120, 85, 50, 15]):
        r = 1.15 if i == 3 else 0.95
        lw = "1.4pt" if i == 3 else "0.8pt"
        tx = ox + (r + 0.28) * math.cos(math.radians(ang))
        ty = oy + (r + 0.28) * math.sin(math.radians(ang))
        L.append(rf"\draw[->, state{i}, line width={lw}] ({ox}, {oy}) -- "
                 rf"++({ang}:{r});")
        L.append(rf"\node[font=\tiny, state{i}!75!black] at "
                 rf"({tx:.2f}, {ty:.2f}) {{$d_{i}$}};")
    # injection arrow: active direction -> band, with the exact formula
    L.append(rf"\node[circle, draw=state3!75!black, fill=white, "
             rf"line width=0.7pt, inner sep=0pt, minimum size=8pt, "
             rf"font=\scriptsize] (plus) at (10.28, 3.4) {{$+$}};")
    L.append(rf"\draw[->, state3!75!black, line width=0.9pt] "
             rf"(8.55, 3.4) -- (plus);")
    L.append(
        rf"\node[font=\scriptsize, anchor=south] at (9.42, 3.55) "
        rf"{{$s\,a_{{z_t}}\frac{{\rho_\ell}}{{\rho_{{12}}}}\,d_{{z_t}}$}};")
    L.append(rf"\node[font=\tiny, gray!25!black, anchor=north, "
             rf"align=center] at (9.42, 3.28) {{added at every\\layer in the band}};")

    # autoregressive loop on the right
    L.append(
        rf"\draw[->, black!60, line width=0.7pt, rounded corners=6pt] "
        rf"(samp.east) -| (14.85, 0.85) -- (13.25, 0.85);")
    L.append(
        rf"\node[font=\tiny, gray!20!black, anchor=west, align=left] "
        rf"at (15.02, 3.5) {{append $x_t$,\\ move to $z_{{t+1}}$,\\ "
        rf"rebuild the KV cache\\ \textit{{(steering never}}\\ "
        rf"\textit{{touches the visible}}\\ \textit{{context)}}}};")
    return "\n".join(L)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    pairs = excerpt_tokens()
    segs = []
    for _, s in pairs:
        if segs and segs[-1][0] == s:
            segs[-1][1] += 1
        else:
            segs.append([s, 1])
    assert all(s <= 4 for s, _ in segs), segs

    color_defs = "\n".join(
        rf"\definecolor{{state{i}}}{{HTML}}{{{c}}}" for i, c in enumerate(TAB10))
    row1 = "\n".join([
        r"\begin{tikzpicture}[>={Stealth[length=4pt]}, line cap=round]",
        r"\useasboundingbox (-0.1, 0.1) rectangle (17.35, 8.35);",
        # panel (a) header
        r"\node[font=\small\bfseries, anchor=west] at (0.0, 7.98) "
        r"{(a)\; plant a latent variable};",
        r"\node[font=\scriptsize, gray!25!black, anchor=west] at (0.0, 7.56) "
        r"{any Markov chain over $K$ states (ours: a ring, $K{=}8$)};",
        chain_panel(),
        path_strip(segs),
        # panel (b) header
        r"\node[font=\small\bfseries, anchor=west] at (7.0, 7.98) "
        r"{(b)\; steer a teacher LLM while it writes};",
        r"\node[font=\scriptsize, gray!25!black, anchor=west] at (7.0, 7.56) "
        r"{the current state picks the direction added to the residual "
        r"stream};",
        # (a) -> (b) selection arrow
        rf"\draw[->, dashed, {GRAY}, line width=0.7pt] "
        rf"(n3.east) to[bend right=12] (5.65, 3.9);",
        rf"\node[font=\scriptsize, gray!25!black, anchor=north, align=center] "
        rf"at (4.75, 3.15) {{$z_t$ selects\\ direction $d_{{z_t}}$}};",
        teacher_panel(),
        r"\end{tikzpicture}"])

    tex = (r"""\documentclass[border=8pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\setlength{\fboxsep}{0pt}
""" + color_defs + r"""
\begin{document}
\begin{minipage}{17.4cm}
\centering
""" + row1 + r"""

\vspace{2.5mm}
\begin{minipage}{16.2cm}
\raggedright\small\setlength{\parindent}{0pt}%
{\small\bfseries (c)\; natural-looking text, secretly labeled}\\[1.5pt]
{\itshape\scriptsize a real corpus excerpt --- each token is colored by the
state $z_t$ that was active while it was written:}\\[3.5pt]
\ldots\,%
""" + strip_tex(pairs) + r"\,\ldots" + r"""
\end{minipage}
\end{minipage}
\end{document}
""")
    (WORK / "method_pipeline.tex").write_text(tex)
    subprocess.run(["tectonic", "method_pipeline.tex"], cwd=WORK, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    "method_pipeline.pdf", "fig_method_pipeline"],
                   cwd=WORK, check=True)
    shutil.copy(WORK / "fig_method_pipeline.png",
                OUT / "fig_method_pipeline.png")
    shutil.copy(WORK / "fig_method_pipeline.png",
                Path.home() / "Desktop" / "fig_method_pipeline.png")
    print("wrote", OUT / "fig_method_pipeline.png", "and copied to Desktop")


if __name__ == "__main__":
    main()
