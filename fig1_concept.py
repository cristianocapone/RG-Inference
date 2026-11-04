"""
Figure 1 of the manuscript: the idea and the distilled result.

 (a) The two grouping rules in the same visual language.  Same network: two populations
     A and B driven by the same source (they fire together) that project to different
     targets.  Row 1, group by correlation: the equal-time similarity cannot tell A from
     B, they are merged into one block and the reduced network loses the direction of
     the flow.  Row 2, group by inferred in/out couplings: the similarity of the rows and
     columns of J separates A from B, each becomes a block and the flow is kept.
 (b) The result in one plot: advantage of the inference-based reduction over the
     correlation-based one (closure, output similarity) against the directedness of the
     recording at the model's timescale, for the cultures and for every in-vivo
     recording and condition (results/closure.npz, directedness.npz, allen_final/, hc11/).
Output: Figures/Fig1_concept.png / .pdf.  Run from the folder holding results/.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

CORAL, TEAL, ORANGE, GREY, INK = "#e8707e", "#2a9d8f", "#f6c177", "#c4c4c4", "0.25"
RED, OLIVE = "#d62728", "#8c8c00"
CARD_OLIVE, CARD_RED = "#f4f4e4", "#fbeeee"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})


def node(ax, xy, color, r=0.34, ring=None):
    ax.add_patch(Circle((xy[0] + 0.05, xy[1] - 0.05), r, fc="0.80", ec="none", zorder=2, alpha=0.55))
    ax.add_patch(Circle(xy, r, fc=color, ec=ring or "white", lw=2.0 if ring else 1.2, zorder=3))


def edge(ax, a, b, color, lw=1.1, rad=0.1, alpha=0.85, r=0.34):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=r * 13, shrinkB=r * 13, zorder=2, capstyle="round"))


def matrix(ax, x0, y0, cells, cell=0.55, labels=None):
    n = len(cells)
    for i in range(n):
        for j in range(n):
            ax.add_patch(FancyBboxPatch((x0 + j * cell + 0.04, y0 + (n - 1 - i) * cell + 0.04), cell - 0.08, cell - 0.08,
                                        boxstyle="round,pad=0.0,rounding_size=0.07", fc=cells[i][j], ec="none", zorder=2))
    if labels:
        for k, (lab, col) in enumerate(labels):
            ax.text(x0 - 0.25, y0 + (n - 0.5 - k) * cell, lab, ha="right", va="center", fontsize=6.5, color=col)
            ax.text(x0 + (k + 0.5) * cell, y0 - 0.25, lab, ha="center", va="top", fontsize=6.5, color=col)


def row(ax, rule):
    """One row of panel a: network -> similarity -> coarse-grained network."""
    ax.set_xlim(0, 30); ax.set_ylim(0, 8.6); ax.set_aspect("equal"); ax.axis("off")
    corr = rule == "correlation"
    col, card, title = (OLIVE, CARD_OLIVE, "group by correlation") if corr else (RED, CARD_RED, "group by inferred in/out couplings")
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 29.4, 8.0, boxstyle="round,pad=0.1,rounding_size=0.5", fc=card, ec="none", zorder=0))
    ax.text(0.9, 8.0, title, fontsize=9, color=col, weight="bold", va="top")
    # ---- the network (identical in both rows)
    src = (2.2, 4.2)
    A = [(5.0, 5.9), (5.0, 4.9)]
    B = [(5.0, 3.5), (5.0, 2.5)]
    TA = [(8.2, 6.2), (8.2, 5.1)]
    TB = [(8.2, 3.3), (8.2, 2.2)]
    for xy in A + B:
        edge(ax, src, xy, GREY, rad=0.05)
    for a in A:
        for t in TA:
            edge(ax, a, t, CORAL)
    for b in B:
        for t in TB:
            edge(ax, b, t, TEAL)
    node(ax, src, GREY)
    for xy in A:
        node(ax, xy, CORAL)
    for xy in B:
        node(ax, xy, TEAL)
    for xy in TA:
        node(ax, xy, "white", ring=CORAL)
    for xy in TB:
        node(ax, xy, "white", ring=TEAL)
    ax.text(5.0, 6.75, "A", ha="center", fontsize=8, color=CORAL, weight="bold")
    ax.text(5.0, 1.55, "B", ha="center", fontsize=8, color=TEAL, weight="bold")
    ax.text(8.2, 7.1, "targets", ha="center", fontsize=6.5, color="0.45")
    ax.text(5.2, 0.75, "A and B fire together, project to different targets", ha="center", fontsize=6.8, color="0.4", style="italic")
    # ---- the similarity used for grouping
    ax.add_patch(FancyArrowPatch((9.8, 4.3), (11.3, 4.3), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="0.5", capstyle="round"))
    labs = [("A", CORAL), ("A", CORAL), ("B", TEAL), ("B", TEAL)]
    x0, y0 = 13.0, 2.6
    if corr:
        cells = [[("white" if i == j else ORANGE) for j in range(4)] for i in range(4)]
        head, sub = "similarity of the activities", "A and B are correlated:\nindistinguishable"
    else:
        cells = [[("white" if i == j else (CORAL if (i < 2 and j < 2) else TEAL if (i >= 2 and j >= 2) else "#ececec")) for j in range(4)] for i in range(4)]
        head, sub = "similarity of in/out couplings", "A and B have different outputs:\ntwo groups"
    matrix(ax, x0, y0, cells, labels=labs)
    ax.text(x0 + 1.1, 5.7, head, ha="center", fontsize=7.5, color=INK, weight="bold")
    ax.text(x0 + 1.1, 1.75, sub, ha="center", va="top", fontsize=6.8, color=col, style="italic", linespacing=1.25)
    # ---- the coarse-grained network
    ax.add_patch(FancyArrowPatch((17.3, 4.3), (18.8, 4.3), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="0.5", capstyle="round"))
    ax.text(18.05, 4.9, "group", ha="center", fontsize=7, color="0.4")
    src2 = (20.6, 4.2)
    TA2 = [(26.6, 6.2), (26.6, 5.1)]
    TB2 = [(26.6, 3.3), (26.6, 2.2)]
    if corr:
        blk = (23.4, 4.2)
        edge(ax, src2, blk, GREY, rad=0.0, r=0.6)
        for t in TA2 + TB2:
            edge(ax, blk, t, "0.55", lw=1.3, r=0.6)
        node(ax, blk, ORANGE, r=0.72)
        ax.text(23.4, 3.1, "A+B", ha="center", va="top", fontsize=7, color=INK)
        ax.text(25.0, 0.75, "one block, two targets: direction lost", ha="center", fontsize=6.8, color=col, style="italic")
    else:
        bA, bB = (23.4, 5.4), (23.4, 3.0)
        for blk in (bA, bB):
            edge(ax, src2, blk, GREY, rad=0.05, r=0.5)
        for t in TA2:
            edge(ax, bA, t, CORAL, lw=1.4, r=0.5)
        for t in TB2:
            edge(ax, bB, t, TEAL, lw=1.4, r=0.5)
        node(ax, bA, CORAL, r=0.55)
        node(ax, bB, TEAL, r=0.55)
        ax.text(23.4, 6.45, "A", ha="center", fontsize=7, color=CORAL, weight="bold")
        ax.text(23.4, 1.95, "B", ha="center", va="top", fontsize=7, color=TEAL, weight="bold")
        ax.text(25.0, 0.75, "two blocks, each with its target: direction kept", ha="center", fontsize=6.8, color=col, style="italic")
    node(ax, src2, GREY)
    for xy in TA2:
        node(ax, xy, "white", ring=CORAL)
    for xy in TB2:
        node(ax, xy, "white", ring=TEAL)


def panel_result(ax):
    cl, dr = np.load("results/closure.npz"), np.load("results/directedness.npz")
    cults = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
    xc = [float(dr[f"cultures/{c}/lag1"]) for c in cults]
    yc = [np.nanmean((cl[f"cultures/{c}/inference_out/closure"] - cl[f"cultures/{c}/correlation/closure"])[1:7]) for c in cults]
    xv = [float(dr[f"mice/{m}/lag1"]) for m in ["Krebs", "Waksman", "Robbins"]]
    yv = [np.nanmean((cl[f"mice/{m}/inference_out/closure"] - cl[f"mice/{m}/correlation/closure"])[1:7]) for m in ["Krebs", "Waksman", "Robbins"]]
    for f in sorted(glob.glob("results/allen_final/*.npz")):
        r = dict(np.load(f))
        if "skipped" not in r:
            xv += [r[k][0] for k in r if k.startswith("top/")]; yv += [r[k][2] for k in r if k.startswith("top/")]
    for f in sorted(glob.glob("results/hc11/*.npz")):
        r = dict(np.load(f))
        xv += [r[k][0] for k in r if k.endswith("/10")]; yv += [r[k][2] for k in r if k.endswith("/10")]
    xv, yv = np.array(xv), np.array(yv)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.6")
    ax.tick_params(colors="0.4", labelsize=7)
    ax.axhline(0, color="0.75", ls=":", lw=1)
    ax.scatter(xv, yv, s=11, color="#9fb6c9", alpha=0.75, zorder=2)
    ax.scatter(xc, yc, s=46, color=RED, ec="white", lw=0.8, zorder=3)
    ax.set_xlabel("directedness of the dynamics", fontsize=8, color="0.35")
    ax.set_ylabel("advantage of inference\nover correlation", fontsize=8, color="0.35")
    ax.set_xlim(-0.005, 0.26); ax.set_ylim(-0.12, 0.115)
    ax.set_yticks([-0.1, 0, 0.1]); ax.set_xticks([0, 0.1, 0.2])
    ax.text(0.03, 0.095, "in vitro (9 cultures)", fontsize=8, color=RED, weight="bold")
    ax.text(0.03, -0.1, f"in vivo ({len(xv)} recordings)", fontsize=8, color="#4a6f8f", weight="bold")


def main():
    fig = plt.figure(figsize=(8.8, 7.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.45], hspace=0.08, wspace=0.0, left=0.03, right=0.99, top=0.97, bottom=0.07)
    row(fig.add_subplot(gs[0, :]), "correlation")
    row(fig.add_subplot(gs[1, :]), "inference")
    fig.text(0.01, 0.975, "a", fontsize=11, weight="bold")
    ax = fig.add_subplot(gs[2, 1:5])
    panel_result(ax)
    fig.text(0.01, 0.40, "b", fontsize=11, weight="bold")
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig1_concept.png", dpi=200)
    fig.savefig("Figures/Fig1_concept.pdf")
    print("saved Figures/Fig1_concept.png / .pdf")


if __name__ == "__main__":
    main()
