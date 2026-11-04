"""
Figure 1 of the Nat Comm manuscript: the method and the idea.

 (a) One step of inference-based RG on the sequence network of Fig. 3 (N = 512):
     activity -> inferred couplings J -> similarity of input/output synaptic vectors
     -> pairs averaged into block units -> couplings re-inferred on the blocks.
 (b) Why it matters: two sub-populations that fire together but project to different
     targets are merged by correlation-based grouping and kept apart by grouping on
     inferred couplings (same cartoon as the cover-letter figure).
(No data needed: both panels are schematic.)
Output: Figures/Fig1_concept.png / .pdf.  Run from the folder holding sim_cache/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cover_figure import panel_idea


def big_arrow(fig, x0, x1, y, text=""):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=11, lw=1.3, color="0.45", capstyle="round"))
    if text:
        fig.text((x0 + x1) / 2, y + 0.06, text, ha="center", fontsize=7.5, color="0.35")


def stage(ax, img, title, xlabel="", ylabel="", **kw):
    """One stage of the RG step: image on a soft rounded card, no frame."""
    ax.imshow(img, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8.5, color="0.2", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color="0.4", labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color="0.4", labelpad=2)
    bb = ax.get_position()
    pad = 0.012
    ax.figure.patches.append(FancyBboxPatch((bb.x0 - pad, bb.y0 - pad - 0.03), bb.width + 2 * pad, bb.height + 2 * pad + 0.06,
                                            boxstyle="round,pad=0.01,rounding_size=0.015", transform=ax.figure.transFigure,
                                            fc="0.96", ec="none", zorder=-1))



CORAL, TEAL, GREY, INK = "#e8707e", "#2a9d8f", "#c4c4c4", "0.25"


def snode(ax, xy, color, r=0.32, z=3):
    ax.add_patch(plt.Circle((xy[0] + 0.05, xy[1] - 0.05), r, fc="0.80", ec="none", zorder=z - 1, alpha=0.55))
    ax.add_patch(plt.Circle(xy, r, fc=color, ec="white", lw=1.2, zorder=z))


def sedge(ax, a, b, color, lw=1.0, rad=0.15, alpha=0.55, arrow=False, r=0.32):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if arrow else "-", mutation_scale=9, lw=lw, color=color, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=r * 14, shrinkB=r * 14, zorder=2, capstyle="round"))


def panel_scheme(ax):
    """Panel a as a schematic: network -> inferred couplings and similarity -> coarse-grained network."""
    ax.set_xlim(0, 30); ax.set_ylim(0, 10.5); ax.set_aspect("equal"); ax.axis("off")
    # ---- stage 1: the network
    A = [(2.0, 7.2), (3.0, 8.1), (3.7, 6.9), (2.3, 6.0), (4.0, 7.9)]
    B = [(2.2, 3.2), (3.2, 2.4), (3.8, 3.6), (2.6, 4.3), (1.6, 2.5)]
    G = [(5.6, 8.7), (6.7, 7.0), (5.3, 5.3), (7.5, 5.5), (6.3, 3.0), (7.7, 2.2), (8.3, 7.9), (8.5, 3.9)]
    for grp, col in [(A, CORAL), (B, TEAL)]:
        for i, j in [(0, 1), (1, 4), (2, 3), (0, 3)]:
            sedge(ax, grp[i], grp[j], col, lw=0.8, rad=0.2, alpha=0.3)
    for a, b in [(0, 1), (2, 3), (4, 5), (3, 6), (2, 4), (1, 3), (6, 7), (3, 7)]:
        sedge(ax, G[a], G[b], GREY, lw=0.9, rad=0.15, alpha=0.8)
    for dst in A:
        sedge(ax, G[2], dst, GREY, lw=0.9, rad=0.1, alpha=0.8, arrow=True)
    for dst in B:
        sedge(ax, G[4], dst, GREY, lw=0.9, rad=0.1, alpha=0.8, arrow=True)
    for src in A:
        for dst in (G[1], G[6]):
            sedge(ax, src, dst, CORAL, lw=1.0, rad=0.1, alpha=0.7, arrow=True)
    for src in B:
        for dst in (G[5], G[7]):
            sedge(ax, src, dst, TEAL, lw=1.0, rad=0.1, alpha=0.7, arrow=True)
    for xy in G:
        snode(ax, xy, GREY)
    for xy in A:
        snode(ax, xy, CORAL)
    for xy in B:
        snode(ax, xy, TEAL)
    ax.text(5.0, 10.0, "network", ha="center", fontsize=9, color=INK, weight="bold")
    ax.text(5.0, 0.55, "only the activity $S_i(t)$ is observed", ha="center", fontsize=7.5, color="0.4", style="italic")
    # ---- stage 2: inferred couplings and similarity
    x0, y0, cell = 11.3, 2.6, 0.62
    import itertools
    rng = np.random.default_rng(3)
    for i, j in itertools.product(range(8), range(8)):
        if i == j:
            col = "white"
        elif i < 3 and j < 3:
            col = CORAL
        elif 3 <= i < 6 and 3 <= j < 6:
            col = TEAL
        elif i < 3 and j >= 6:
            col = "#f5b8bf"
        elif 3 <= i < 6 and j >= 6:
            col = "#9fd3cc"
        else:
            col = (0.93, 0.93, 0.93) if rng.random() < 0.6 else (0.86, 0.86, 0.86)
        ax.add_patch(FancyBboxPatch((x0 + j * cell + 0.05, y0 + (7 - i) * cell + 0.05), cell - 0.1, cell - 0.1,
                                    boxstyle="round,pad=0.0,rounding_size=0.08", fc=col, ec="none", zorder=2))
    ax.text(x0 + 4 * cell, y0 + 8 * cell + 0.35, "inferred couplings $J_{ij}$", ha="center", fontsize=9, color=INK, weight="bold")
    ax.text(x0 - 0.3, y0 + 4 * cell, "post", ha="right", va="center", fontsize=7.5, color="0.4", rotation=90)
    ax.text(x0 + 4 * cell, y0 - 0.35, "pre", ha="center", va="top", fontsize=7.5, color="0.4")
    # brackets: rows with the same in/out vectors
    for (r0, r1, col) in [(0, 3, CORAL), (3, 6, TEAL)]:
        ax.plot([x0 + 8 * cell + 0.25] * 2, [y0 + (8 - r1) * cell + 0.08, y0 + (8 - r0) * cell - 0.08], color=col, lw=2.2, solid_capstyle="round", zorder=3)
    ax.text(x0 + 8 * cell + 0.5, y0 + (8 - 1.5) * cell, "same\nin/out", ha="left", va="center", fontsize=7, color=CORAL)
    ax.text(x0 + 8 * cell + 0.5, y0 + (8 - 4.5) * cell, "same\nin/out", ha="left", va="center", fontsize=7, color=TEAL)
    ax.text(x0 + 4 * cell + 1.0, 0.55, "units with the same inputs and outputs $\\to$ one block (one colour)", ha="center", fontsize=7.5, color="0.4", style="italic")
    # ---- stage 3: coarse-grained network
    bA, bB = (23.0, 7.2), (23.0, 3.2)
    bG = [(26.3, 7.7), (26.0, 5.1), (26.6, 2.6), (28.6, 6.6), (28.4, 3.7)]
    for a, b in [(0, 3), (1, 3), (1, 4), (2, 4)]:
        sedge(ax, bG[a], bG[b], GREY, lw=1.1, rad=0.15, alpha=0.8, r=0.5)
    for dst in (bA, bB):
        sedge(ax, bG[1], dst, GREY, lw=1.2, rad=0.1, alpha=0.9, arrow=True, r=0.6)
    for dst in (bG[0], bG[3]):
        sedge(ax, bA, dst, CORAL, lw=1.8, rad=0.1, alpha=0.9, arrow=True, r=0.6)
    for dst in (bG[2], bG[4]):
        sedge(ax, bB, dst, TEAL, lw=1.8, rad=0.1, alpha=0.9, arrow=True, r=0.6)
    for xy in bG:
        snode(ax, xy, GREY, r=0.5)
    snode(ax, bA, CORAL, r=0.85)
    snode(ax, bB, TEAL, r=0.85)
    ax.text(25.5, 10.0, "coarse-grained network", ha="center", fontsize=9, color=INK, weight="bold")
    ax.text(25.5, 0.55, "blocks keep their inputs and outputs", ha="center", fontsize=7.5, color="0.4", style="italic")
    # ---- arrows between stages
    for xa, xb, lab in [(9.2, 10.9, "infer"), (19.0, 20.9, "group")]:
        ax.add_patch(FancyArrowPatch((xa, 5.4), (xb, 5.4), arrowstyle="-|>", mutation_scale=12, lw=1.4, color="0.45", capstyle="round"))
        ax.text((xa + xb) / 2, 5.95, lab, ha="center", fontsize=8, color="0.35")
    ax.annotate("", xy=(5.0, 9.3), xytext=(25.5, 9.3), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.6", ls="--",
                                                                        connectionstyle="arc3,rad=0.12", mutation_scale=10))
    ax.text(15.3, 9.55, "re-infer and iterate:  $N \\to N/2 \\to N/4 \\to \\dots$", ha="center", va="bottom", fontsize=8, color="0.35",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc="0.96", ec="none"))


def main():
    plt.rcParams.update({"font.size": 8, "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(8.8, 5.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.95], width_ratios=[0.7, 1, 1, 1, 0.7],
                          hspace=0.2, wspace=0.0, left=0.03, right=0.99, top=0.97, bottom=0.03)
    t = slice(0, 600)
    panel_scheme(fig.add_subplot(gs[0, :]))
    fig.text(0.01, 0.96, "a", fontsize=11, weight="bold")

    ax = fig.add_subplot(gs[1, 1:4])
    panel_idea(ax, fs=0.85)
    ax.set_title("", loc="left")
    fig.text(0.01, 0.46, "b", fontsize=11, weight="bold")
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig1_concept.png", dpi=200)
    fig.savefig("Figures/Fig1_concept.pdf")
    print("saved Figures/Fig1_concept.png / .pdf")


if __name__ == "__main__":
    main()
