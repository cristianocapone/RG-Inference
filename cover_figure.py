"""
Three-panel figure for the cover letter (the basic idea in one row).

 (1) Idea: two sub-populations fire together (same correlations) but project to
     different targets.  Correlation-based coarse-graining merges them and loses the
     direction of the flow; inference-based coarse-graining groups units with the
     same inferred inputs and outputs and keeps it.
 (2) Ground truth (sequence network of Fig. 3, N = 1024, 64 blocks): stimulating A in
     the reduced model biases the transition towards A's target only when the model
     was reduced by inference (results/fig3_ml.npz from repro_figure3_ml.py).
 (3) Data: closure advantage of inference over correlation vs the number of observed
     electrodes in the cultures (random subsets), one line per culture coloured by its
     directedness: the advantage needs both directional dynamics and enough of the
     observed circuit (results/partial_observation_cultures.npz).
Inputs: results/fig3_ml.npz, results/partial_observation_cultures.npz.
Output: Figures/Cover_figure.png / .pdf.  Run from the folder holding results/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

RED, OLIVE, BLUE, GREY = "#d62728", "#8c8c00", "#4a7fd6", "0.55"
CHAINS = [0.0, 0.05, 0.1, 0.2]
CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
MICE = ["Krebs", "Waksman", "Robbins"]


plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "axes.titleweight": "regular"})
INK, CARD_OLIVE, CARD_RED = "0.25", "#f4f4e4", "#fbeeee"


def arrow(ax, a, b, color="k", lw=1.6, rad=0.0, alpha=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=9, shrinkB=9, alpha=alpha, capstyle="round"))


def node(ax, xy, color, r=0.17, ec="white"):
    ax.add_patch(Circle((xy[0] + 0.03, xy[1] - 0.03), r, fc="0.82", ec="none", zorder=2, alpha=0.6))   # soft shadow
    ax.add_patch(Circle(xy, r, fc=color, ec=ec, lw=1.2, zorder=3))


def card(ax, xy, w, h, fc, ec="none", lw=0, r=0.18):
    ax.add_patch(FancyBboxPatch((xy[0] + 0.05, xy[1] - 0.05), w, h, boxstyle=f"round,pad=0.04,rounding_size={r}", fc="0.86",
                                ec="none", alpha=0.45, zorder=0))
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.04,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=1))


def panel_idea(ax, fs=1.0):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    cA, cB, cP, cQ = "#f6c177", "#f6c177", "#e8707e", "#b3b040"
    # microscopic network (left)
    src = (0.7, 3.1)
    A = [(2.0, 4.0), (2.0, 3.5)]
    Bn = [(2.0, 2.7), (2.0, 2.2)]
    P = [(3.4, 4.3), (3.4, 3.8)]
    Q = [(3.4, 2.4), (3.4, 1.9)]
    node(ax, src, "0.8")
    for xy in A + Bn:
        node(ax, xy, cA)
    for xy in P:
        node(ax, xy, cP)
    for xy in Q:
        node(ax, xy, cQ)
    for xy in A + Bn:
        arrow(ax, src, xy, color=GREY, lw=1.0)
    for a in A:
        for p in P:
            arrow(ax, a, p, color=cP, lw=1.2)
    for b in Bn:
        for q in Q:
            arrow(ax, b, q, color=cQ, lw=1.2)
    ax.text(2.0, 4.65, "A", ha="center", fontsize=10 * fs, color=INK)
    ax.text(2.0, 1.55, "B", ha="center", fontsize=10 * fs, color=INK)
    ax.text(2.0, 5.55, "A and B fire together,\nproject to different targets", ha="center", fontsize=8.5 * fs, color=INK)
    ax.annotate("", xy=(5.0, 4.6), xytext=(4.0, 3.6), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=OLIVE, alpha=0.8, connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(5.0, 1.6), xytext=(4.0, 2.6), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED, alpha=0.8, connectionstyle="arc3,rad=0.2"))
    # correlation-based (top right)
    card(ax, (5.1, 3.55), 4.6, 2.35, CARD_OLIVE)
    ax.text(7.4, 5.6, "group by correlation", ha="center", fontsize=9 * fs, color=OLIVE, weight="bold")
    m = (6.3, 4.45)
    node(ax, m, cA, r=0.26)
    node(ax, (8.4, 5.0), cP, r=0.22)
    node(ax, (8.4, 3.95), cQ, r=0.22)
    arrow(ax, m, (8.4, 5.0), color="0.5", lw=1.6)
    arrow(ax, m, (8.4, 3.95), color="0.5", lw=1.6)
    ax.text(6.3, 3.85, "A+B", ha="center", fontsize=8 * fs, color=INK)
    ax.text(7.4, 3.65, "direction lost", ha="center", fontsize=8 * fs, color=OLIVE, style="italic")
    # inference-based (bottom right)
    card(ax, (5.1, 0.25), 4.6, 2.35, CARD_RED)
    ax.text(7.4, 2.3, "group by inferred in/out couplings", ha="center", fontsize=9 * fs, color=RED, weight="bold")
    a, b = (6.3, 1.75), (6.3, 0.85)
    node(ax, a, cA, r=0.22)
    node(ax, b, cB, r=0.22)
    node(ax, (8.4, 1.75), cP, r=0.22)
    node(ax, (8.4, 0.85), cQ, r=0.22)
    arrow(ax, a, (8.4, 1.75), color=cP, lw=1.8)
    arrow(ax, b, (8.4, 0.85), color=cQ, lw=1.8)
    ax.text(5.75, 1.75, "A", ha="center", va="center", fontsize=8 * fs, color=INK)
    ax.text(5.75, 0.85, "B", ha="center", va="center", fontsize=8 * fs, color=INK)
    ax.text(7.4, 0.4, "direction kept", ha="center", fontsize=8 * fs, color=RED, style="italic")
    ax.set_title("1  Idea", loc="left", fontsize=11 * fs, weight="bold")


def panel_ground_truth(ax):
    """Sequence network of Fig. 3 (N = 1024): stimulating A biases the transition
    towards A's target only in the model reduced by inference."""
    d = np.load("results/fig3_ml.npz")
    for rule, col, lab in [("inference", RED, "inference-based"), ("correlation", OLIVE, "correlation-based")]:
        Is = d[f"{rule}/Is"]
        ax.plot(Is, d[f"{rule}/p3"], "o-", color=col, ms=4, lw=1.8, label=f"{lab}: to A's target")
        ax.plot(Is, d[f"{rule}/p4"], "s--", color=col, ms=3, lw=1.2, alpha=0.7, label=f"{lab}: to B's target")
    ax.axhline(0.5, color="0.7", ls=":")
    ax.set_ylim(0.15, 0.8)
    ax.set_xlabel("stimulation of A in the reduced model")
    ax.set_ylabel("transition probability")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("2  Reduced model, ground truth", loc="left", fontsize=11, weight="bold")


def panel_data(ax):
    """Cultures: advantage of inference vs number of observed electrodes (random
    subsets), one line per culture coloured by its directedness."""
    d = np.load("results/partial_observation_cultures.npz")
    cults = sorted({k.split("/")[0] for k in d.files}, key=lambda c: int(c[1:]))
    dirs = {c: np.mean([d[k][:, 7].mean() for k in d.files if k.startswith(c + "/")]) for c in cults}
    norm = matplotlib.colors.Normalize(0.0, 0.3)
    cmap = matplotlib.colormaps["viridis"]
    for c in cults:
        ns = sorted(int(k.split("/")[1]) for k in d.files if k.startswith(c + "/"))
        ax.plot(ns, [d[f"{c}/{n}"][:, 2].mean() for n in ns], "o-", color=cmap(norm(dirs[c])), ms=3, lw=1.6)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed electrodes")
    ax.set_ylabel("advantage of inference\nover correlation (closure)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, label="directedness of the activity (lag 1)")
    ax.set_title("3  Data: needs directedness and enough circuit", loc="left", fontsize=11, weight="bold")


def main():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    panel_idea(axs[0])
    panel_ground_truth(axs[1])
    panel_data(axs[2])
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Cover_figure.png", dpi=200)
    fig.savefig("Figures/Cover_figure.pdf")
    print("saved Figures/Cover_figure.png / .pdf")


if __name__ == "__main__":
    main()
