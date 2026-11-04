"""
Figure 1 of the Nat Comm manuscript: the method and the idea.

 (a) One step of inference-based RG on the sequence network of Fig. 3 (N = 512):
     activity -> inferred couplings J -> similarity of input/output synaptic vectors
     -> pairs averaged into block units -> couplings re-inferred on the blocks.
 (b) Why it matters: two sub-populations that fire together but project to different
     targets are merged by correlation-based grouping and kept apart by grouping on
     inferred couplings (same cartoon as the cover-letter figure).
Input: sim_cache/seq512_T300000.npy (make_sequence_data.py).
Output: Figures/Fig1_concept.png / .pdf.  Run from the folder holding sim_cache/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import repro_figure3 as F
from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity
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


def main():
    S = np.load("sim_cache/seq512_T300000.npy")[:60000].astype(np.float32)
    F.N = 512
    _, xi, lab = F.seq_network()
    J = infer_J(S, 0)
    G = synaptic_similarity(J)
    p = greedy_pairs(G)
    Sb = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
    pos = 0.5 * (p[:, 0] + p[:, 1])
    o = np.argsort(pos)
    Jb = infer_J(Sb, 1)[np.ix_(o, o)]

    plt.rcParams.update({"font.size": 8, "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(8.8, 4.9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], width_ratios=[1.3, 1, 1, 1.3, 1],
                          hspace=0.6, wspace=0.6, left=0.04, right=0.99, top=0.92, bottom=0.03)
    t = slice(0, 600)
    stage(fig.add_subplot(gs[0, 0]), S[t].T, "activity $S_i(t)$", "time", "unit", aspect="auto", cmap="Greys", interpolation="nearest")
    stage(fig.add_subplot(gs[0, 1]), J, "inferred couplings $J_{ij}$", "pre", "post", cmap="RdBu_r",
          vmin=-np.percentile(np.abs(J), 99), vmax=np.percentile(np.abs(J), 99))
    stage(fig.add_subplot(gs[0, 2]), G, "similarity of\nin/out vectors", cmap="viridis", vmin=-0.2, vmax=0.6)
    stage(fig.add_subplot(gs[0, 3]), Sb[t][:, o].T, "block activity", "time", "block", aspect="auto", cmap="Greys", interpolation="nearest")
    stage(fig.add_subplot(gs[0, 4]), Jb, "re-inferred\ncouplings", cmap="RdBu_r",
          vmin=-np.percentile(np.abs(Jb), 99), vmax=np.percentile(np.abs(Jb), 99))
    fig.text(0.01, 0.96, "a", fontsize=11, weight="bold")
    for x0, x1, txt in [(0.197, 0.235, "infer"), (0.418, 0.452, "compare"), (0.585, 0.615, "pair"),
                        (0.83, 0.862, "re-infer")]:
        big_arrow(fig, x0, x1, 0.77, txt)
    fig.text(0.5, 0.525, "iterate:  $N \\to N/2 \\to N/4 \\to \\dots$", ha="center", va="center", fontsize=9.5, color="0.3",
             bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc="0.96", ec="none"))

    ax = fig.add_subplot(gs[1, :3])
    panel_idea(ax, fs=0.8)
    ax.set_title("", loc="left")
    fig.text(0.01, 0.47, "b", fontsize=11, weight="bold")
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    from matplotlib.patches import FancyBboxPatch as FBP
    ax.add_patch(FBP((0.0, 0.02), 0.98, 0.96, boxstyle="round,pad=0.02,rounding_size=0.06", fc="0.96", ec="none", zorder=0))
    ax.text(0.05, 0.92, "What a good reduction must do", fontsize=9, weight="bold", va="top", color="0.2")
    for y, head, body in [(0.76, "closure", "the blocks predict their own future as well\nas the full microscopic state does:\n"
                                            "$\\mathcal{C} = R^2[y(t{+}1)\\,|\\,y(t)] \\;/\\; R^2[y(t{+}1)\\,|\\,S(t)]$"),
                          (0.45, "perturbation", "the reduced model responds to a stimulus\nas the full network does")]:
        ax.plot(0.085, y - 0.02, "o", ms=7, color="#d62728", zorder=2, clip_on=False)
        ax.text(0.14, y, head, fontsize=8.3, weight="bold", va="top", color="#8b0000")
        ax.text(0.14, y - 0.07, body, fontsize=7.6, va="top", color="0.25", linespacing=1.35)
    ax.text(0.05, 0.2, "Units with the same inputs and outputs form an exact\nreduction (lumpability); correlated units need not,\nwhen the dynamics is directional.",
            fontsize=7.6, va="top", color="0.35", style="italic", linespacing=1.35)
    fig.text(0.62, 0.47, "c", fontsize=11, weight="bold")
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig1_concept.png", dpi=200)
    fig.savefig("Figures/Fig1_concept.pdf")
    print("saved Figures/Fig1_concept.png / .pdf")


if __name__ == "__main__":
    main()
