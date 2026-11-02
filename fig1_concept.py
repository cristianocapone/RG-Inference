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
from matplotlib.patches import FancyArrowPatch

import repro_figure3 as F
from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity
from cover_figure import panel_idea


def big_arrow(fig, x0, x1, y, text=""):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=18, lw=2, color="0.3"))
    if text:
        fig.text((x0 + x1) / 2, y + 0.035, text, ha="center", fontsize=8.5, color="0.2")


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

    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], width_ratios=[1.3, 1, 1, 1.3, 1],
                          hspace=0.55, wspace=0.55, left=0.04, right=0.98, top=0.93, bottom=0.04)
    t = slice(0, 600)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(S[t].T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_title("activity $S_i(t)$", fontsize=10)
    ax.set_xlabel("time")
    ax.set_ylabel("unit")
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(J, cmap="RdBu_r", vmin=-np.percentile(np.abs(J), 99), vmax=np.percentile(np.abs(J), 99))
    ax.set_title("inferred couplings $J_{ij}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("pre"); ax.set_ylabel("post")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(G, cmap="viridis", vmin=-0.2, vmax=0.6)
    ax.set_title("similarity of\ninput/output vectors", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(Sb[t][:, o].T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_title("block activity (pairs averaged)", fontsize=10)
    ax.set_xlabel("time"); ax.set_ylabel("block")
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(Jb, cmap="RdBu_r", vmin=-np.percentile(np.abs(Jb), 99), vmax=np.percentile(np.abs(Jb), 99))
    ax.set_title("couplings re-inferred\non the blocks", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.96, "a", fontsize=14, weight="bold")
    for x0, x1, txt in [(0.198, 0.232, "infer"), (0.425, 0.45, "compare"), (0.585, 0.61, "pair most\nsimilar"),
                        (0.835, 0.86, "re-infer")]:
        big_arrow(fig, x0, x1, 0.77, txt)
    fig.text(0.5, 0.525, "iterate: $N \\to N/2 \\to N/4 \\to \\dots$", ha="center", fontsize=10, color="0.25")

    ax = fig.add_subplot(gs[1, :3])
    panel_idea(ax)
    ax.set_title("", loc="left")
    fig.text(0.01, 0.47, "b", fontsize=14, weight="bold")
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    ax.text(0.0, 0.92, "What a good reduction must do", fontsize=11, weight="bold", va="top")
    ax.text(0.0, 0.74,
            "closure: the blocks predict their own future\n"
            "as well as the full microscopic state does,\n"
            "$\\mathcal{C} = R^2[y(t{+}1) \\,|\\, y(t)] \\;/\\; R^2[y(t{+}1) \\,|\\, S(t)]$\n\n"
            "perturbation: the reduced model responds to a\n"
            "stimulus as the full network does\n\n"
            "Grouping units with the same inputs and outputs\n"
            "makes each block an exact reduction (lumpability);\n"
            "grouping by correlation does not, when the\n"
            "dynamics is directional.",
            fontsize=9.5, va="top", linespacing=1.4)
    fig.text(0.62, 0.47, "c", fontsize=14, weight="bold")
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig1_concept.png", dpi=200)
    fig.savefig("Figures/Fig1_concept.pdf")
    print("saved Figures/Fig1_concept.png / .pdf")


if __name__ == "__main__":
    main()
