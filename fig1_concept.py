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
Output: Figures/Fig1_concept{,_row3,_cols}.png / .pdf (the manuscript uses _cols).  Run from the folder holding results/.
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


def _network(ax, ox, oy, caption):
    """The micro network (identical for both rules), drawn with its lower-left corner at (ox, oy)."""
    o = np.array([ox, oy])
    src = tuple(o + (0.9, 2.9))
    A = [tuple(o + (3.7, 4.6)), tuple(o + (3.7, 3.6))]
    B = [tuple(o + (3.7, 2.2)), tuple(o + (3.7, 1.2))]
    TA = [tuple(o + (6.9, 4.9)), tuple(o + (6.9, 3.8))]
    TB = [tuple(o + (6.9, 2.0)), tuple(o + (6.9, 0.9))]
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
    ax.text(ox + 3.7, oy + 5.45, "A", ha="center", fontsize=8, color=CORAL, weight="bold")
    ax.text(ox + 3.7, oy + 0.25, "B", ha="center", fontsize=8, color=TEAL, weight="bold")
    ax.text(ox + 6.9, oy + 5.8, "targets", ha="center", fontsize=6.5, color="0.45")
    ax.text(ox + 3.9, oy - 0.55, caption, ha="center", va="center", fontsize=6.8, color="0.4", style="italic", linespacing=1.25)


def _similarity(ax, ox, oy, corr, col, sub):
    labs = [("A", CORAL), ("A", CORAL), ("B", TEAL), ("B", TEAL)]
    if corr:
        cells = [[("white" if i == j else ORANGE) for j in range(4)] for i in range(4)]
        head = "similarity of the activities"
    else:
        cells = [[("white" if i == j else (CORAL if (i < 2 and j < 2) else TEAL if (i >= 2 and j >= 2) else "#ececec")) for j in range(4)] for i in range(4)]
        head = "similarity of in/out couplings"
    matrix(ax, ox + 1.3, oy + 1.3, cells, labels=labs)
    ax.text(ox + 2.4, oy + 4.4, head, ha="center", fontsize=7.5, color=INK, weight="bold")
    ax.text(ox + 2.4, oy + 0.45, sub, ha="center", va="top", fontsize=6.8, color=col, style="italic", linespacing=1.25)


def _coarse(ax, ox, oy, corr, col, caption):
    o = np.array([ox, oy])
    src2 = tuple(o + (0.9, 2.9))
    TA2 = [tuple(o + (6.9, 4.9)), tuple(o + (6.9, 3.8))]
    TB2 = [tuple(o + (6.9, 2.0)), tuple(o + (6.9, 0.9))]
    if corr:
        blk = tuple(o + (3.7, 2.9))
        edge(ax, src2, blk, GREY, rad=0.0, r=0.6)
        for t in TA2 + TB2:
            edge(ax, blk, t, "0.55", lw=1.3, r=0.6)
        node(ax, blk, ORANGE, r=0.72)
        ax.text(ox + 3.7, oy + 1.8, "A+B", ha="center", va="top", fontsize=7, color=INK)
    else:
        bA, bB = tuple(o + (3.7, 4.1)), tuple(o + (3.7, 1.7))
        for blk in (bA, bB):
            edge(ax, src2, blk, GREY, rad=0.05, r=0.5)
        for t in TA2:
            edge(ax, bA, t, CORAL, lw=1.4, r=0.5)
        for t in TB2:
            edge(ax, bB, t, TEAL, lw=1.4, r=0.5)
        node(ax, bA, CORAL, r=0.55)
        node(ax, bB, TEAL, r=0.55)
        ax.text(ox + 3.7, oy + 5.15, "A", ha="center", fontsize=7, color=CORAL, weight="bold")
        ax.text(ox + 3.7, oy + 0.65, "B", ha="center", va="top", fontsize=7, color=TEAL, weight="bold")
    node(ax, src2, GREY)
    for xy in TA2:
        node(ax, xy, "white", ring=CORAL)
    for xy in TB2:
        node(ax, xy, "white", ring=TEAL)
    ax.text(ox + 5.3, oy - 0.55, caption, ha="center", va="center", fontsize=6.8, color=col, style="italic", linespacing=1.25)


def _arrow(ax, a, b, label=None, label_dxy=(0, 0.6)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.3, color="0.5", capstyle="round"))
    if label:
        ax.text((a[0] + b[0]) / 2 + label_dxy[0], (a[1] + b[1]) / 2 + label_dxy[1], label, ha="center", va="center", fontsize=7, color="0.4")


def row(ax, rule, vertical=False):
    """One card of panel a: network -> similarity -> coarse-grained network, laid out left-to-right
    (vertical=False) or top-to-bottom (vertical=True)."""
    corr = rule == "correlation"
    col, card, title = (OLIVE, CARD_OLIVE, "group by correlation") if corr else (RED, CARD_RED, "group by inferred in/out couplings")
    if corr:
        sub, cap2 = "A and B are correlated:\nindistinguishable", "one block, two targets: direction lost"
    else:
        sub, cap2 = "A and B have different outputs:\ntwo groups", "two blocks, each with its target: direction kept"
    cap1 = "A and B fire together, project to different targets"
    ax.set_aspect("equal"); ax.axis("off")
    if not vertical:
        ax.set_xlim(0, 30); ax.set_ylim(0, 8.6)
        ax.add_patch(FancyBboxPatch((0.3, 0.3), 29.4, 8.0, boxstyle="round,pad=0.1,rounding_size=0.5", fc=card, ec="none", zorder=0))
        ax.text(0.9, 8.0, title, fontsize=9, color=col, weight="bold", va="top")
        _network(ax, 1.3, 1.3, cap1)
        _arrow(ax, (9.8, 4.3), (11.3, 4.3))
        _similarity(ax, 11.7, 1.3, corr, col, sub)
        _arrow(ax, (17.3, 4.3), (18.8, 4.3), "group")
        _coarse(ax, 19.7, 1.3, corr, col, cap2)
    else:
        W, H = 10.4, 25.0
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.add_patch(FancyBboxPatch((0.3, 0.3), W - 0.6, H - 0.6, boxstyle="round,pad=0.1,rounding_size=0.5", fc=card, ec="none", zorder=0))
        ax.text(W / 2, H - 0.7, title.replace("inferred in/out couplings", "inferred\nin/out couplings"), fontsize=9, color=col, weight="bold", va="top", ha="center", linespacing=1.15)
        y_net = H - 2.9 - 5.8
        _network(ax, 1.2, y_net, cap1.replace(", project", ",\nproject"))
        _arrow(ax, (W / 2, y_net - 1.2), (W / 2, y_net - 2.2))
        y_sim = y_net - 2.5 - 4.6
        _similarity(ax, W / 2 - 2.4, y_sim, corr, col, sub)
        _arrow(ax, (W / 2, y_sim - 0.9), (W / 2, y_sim - 1.9), "group", label_dxy=(1.3, 0))
        y_cg = y_sim - 2.1 - 5.4
        _coarse(ax, 1.2, y_cg, corr, col, cap2.replace(": ", ":\n"))


def panel_result(ax):
    cl, dr = np.load("results/closure.npz"), np.load("results/directedness.npz")
    cults = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
    xc = [float(dr[f"cultures/{c}/lag1"]) for c in cults]
    yc = [np.nanmean((cl[f"cultures/{c}/inference_out/closure"] - cl[f"cultures/{c}/correlation/closure"])[1:7]) for c in cults]
    xv = [float(dr[f"mice/{m}/lag1"]) for m in ["Krebs", "Waksman", "Robbins"]]
    yv = [np.nanmean((cl[f"mice/{m}/inference_out/closure"] - cl[f"mice/{m}/correlation/closure"])[1:7]) for m in ["Krebs", "Waksman", "Robbins"]]
    src = ["cortex, 2-photon"] * 3
    for f in sorted(glob.glob("results/allen_final/*.npz")):  # one point per session: mean over conditions
        r = dict(np.load(f))
        if "skipped" not in r:
            ks = [k for k in r if k.startswith("top/")]
            xv.append(np.mean([r[k][0] for k in ks])); yv.append(np.mean([r[k][2] for k in ks])); src.append("cortex, Neuropixels")
    for f in sorted(glob.glob("results/hc11/*.npz")):  # one point per session: mean over epochs
        r = dict(np.load(f))
        ks = [k for k in r if k.endswith("/10")]
        xv.append(np.mean([r[k][0] for k in ks])); yv.append(np.mean([r[k][2] for k in ks])); src.append("hippocampus")
    xv, yv, src = np.array(xv), np.array(yv), np.array(src)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.6")
    ax.tick_params(colors="0.4", labelsize=7)
    ax.axhline(0, color="0.75", ls=":", lw=1)

    def msem(v):
        v = np.asarray(v)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    BLUE = "#7a9ab8"  # in vivo: one point, mean over all sessions
    (mx, ex), (my, ey) = msem(xv), msem(yv)
    ax.errorbar(mx, my, xerr=ex, yerr=ey, fmt="o", color=BLUE, ms=8, mec="white", mew=0.8, capsize=2.5, zorder=2)
    o = np.argsort(xc)  # in vitro: tertiles of directedness
    pts = [(msem(np.asarray(xc)[ix]), msem(np.asarray(yc)[ix])) for ix in np.array_split(o, 3)]
    ax.errorbar([p[0][0] for p in pts], [p[1][0] for p in pts], xerr=[p[0][1] for p in pts], yerr=[p[1][1] for p in pts],
                fmt="o-", color=RED, ms=7, mec="white", mew=0.8, lw=1.2, capsize=2.5, zorder=3)
    ax.set_xlabel("directedness of the dynamics", fontsize=8, color="0.35")
    ax.set_ylabel("advantage of inference\nover correlation", fontsize=8, color="0.35")
    ax.set_xlim(-0.005, 0.26); ax.set_ylim(-0.06, 0.115)
    ax.set_yticks([-0.05, 0, 0.05, 0.1]); ax.set_xticks([0, 0.1, 0.2])
    ax.text(0.03, 0.095, "in vitro (9 cultures, tertiles)", fontsize=8, color=RED, weight="bold")
    ax.text(0.03, -0.045, f"in vivo ({len(xv)} sessions)", fontsize=8, color="#4a6f8f", weight="bold")


def main():
    os.makedirs("Figures", exist_ok=True)
    # ---- stacked: two horizontal cards on top, result below (manuscript version)
    fig = plt.figure(figsize=(8.8, 7.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.45], hspace=0.08, wspace=0.0, left=0.03, right=0.99, top=0.97, bottom=0.07)
    row(fig.add_subplot(gs[0, :]), "correlation")
    row(fig.add_subplot(gs[1, :]), "inference")
    fig.text(0.01, 0.975, "a", fontsize=11, weight="bold")
    panel_result(fig.add_subplot(gs[2, 1:5]))
    fig.text(0.01, 0.40, "b", fontsize=11, weight="bold")
    fig.savefig("Figures/Fig1_concept.png", dpi=200)
    fig.savefig("Figures/Fig1_concept.pdf")
    plt.close(fig)
    # ---- row3: three panels side by side (two vertical cards, result)
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.08, left=0.02, right=0.99, top=0.97, bottom=0.03)
    row(fig.add_subplot(gs[0, 0]), "correlation", vertical=True)
    row(fig.add_subplot(gs[0, 1]), "inference", vertical=True)
    fig.text(0.01, 0.965, "a", fontsize=11, weight="bold")
    ax = fig.add_subplot(gs[0, 2])
    panel_result(ax)
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.07, 0.30, pos.width - 0.08, 0.42])
    fig.text(pos.x0 + 0.01, 0.965, "b", fontsize=11, weight="bold")
    fig.savefig("Figures/Fig1_concept_row3.png", dpi=200)
    fig.savefig("Figures/Fig1_concept_row3.pdf")
    plt.close(fig)
    # ---- cols: two horizontal cards stacked on the left, result on the right
    fig = plt.figure(figsize=(12.6, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], hspace=0.06, wspace=0.12, left=0.02, right=0.98, top=0.95, bottom=0.05)
    row(fig.add_subplot(gs[0, 0]), "correlation")
    row(fig.add_subplot(gs[1, 0]), "inference")
    fig.text(0.01, 0.955, "a", fontsize=11, weight="bold")
    ax = fig.add_subplot(gs[:, 1])
    panel_result(ax)
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.05, 0.17, pos.width - 0.06, 0.68])
    fig.text(pos.x0 + 0.005, 0.955, "b", fontsize=11, weight="bold")
    fig.savefig("Figures/Fig1_concept_cols.png", dpi=200)
    fig.savefig("Figures/Fig1_concept_cols.pdf")
    print("saved Figures/Fig1_concept{,_row3,_cols}.png / .pdf")


if __name__ == "__main__":
    main()
