"""
Figure 3 of the Nat Comm manuscript (sequence network, N = 1024, exact-ML inference),
from results/fig3_ml.npz written by repro_figure3_ml.py.

 (a) true couplings          (b) activity of the network: 1 -> 2 -> {3|4} -> 5 -> 1
 (c) stimulation protocol    (d) couplings of the reduced models at RG step 5 (64 blocks)
 (e) activity of the reduced models without and with stimulation of 2a
 (f) transition probability 2 -> 3 and 2 -> 4 vs stimulus intensity
Output: Figures/Fig3_N1024.png / .pdf.  Run from the folder holding results/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

import repro_figure3 as F

RED, OLIVE = "#d62728", "#8c8c00"
P3, P4 = "#e06070", "#b8a52a"
LO, HI = 0.0, float(sys.argv[1]) if len(sys.argv) > 1 else 8.0


def raster(ax, R, comp, title, label):
    ov = F.overlaps(R, comp)
    ax.imshow(R.T, aspect="auto", cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    for nm, col in [("3", P3), ("4", P4)]:
        act = F.activation_wave(ov[nm])
        ax.fill_between(np.arange(len(act)), -0.5, R.shape[1] - 0.5, where=act, color=col, alpha=0.35,
                        step="mid", lw=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=7)
    if title:
        ax.set_title(title, fontsize=8.5)


def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for xy, r, c, t in [((2.4, 3.9), 0.75, "#f2b36b", "2a"), ((2.4, 2.1), 0.75, "#f2b36b", "2b"),
                        ((7.3, 4.6), 0.85, P3, "3"), ((7.3, 1.4), 0.85, P4, "4")]:
        ax.add_patch(Circle(xy, r, fc=c, ec="k", lw=0.8))
        ax.text(*xy, t, ha="center", va="center", fontsize=8)
    for a, b, c in [((3.2, 4.1), (6.4, 4.5), P3), ((3.2, 1.9), (6.4, 1.5), P4)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.4, color=c))
    ax.text(0.2, 3.9, "+I", fontsize=9, color=RED, va="center", weight="bold")
    ax.text(0.2, 2.1, "$-$I", fontsize=9, color="0.3", va="center", weight="bold")
    ax.text(5.0, 5.6, "stimulate 2a, inhibit 2b", ha="center", fontsize=7.5)


def main():
    d = np.load("results/fig3_ml.npz")
    plt.rcParams.update({"font.size": 7.5})
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 2.6, 1.15], height_ratios=[1.1, 0.1, 1.25, 1.25],
                          hspace=0.55, wspace=0.35, left=0.06, right=0.98, top=0.95, bottom=0.06)
    J = d["J_true"]
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(J * 1e4, cmap="viridis", vmin=-2, vmax=16)
    ax.set_title("couplings", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("pre"); ax.set_ylabel("post")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("$J_{ij}\\times10^{4}$", fontsize=6.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(d["S0_snapshot"][:600].T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_title("activity of the network", fontsize=8.5)
    ax.set_xlabel("time"); ax.set_ylabel("unit")
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    schematic(ax)
    for k, (lab, x, y) in enumerate([("a", 0.01, 0.97), ("b", 0.27, 0.97), ("c", 0.77, 0.97)]):
        fig.text(x, y, lab, fontsize=11, weight="bold")

    rows = {"correlation": (2, "grouped by correlation", OLIVE), "inference": (3, "grouped by inferred couplings", RED)}
    for rule, (r0, name, col) in rows.items():
        Jk = d[f"{rule}/J"]
        comp = d[f"{rule}/comp"]
        ax = fig.add_subplot(gs[r0, 0])
        im = ax.imshow(Jk, cmap="viridis", vmin=-1, vmax=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step 5, {name.split()[-1]}", fontsize=7.5, color=col)
        sub = gs[r0, 1].subgridspec(2, 1, hspace=0.12)
        raster(fig.add_subplot(sub[0]), d[f"{rule}/raster/{LO:g}"], comp, name, "no stim.")
        raster(fig.add_subplot(sub[1]), d[f"{rule}/raster/{HI:g}"], comp, None, f"stim. {HI:g}")
        ax = fig.add_subplot(gs[r0, 2])
        Is = d[f"{rule}/Is"]
        ax.plot(Is, d[f"{rule}/p3"], "o-", color=P3, ms=2.5, lw=1.4, label="2 $\\to$ 3")
        ax.plot(Is, d[f"{rule}/p4"], "s-", color=P4, ms=2.5, lw=1.4, label="2 $\\to$ 4")
        ax.axvline(HI, color="0.6", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("transition prob.")
        if r0 == 3:
            ax.set_xlabel("stimulus intensity")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.text(0.01, 0.6, "d", fontsize=11, weight="bold")
    fig.text(0.27, 0.6, "e", fontsize=11, weight="bold")
    fig.text(0.77, 0.6, "f", fontsize=11, weight="bold")
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig3_N1024.png", dpi=250)
    fig.savefig("Figures/Fig3_N1024.pdf")
    print("saved Figures/Fig3_N1024.png / .pdf")


if __name__ == "__main__":
    main()
