"""
Supplementary figure: closure of the coarse-grained dynamics at matched reduced dimension
for the grouping rules and for reductions that do not produce groups (PCA, reduced-rank
regression, NMF), from results/dimred_comparison.npz: cultures (mean +/- s.e.m. over 9),
mice (3) and spiking networks (2).  Output: Figures/Supp_dimred.png / .pdf.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = {"cultures": ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"], "mice": ["Krebs", "Waksman", "Robbins"], "spiking": ["clustered", "chain"]}
METH = [("inference_out", "inference (output)", "#8b0000", "o-"), ("inference", "inference (in+out)", "#d62728", "o-"),
        ("lagged_out", "lagged correlation (output)", "#5e3c99", "o-"), ("correlation", "correlation", "#8c8c00", "o-"),
        ("nmf", "NMF", "#2a9d8f", "s--"), ("pca", "PCA", "#7f7f7f", "s--"), ("rrr", "reduced-rank regression", "#1f77b4", "s--"),
        ("random", "random", "#4a7fd6", "o:")]


def main():
    d = np.load("results/dimred_comparison.npz")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (grp, names) in zip(axs, SETS.items()):
        for key, lab, col, fmt in METH:
            A = np.array([[d[f"{grp}/{n}/{lev}/{key}"] for lev in (3, 4, 5)] for n in names if f"{grp}/{n}/3/{key}" in d.files])
            if not len(A):
                continue
            ax.errorbar([64, 32, 16], A.mean(0), A.std(0) / np.sqrt(len(A)), fmt=fmt, color=col, ms=4, capsize=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 32, 16])
        ax.set_xticklabels(["64", "32", "16"])
        ax.set_xlabel("reduced dimension (blocks or components)")
        ax.set_ylabel("closure, held-out")
        ax.set_title(f"({'abc'[list(SETS).index(grp)]}) {grp} (n = {len(names)})", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_dimred.png", dpi=180)
    fig.savefig("Figures/Supp_dimred.pdf")
    print("saved Figures/Supp_dimred.png / .pdf")


if __name__ == "__main__":
    main()
