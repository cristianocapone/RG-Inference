"""
Figure from allen_final.py (final pipeline, streamed Allen Brain Observatory sessions):
 (a) rest vs stimuli, 256 units: change of the noise-corrected lag-1 directedness and of
     the closure advantage of the inference-based (output) and model-free lagged rules
     over correlation, stimulus minus spontaneous (mean +/- s.e.m. over sessions)
 (b) all condition-session pairs: advantage of the inference-based reduction (output)
     against directedness, coloured by condition
 (c) number of observed units: advantage vs subset size for spontaneous activity, drifting
     gratings and natural movie (inference solid, lagged dashed)
Output: Figures/Fig_allen_final.png / .pdf.  Run from the folder holding results/.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

STIM = ["natural_movie_three", "natural_movie_one", "natural_scenes", "drifting_gratings", "static_gratings", "gabors"]
LAB = {"spontaneous": "spontaneous", "natural_movie_three": "movie 3", "natural_movie_one": "movie 1", "natural_scenes": "scenes",
       "drifting_gratings": "drifting gratings", "static_gratings": "static gratings", "gabors": "gabors"}
CCOL = {"spontaneous": "0.4", "natural_movie_three": "#f4a261", "natural_movie_one": "#e9c46a", "natural_scenes": "#2a9d8f",
        "drifting_gratings": "#d62728", "static_gratings": "#e76f51", "gabors": "#8e9aaf"}


def rows():
    r = [dict(np.load(f)) for f in sorted(glob.glob("results/allen_final/*.npz"))]
    return [x for x in r if "skipped" not in x]


def panel_stimuli(ax, R):
    stim = [c for c in STIM if sum(f"top/{c}" in r for r in R) >= 6]
    x = np.arange(len(stim))
    for j, (col, lab, color) in enumerate([(0, "directedness (lag 1)", "0.35"), (2, "advantage of inference (output)", "#8b0000"),
                                           (3, "advantage of lagged correlation", "#5e3c99")]):
        d = [np.array([r[f"top/{c}"][col] - r["top/spontaneous"][col] for r in R if f"top/{c}" in r]) for c in stim]
        ax.bar(x + (j - 1) * 0.27, [v.mean() for v in d], 0.25, yerr=[v.std() / np.sqrt(len(v)) for v in d], color=color, capsize=2, label=lab)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[c] for c in stim], fontsize=7, rotation=20)
    ax.set_ylabel("stimulus minus spontaneous")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"(a) visual stimuli vs spontaneous activity (n = {len(R)})", loc="left", fontsize=10)


def panel_scatter(ax, R):
    xs, ys = [], []
    for c in ["spontaneous"] + STIM:
        x = [r[f"top/{c}"][0] for r in R if f"top/{c}" in r]
        y = [r[f"top/{c}"][2] for r in R if f"top/{c}" in r]
        ax.scatter(x, y, s=14, color=CCOL[c], label=LAB[c], alpha=0.8)
        xs += x
        ys += y
    rho, p = spearmanr(xs, ys)
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("directedness, lag 1 (noise-corrected)")
    ax.set_ylabel("advantage of inference (output) over correlation")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_title(f"(b) all pairs (n = {len(xs)}): rho = {rho:.2f}, p = {p:.2g}", loc="left", fontsize=10)


def panel_density(ax, R):
    sizes = [64, 128, 256, 512]
    for c in ["spontaneous", "natural_movie_three", "drifting_gratings"]:
        A = np.array([[r[f"dens/{c}/{n}"] for n in sizes] for r in R if all(f"dens/{c}/{n}" in r for n in sizes)])
        if not len(A):
            continue
        ax.errorbar(sizes, A[:, :, 2].mean(0), A[:, :, 2].std(0) / np.sqrt(len(A)), fmt="o-", color=CCOL[c], ms=4, capsize=2, lw=1.6,
                    label=f"{LAB[c]} (n = {len(A)}, dir. {A[:, :, 0].mean():.3f})")
        ax.plot(sizes, A[:, :, 3].mean(0), "s--", color=CCOL[c], ms=3, lw=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed units")
    ax.set_ylabel("advantage over correlation\n(inference solid, lagged dashed)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(c) more units, by condition", loc="left", fontsize=10)


def main():
    R = rows()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1, 0.95]))
    panel_stimuli(axs[0], R)
    panel_scatter(axs[1], R)
    panel_density(axs[2], R)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig_allen_final.png", dpi=180)
    fig.savefig("Figures/Fig_allen_final.pdf")
    print(f"saved Figures/Fig_allen_final.png / .pdf ({len(R)} sessions)")


if __name__ == "__main__":
    main()
