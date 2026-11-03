"""
Figure: the advantage of inference-based coarse-graining requires observing a
sufficient part of the circuit that generates the directional dynamics.

 (a) spiking E/I networks with known connectivity (2048 neurons): directedness of the
     observed activity (dashed) is independent of the observed fraction, the closure
     advantage of inference (output similarity) over correlation (solid) vanishes
     below a few percent                         [results/partial_observation.npz]
 (b) cultures: same analysis on random subsets of electrodes, one line per culture
     (colour = directedness of the culture)      [results/partial_observation_cultures.npz]
 (c) Allen Brain Observatory Neuropixels, 32 mice, 256 units out of millions: every
     stimulus makes the activity more directional than spontaneous activity, but the
     advantage does not grow                     [results/allen/*.npz]
 (d) Stringer mice: advantage vs number of observed units, 64 to 2048 random units
                                                 [results/invivo_density.npz]
Output: Figures/Fig_partial_observation.png / .pdf.  Run from the folder holding results/.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

N_NET = 2048
NETS = {"clustered": ("#7f7f7f", "clustered"), "chain0.1": ("#f08080", "chain 0.1"), "chain0.2": ("#d62728", "chain 0.2")}
STIM = ["natural_movie_three", "natural_scenes", "drifting_gratings", "static_gratings", "gabors"]
STIM_LAB = ["movie", "scenes", "drifting\ngratings", "static\ngratings", "gabors"]


def panel_sim(ax):
    d = np.load("results/partial_observation.npz")
    sizes = sorted({int(k.split("/")[1]) for k in d.files}, reverse=True)
    frac = np.array(sizes) / N_NET * 100
    ax2 = ax.twinx()
    for name, (col, lab) in NETS.items():
        A = np.stack([d[f"{name}/{n}"] for n in sizes])
        ax.errorbar(frac, A[:, :, 2].mean(1), A[:, :, 2].std(1), fmt="o-", color=col, ms=4, capsize=2, lw=1.8, label=lab)
        ax2.plot(frac, A[:, :, 0].mean(1), "--", color=col, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(frac)
    ax.set_xticklabels([f"{f:g}" for f in np.round(frac, 1)])
    ax.minorticks_off()
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed fraction of neurons (%)")
    ax.set_ylabel("advantage of inference over correlation\n(closure, solid)")
    ax2.set_ylabel("directedness (dashed)", color="0.4")
    ax2.set_ylim(0, 0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(a) spiking networks, ground truth", loc="left", fontsize=10)


def panel_cultures(ax):
    d = np.load("results/partial_observation_cultures.npz")
    cults = sorted({k.split("/")[0] for k in d.files}, key=lambda c: int(c[1:]))
    dirs = {c: np.mean([d[k][:, 0].mean() for k in d.files if k.startswith(c + "/")]) for c in cults}
    norm = colors.Normalize(0.2, 0.75)
    cmap = matplotlib.colormaps["viridis"]
    for c in cults:
        ns = sorted(int(k.split("/")[1]) for k in d.files if k.startswith(c + "/"))
        y = [d[f"{c}/{n}"][:, 2].mean() for n in ns]
        ax.plot(ns, y, "o-", color=cmap(norm(dirs[c])), ms=3, lw=1.5)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed electrodes")
    ax.set_ylabel("advantage of inference over correlation")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, label="directedness of the culture")
    ax.set_title("(b) cultures, random subsets of electrodes", loc="left", fontsize=10)


def panel_allen(ax):
    rows = []
    for f in sorted(glob.glob("results/allen/*.npz")):
        r = np.load(f)
        if "skipped" not in r.files:
            rows.append(r)
    x = np.arange(len(STIM))
    dd = [np.array([r[c][0] - r["spontaneous"][0] for r in rows if c in r.files]) for c in STIM]
    da = [np.array([r[c][2] - r["spontaneous"][2] for r in rows if c in r.files]) for c in STIM]
    ax.bar(x - 0.2, [v.mean() for v in dd], 0.38, yerr=[v.std() / np.sqrt(len(v)) for v in dd], color="0.35",
           capsize=2, label="directedness")
    ax.bar(x + 0.2, [v.mean() for v in da], 0.38, yerr=[v.std() / np.sqrt(len(v)) for v in da], color="#d62728",
           capsize=2, label="advantage of inference (output)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STIM_LAB, fontsize=8)
    ax.set_ylabel("stimulus minus spontaneous")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"(c) mice in vivo, Neuropixels (n = {len(rows)})", loc="left", fontsize=10)


def panel_density(ax):
    d = np.load("results/invivo_density.npz")
    mice = sorted({k.split("/")[0] for k in d.files})
    cols = {"Krebs": "#1f77b4", "Waksman": "#6baed6", "Robbins": "#08306b"}
    for m in mice:
        ns = sorted(int(k.split("/")[1]) for k in d.files if k.startswith(m + "/"))
        A = np.stack([d[f"{m}/{n}"] for n in ns])
        ax.errorbar(ns, A[:, :, 2].mean(1), A[:, :, 2].std(1), fmt="o-", color=cols.get(m, "0.4"), ms=3,
                    capsize=2, lw=1.4, label=m)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed units")
    ax.set_ylabel("advantage of inference over correlation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(d) mice, spontaneous: more units", loc="left", fontsize=10)


def main():
    fig, axs = plt.subplots(1, 4, figsize=(19.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.05, 1, 0.85]))
    panel_sim(axs[0])
    panel_cultures(axs[1])
    panel_allen(axs[2])
    panel_density(axs[3])
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig_partial_observation.png", dpi=200)
    fig.savefig("Figures/Fig_partial_observation.pdf")
    print("saved Figures/Fig_partial_observation.png / .pdf")


if __name__ == "__main__":
    main()
