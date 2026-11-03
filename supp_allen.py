"""
Supplementary figure: Allen Brain Observatory (Visual Coding - Neuropixels, DANDI 000021),
preliminary analysis computed with the first version of the pipeline (directedness
1 - symmetry of D(1), uncorrected for sampling noise; pairing on the whole recording).
The data are streamed and were not re-analysed with the final pipeline.
 (a) 32 sessions, 256 most active units: change of directedness and of the closure
     advantage of the inference-based reduction (output similarity) over the
     correlation-based one under each visual stimulus, relative to spontaneous activity
     in the same session                                    [results/allen/*.npz]
 (b) first 10 sessions, all good units: closure advantage vs number of observed units
     (64 to 512), spontaneous activity, natural movie and drifting gratings
                                                            [results/allen_density/*.npz]
Output: Figures/Supp_allen.png / .pdf.  Run from the folder holding results/.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STIM = ["natural_movie_three", "natural_scenes", "drifting_gratings", "static_gratings", "gabors"]
STIM_LAB = ["movie", "scenes", "drifting\ngratings", "static\ngratings", "gabors"]


def panel_stimuli(ax):
    rows = [np.load(f) for f in sorted(glob.glob("results/allen/*.npz"))]
    rows = [r for r in rows if "skipped" not in r.files]
    x = np.arange(len(STIM))
    dd = [np.array([r[c][0] - r["spontaneous"][0] for r in rows if c in r.files]) for c in STIM]
    da = [np.array([r[c][2] - r["spontaneous"][2] for r in rows if c in r.files]) for c in STIM]
    ax.bar(x - 0.2, [v.mean() for v in dd], 0.38, yerr=[v.std() / np.sqrt(len(v)) for v in dd], color="0.35",
           capsize=2, label="directedness (uncorrected)")
    ax.bar(x + 0.2, [v.mean() for v in da], 0.38, yerr=[v.std() / np.sqrt(len(v)) for v in da], color="#d62728",
           capsize=2, label="advantage of inference (output)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STIM_LAB, fontsize=8)
    ax.set_ylabel("stimulus minus spontaneous")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"(a) visual stimuli vs spontaneous activity (n = {len(rows)})", loc="left", fontsize=10)


def panel_density(ax):
    rows = [r for r in (dict(np.load(f)) for f in sorted(glob.glob("results/allen_density/*.npz"))) if "skipped" not in r]
    sizes = [64, 128, 256, 512]
    for c, col, lab in [("spontaneous", "0.5", "spontaneous"), ("natural_movie_three", "#f4a261", "natural movie"),
                        ("drifting_gratings", "#d62728", "drifting gratings")]:
        A = np.array([[r[f"{c}/{n}"][2] for n in sizes] for r in rows])
        ax.errorbar(sizes, A.mean(0), A.std(0) / np.sqrt(len(A)), fmt="o-", color=col, ms=4, capsize=2, lw=1.6, label=lab)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed units")
    ax.set_ylabel("advantage of inference over correlation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"(b) more units, by condition (n = {len(rows)})", loc="left", fontsize=10)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    panel_stimuli(axs[0])
    panel_density(axs[1])
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_allen.png", dpi=180)
    fig.savefig("Figures/Supp_allen.pdf")
    print("saved Figures/Supp_allen.png / .pdf")


if __name__ == "__main__":
    main()
