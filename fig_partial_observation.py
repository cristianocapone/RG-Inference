"""
Figure: the advantage of inference-based coarse-graining requires observing a
sufficient part of the circuit that generates the directional dynamics.

 (a) spiking E/I networks with known connectivity (2048 neurons): directedness of the
     observed activity (dashed) is independent of the observed fraction, the closure
     advantage of inference (output similarity) over correlation (solid) vanishes
     below a few percent                         [results/partial_observation.npz]
 (b) cultures: same analysis on random subsets of electrodes, one line per culture
     (colour = directedness of the culture)      [results/partial_observation_cultures.npz]
 (c) Stringer mice, spontaneous: advantage vs number of observed units, 64 to 2048
     random units                                [results/invivo_density.npz]
 (d) Allen Brain Observatory, final pipeline: visual stimuli vs spontaneous activity,
     change of directedness and of the advantages  [results/allen_final/*.npz]
 (e) CA1 (Grosmark & Buzsaki 2016): directedness and advantage by epoch and sleep state,
     10 and 20 ms                                 [results/hc11/*.npz]
 (f) synthesis at 10 ms: advantage of the inference-based reduction (output) against
     directedness for every recording and condition: cultures, Stringer mice, Allen
     condition-session pairs, CA1 epochs and states
Directedness: noise-corrected lag-1 index (directedness.py).  The model-free rule
(columns of the lag-1 cross-correlation, "lagged_out") is shown dashed in (a) and (c).
The Allen Brain Observatory panels (old pipeline) are in supp_allen.py.
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
        ax.plot(frac, A[:, :, 6].mean(1), "s--", color=col, ms=3, lw=1, alpha=0.7)
        ax2.plot(frac, A[:, :, 8].mean(1), ":", color=col, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(frac)
    ax.set_xticklabels([f"{f:g}" for f in np.round(frac, 1)])
    ax.minorticks_off()
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed fraction of neurons (%)")
    ax.set_ylabel("advantage over correlation\n(inference solid, lagged dashed)")
    ax2.set_ylabel("directedness, lag 1 (dotted)", color="0.4")
    ax2.set_ylim(0, 0.06)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(a) spiking networks, ground truth", loc="left", fontsize=10)


def panel_cultures(ax):
    d = np.load("results/partial_observation_cultures.npz")
    cults = sorted({k.split("/")[0] for k in d.files}, key=lambda c: int(c[1:]))
    dirs = {c: np.mean([d[k][:, 7].mean() for k in d.files if k.startswith(c + "/")]) for c in cults}
    norm = colors.Normalize(0.0, 0.3)
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
    plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, label="directedness of the culture (lag 1)")
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
        ax.plot(ns, A[:, :, 5].mean(1), "s--", color=cols.get(m, "0.4"), ms=2.5, lw=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed units")
    ax.set_ylabel("advantage of inference over correlation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(c) mice in vivo, spontaneous: more units", loc="left", fontsize=10)


def panel_allen_density(ax):
    rows = [r for r in (dict(np.load(f)) for f in sorted(glob.glob("results/allen_density/*.npz"))) if "skipped" not in r]
    sizes = [64, 128, 256, 512]
    for c, col, lab in [("spontaneous", "0.5", "spontaneous"), ("natural_movie_three", "#f4a261", "natural movie"),
                        ("drifting_gratings", "#d62728", "drifting gratings")]:
        A = np.array([[r[f"{c}/{n}"][2] for n in sizes] for r in rows])
        ax.errorbar(sizes, A.mean(0), A.std(0) / np.sqrt(len(A)), fmt="o-", color=col, ms=4, capsize=2, lw=1.6,
                    label=f"{lab} (dir. {np.mean([r[f'{c}/{n}'][0] for r in rows for n in sizes]):.2f})")
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="0.7", ls=":")
    ax.set_xlabel("observed units")
    ax.set_ylabel("advantage of inference over correlation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"(d) mice, Allen: more units (n = {len(rows)})", loc="left", fontsize=10)


def panel_allen_final(ax):
    import fig_allen_final as FA
    FA.panel_stimuli(ax, FA.rows())
    ax.set_title(ax.get_title(loc="left").replace("(a)", "(d) mice, Allen:"), loc="left", fontsize=10)


def panel_hc11(ax):
    rows = [dict(np.load(f)) for f in sorted(glob.glob("results/hc11/*.npz"))]
    keys = [("PREEpoch", "PRE sleep"), ("MazeEpoch", "maze"), ("POSTEpoch", "POST sleep"), ("state_Non-REM", "NREM"), ("state_REM", "REM")]
    x = np.arange(len(keys))
    for j, (dt, alpha) in enumerate([(10, 1.0), (20, 0.45)]):
        d = [np.array([r[f"{k}/{dt}"][0] for r in rows if f"{k}/{dt}" in r]) for k, _ in keys]
        a = [np.array([r[f"{k}/{dt}"][2] for r in rows if f"{k}/{dt}" in r]) for k, _ in keys]
        ax.bar(x + (j - 0.5) * 0.4 - 0.1, [v.mean() for v in d], 0.18, yerr=[v.std() / np.sqrt(len(v)) for v in d], color="0.35", alpha=alpha,
               capsize=2, label=f"directedness, {dt} ms")
        ax.bar(x + (j - 0.5) * 0.4 + 0.1, [v.mean() for v in a], 0.18, yerr=[v.std() / np.sqrt(len(v)) for v in a], color="#8b0000", alpha=alpha,
               capsize=2, label=f"advantage of inference (output), {dt} ms")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(f"(e) rat CA1, 8 sessions, 26-102 units", loc="left", fontsize=10)


def panel_synthesis(ax):
    from scipy.stats import spearmanr
    cl, dr = np.load("results/closure.npz"), np.load("results/directedness.npz")
    cults = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
    xc = [float(dr[f"cultures/{c}/lag1"]) for c in cults]
    yc = [np.nanmean((cl[f"cultures/{c}/inference_out/closure"] - cl[f"cultures/{c}/correlation/closure"])[1:7]) for c in cults]
    ax.scatter(xc, yc, s=50, color="#8b0000", label="cultures (9)", zorder=4)
    xm = [float(dr[f"mice/{m}/lag1"]) for m in ["Krebs", "Waksman", "Robbins"]]
    ym = [np.nanmean((cl[f"mice/{m}/inference_out/closure"] - cl[f"mice/{m}/correlation/closure"])[1:7]) for m in ["Krebs", "Waksman", "Robbins"]]
    ax.scatter(xm, ym, s=45, marker="s", color="#1f77b4", label="mice, 8 probes, spontaneous (3)", zorder=4)
    al = [dict(np.load(f)) for f in sorted(glob.glob("results/allen_final/*.npz"))]
    xa = [r[k][0] for r in al if "skipped" not in r for k in r if k.startswith("top/")]
    ya = [r[k][2] for r in al if "skipped" not in r for k in r if k.startswith("top/")]
    ax.scatter(xa, ya, s=12, marker="^", color="#6baed6", alpha=0.7, label=f"mice, visual stimuli and rest ({len(xa)})", zorder=3)
    hc = [dict(np.load(f)) for f in sorted(glob.glob("results/hc11/*.npz"))]
    xh = [r[k][0] for r in hc for k in r if k.endswith("/10")]
    yh = [r[k][2] for r in hc for k in r if k.endswith("/10")]
    ax.scatter(xh, yh, s=14, marker="D", color="#2a9d8f", alpha=0.8, label=f"rat CA1, epochs and states ({len(xh)})", zorder=3)
    xv, yv = np.r_[xm, xa, xh], np.r_[ym, ya, yh]
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("directedness at lag 1 (noise-corrected), 10-ms bins")
    ax.set_ylabel("advantage of inference (output) over correlation")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title(f"(f) all recordings: in vivo rho = {spearmanr(xv, yv)[0]:.2f} (n = {len(xv)})", loc="left", fontsize=10)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw=dict(width_ratios=[1, 1.1, 0.9]))
    panel_sim(axs[0, 0])
    panel_cultures(axs[0, 1])
    panel_density(axs[0, 2])
    panel_allen_final(axs[1, 0])
    panel_hc11(axs[1, 1])
    panel_synthesis(axs[1, 2])
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig_partial_observation.png", dpi=200)
    fig.savefig("Figures/Fig_partial_observation.pdf")
    print("saved Figures/Fig_partial_observation.png / .pdf")


if __name__ == "__main__":
    main()
