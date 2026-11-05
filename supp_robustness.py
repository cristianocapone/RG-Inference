"""
Supplementary figure: robustness of the data results.
 (a,b) specific predictability advantage of inference over correlation (mean over RG
       steps 2-7) for 9 cultures (a) and 3 mice (b), for the 512 most active or 512
       random units and bins of 5, 10, 20 ms             [results/robustness.npz]
 (c)   excess symmetry of the inferred couplings at step 6 for the three rules, same
       configurations                                    [results/robustness.npz]
 (d)   single Neuropixels probes (local populations): directedness vs closure
       advantage, 10 ms bins                              [results/invivo_local.npz]
Output: Figures/Supp_robustness.png / .pdf.  Run from the folder holding results/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
MICE = ["Krebs", "Waksman", "Robbins"]
CONF = [(m, dt) for dt in (5, 10, 20) for m in ("top", "random")]
COL = {"inference": "#d62728", "correlation": "#8c8c00", "random": "#4a7fd6"}


def main():
    st = np.load("results/robustness.npz")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1]))
    x = np.arange(len(CONF))
    labels = [f"{m}\n{dt} ms" for m, dt in CONF]
    for ax, grp, names, title in [(axs[0], "cult", CULT, "(a) cultures (n = 9)"), (axs[1], "mouse", MICE, "(b) mice (n = 3)")]:
        vals = []
        for m, dt in CONF:
            d = np.array([np.nanmean((st[f"{grp}/{m}/{dt}/{n}/inference/spec"] - st[f"{grp}/{m}/{dt}/{n}/correlation/spec"])[1:7])
                          for n in names])
            vals.append(d)
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=12, color="0.5", zorder=2)
            ax.bar(i, v.mean(), 0.6, color="#d62728", alpha=0.6, zorder=1)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("specific predictability,\ninference $-$ correlation")
        ax.set_title(title, fontsize=10, loc="left")
    ax = axs[2]
    w = 0.13
    for j, (grp, names, mk) in enumerate([("cult", CULT, "o"), ("mouse", MICE, "s")]):
        for i, rule in enumerate(["inference", "correlation", "random"]):
            y = [np.nanmean([st[f"{grp}/{m}/{dt}/{n}/{rule}/xsym"][5] for n in names]) for m, dt in CONF]
            ax.plot(x + (i - 1) * w + (j - 0.5) * 0.05, y, mk, color=COL[rule], ms=5 if j == 0 else 4,
                    label=f"{rule} ({'cultures' if j == 0 else 'mice'})")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("excess symmetry of $J$ at step 6")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.set_title("(c) symmetry vs surrogate", fontsize=10, loc="left")
    ax = axs[3]
    lo = np.load("results/invivo_local.npz")
    for prefix, mk, lab in [("10ms/", "o", "single probe"), ("10ms/", "s", "pooled 512")]:
        ks = [k for k in lo.files if k.startswith(prefix) and (("probe" in k) == (lab == "single probe"))]
        ax.scatter([lo[k][4] for k in ks], [lo[k][2] for k in ks], marker=mk, s=25 if mk == "o" else 45,
                   color="0.3" if mk == "o" else "#1f77b4", label=lab)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("directedness at lag 1 (noise-corrected)")
    ax.set_ylabel("closure advantage (output) $-$ correlation")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(d) mice: single probes", fontsize=10, loc="left")
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_robustness.png", dpi=180)
    fig.savefig("Figures/Supp_robustness.pdf")
    print("saved Figures/Supp_robustness.png / .pdf")


if __name__ == "__main__":
    main()
