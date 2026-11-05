"""
Supplementary figure: in vivo closure advantage vs bin size and coupling estimator
(results/invivo_binsize_*.npz from invivo_binsize.py).
Two panels: Stringer mice (3, spontaneous) and Allen sessions (spontaneous and natural
movie, pooled); x = bin (2, 5, 10 ms), y = closure advantage over the correlation-based
rule (mean +/- s.e.m. over recordings), one line per estimator (nMF, ridge, ML) for the
output rule, plus the in+out rule with nMF and the model-free lagged rule.
Output: Figures/Supp_binsize.png / .pdf and a table on stdout.
Run from the folder holding results/:  python supp_binsize.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

DTS = [2, 5, 10]
SERIES = [("none", "lagged_out", "#5e3c99", "lagged correlation (model-free)", ":"),
          ("nmf", "inference", "#d62728", "inference (in+out), nMF", "-"),
          ("nmf", "inference_out", "#8b0000", "inference (output), nMF", "-"),
          ("ridge", "inference_out", "#e07b39", "inference (output), ridge", "--"),
          ("ml", "inference_out", "#2a9d8f", "inference (output), exact ML", "-.")]


def collect(store, grp):
    recs = sorted({k.rsplit("/dt", 1)[0] for k in store if k.endswith("/dir") and k.startswith(grp)})
    out = {}
    for dt in DTS:
        for est, rule, *_ in SERIES:
            v, d = [], []
            for r in recs:
                k = f"{r}/dt{dt}"
                if f"{k}/{est}/{rule}/closure" in store and f"{k}/none/correlation/closure" in store:
                    c = store[f"{k}/{est}/{rule}/closure"] - store[f"{k}/none/correlation/closure"]
                    L = int(np.sum(~np.isnan(c)))
                    v.append(float(np.nanmean(c[1:max(2, L - 1)])))
                    d.append(float(store[f"{k}/dir"][0]))
            out[(dt, est, rule)] = (np.array(v), np.array(d))
    return out


def main():
    import glob
    store = {}
    for f in sorted(glob.glob("results/invivo_binsize*.npz")):
        store.update(dict(np.load(f)))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, grp, title in [(axs[0], "mice", "Stringer, 3 mice (spontaneous, 256 units)"), (axs[1], "allen", "Allen, sessions x conditions (256 units)")]:
        res = collect(store, grp)
        print(f"\n=== {grp}")
        for est, rule, col, lab, ls in SERIES:
            m, e, xs = [], [], []
            for dt in DTS:
                v, d = res[(dt, est, rule)]
                if len(v):
                    xs.append(dt); m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                    p = wilcoxon(v).pvalue if len(v) >= 5 else np.nan
                    print(f"  {dt:2d} ms {est:5s} {rule:14s} n={len(v):2d} dir {d.mean():.3f}  adv {v.mean():+.4f} ({np.sum(v > 0)}/{len(v)}, p={p:.2g})")
            if xs:
                ax.errorbar(xs, m, e, fmt="o" + ls, color=col, ms=4, capsize=2, label=lab)
        ax.axhline(0, color="0.6", ls=":")
        ax.set_xscale("log"); ax.set_xticks(DTS); ax.set_xticklabels([str(d) for d in DTS]); ax.minorticks_off()
        ax.set_xlabel("bin (ms)"); ax.set_title(title, fontsize=9, loc="left")
    axs[0].set_ylabel("closure advantage over correlation\n(mean over RG steps 2..L-1)")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig("Figures/Supp_binsize.png", dpi=170)
    fig.savefig("Figures/Supp_binsize.pdf")
    print("saved Figures/Supp_binsize.png")


if __name__ == "__main__":
    main()
