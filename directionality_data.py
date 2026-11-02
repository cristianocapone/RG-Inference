"""
Data-side test of the core claim "inference-based RG reduces dimensionality better
when the dynamics is directional": across recordings (9 cultures, 3 mice), the
closure advantage of inference-RG over correlation-RG (mean over steps 2-7, from
results/closure.npz) is plotted against the directedness of the recording, measured
model-free as 1 - symmetry of the lagged covariance D(1) at level 0 (from
results/cultures_metrics.npz, results/stringer_metrics.npz).
Figure: Figures/Directionality_data.png

Run:  env -i /opt/homebrew/bin/python3 directionality_data.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def main():
    cl = np.load("results/closure.npz")
    met = {"cultures": np.load("results/cultures_metrics.npz"), "mice": np.load("results/stringer_metrics.npz")}
    names = {"cultures": ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"], "mice": ["Krebs", "Waksman", "Robbins"]}
    rows = []
    for grp in names:
        for n in names[grp]:
            direct = 1 - met[grp][f"{n}/inference/real/sym_D"][0]
            adv = {r: np.nanmean((cl[f"{grp}/{n}/{r}/closure"] - cl[f"{grp}/{n}/correlation/closure"])[1:7])
                   for r in ["inference", "inference_out"]}
            rows.append((grp, n, direct, adv["inference"], adv["inference_out"]))
    x = np.array([r[2] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, j, lab in [(axs[0], 3, "inference (in+out)"), (axs[1], 4, "inference (output only)")]:
        y = np.array([r[j] for r in rows])
        for grp, mk in [("cultures", "o"), ("mice", "s")]:
            m = np.array([r[0] == grp for r in rows])
            ax.scatter(x[m], y[m], marker=mk, s=50, color="#d62728" if grp == "cultures" else "#1f77b4", label=grp)
        for r in rows:
            ax.annotate(r[1], (r[2], r[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho, p = spearmanr(x, y)
        rp, pp = pearsonr(x, y)
        ax.axhline(0, color="0.6", ls=":")
        ax.set_xlabel("directedness of the recording: 1 - symmetry of D(1)")
        ax.set_ylabel(f"closure advantage, {lab} - correlation")
        ax.set_title(f"Spearman rho = {rho:.2f} (p = {p:.3f}), Pearson r = {rp:.2f}", fontsize=9)
        print(f"{lab}: spearman rho={rho:.2f} p={p:.4f}  pearson r={rp:.2f} p={pp:.4f}")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("Figures/Directionality_data.png", dpi=170)
    print("saved Figures/Directionality_data.png")


if __name__ == "__main__":
    main()
