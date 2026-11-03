"""
New Figure 4: "inference-based RG reduces dimensionality better when the dynamics is
directional" — six panels built from the saved results of the analysis scripts.

 (a) data: closure advantage of inference-RG over correlation-RG vs directedness of
     the recording (9 cultures + 3 mice)            [directionality_data.py, closure.npz]
 (b) cultures: closure along the RG for the pairing rules  [closure.npz]
 (c) spiking E/I networks with ground truth: closure at 32 blocks vs directedness,
     incl. PCA                                       [dir_sweep_spiking.npz]
 (d) spiking: perturbation transfer of the reduced models vs directedness, with the
     micro-model reference and the split-half ceiling [dir_sweep_spiking.npz]
 (e) iterative vs one-shot: closure at strong reductions, cultures and spiking
     (recomputed here and cached in results/iter_deep.npz)
 (f) specific held-out predictability of the coarse-grained dynamics, cultures and
     mice                                            [cultures_metrics.npz, stringer_metrics.npz]
Output: Figures/Fig4_new.png / .pdf

Run:  env -i /opt/homebrew/bin/python3 fig4_new.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.argv = [sys.argv[0], "40", "0.5", "sweep"]
from directionality_sweep_spiking import membership, NET_KW, SIM, cache_name, CHAINS
from closure_test import closure, RULES5
from real_data_rg import load_culture
import spiking_ei_rg as spk

COL = {"inference": "#d62728", "inference_out": "#8b0000", "inference_in": "#f4a3a3", "oneshot": "#ff7f0e",
       "correlation": "#8c8c00", "random": "#4a7fd6", "pca": "#7f7f7f"}
LAB = {"inference": "inference (in+out)", "inference_out": "inference (output)", "inference_in": "inference (input)",
       "oneshot": "one-shot (no re-inference)", "correlation": "correlation", "random": "random", "pca": "PCA"}
CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
MICE = ["Krebs", "Waksman", "Robbins"]


def msem(A):
    A = np.asarray(A, float)
    n = np.sum(~np.isnan(A), 0)
    return np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1))


def pad(a, L=8):
    return np.pad(np.asarray(a, float), (0, L - len(a)), constant_values=np.nan)


def iter_deep(path="results/iter_deep.npz"):
    if os.path.exists(path):
        return dict(np.load(path))
    out = {}
    sets = []
    for c in (0.2, 0.0):
        ts, ids, W, clu = spk.get_spikes(cache_name(c), c, 600, SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
        B = spk.bin_spikes(ts, ids, 10.0)
        r = np.random.default_rng(0)
        sub = np.sort(np.r_[r.choice(spk.NE, 400, replace=False), spk.NE + r.choice(spk.NI, 112, replace=False)])
        sets.append((f"spiking/chain{c:g}", B[:, sub].astype(np.float32)))
    for cu in CULT:
        B_all, _, _ = load_culture(cu, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        sets.append((f"cultures/{cu}", B_all[:, :N].astype(np.float32)))
    for name, S0 in sets:
        for rule in ["inference", "inference_out", "oneshot", "correlation", "random"]:
            v = []
            for level in range(1, 8):
                if S0.shape[1] // 2 ** level < 2:
                    break
                M = membership(S0, rule, level)
                r2c, r2m = closure(S0, M)
                v.append(r2c / r2m)
            out[f"{name}/{rule}"] = pad(v, 7)
            print(f"  {name} {rule}: {np.round(v, 3)}", flush=True)
    np.savez(path, **out)
    return out


def main():
    cl = np.load("results/closure.npz")
    cm = {"cultures": np.load("results/cultures_metrics.npz"), "mice": np.load("results/stringer_metrics.npz")}
    sw = np.load("results/dir_sweep_spiking.npz")
    deep = iter_deep()

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    # ---------------- (a) data: advantage vs directedness ----------------
    ax = axs[0, 0]
    xs, ys, gs, names = [], [], [], []
    for grp, lst in [("cultures", CULT), ("mice", MICE)]:
        for n in lst:
            xs.append(1 - cm[grp][f"{n}/inference/real/sym_D"][0])
            ys.append(np.nanmean((cl[f"{grp}/{n}/inference/closure"] - cl[f"{grp}/{n}/correlation/closure"])[1:7]))
            gs.append(grp)
            names.append(n)
    xs, ys = np.array(xs), np.array(ys)
    for grp, mk, c in [("cultures", "o", "#d62728"), ("mice", "s", "#1f77b4")]:
        m = np.array([g == grp for g in gs])
        ax.scatter(xs[m], ys[m], marker=mk, s=55, color=c, label=grp, zorder=3)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    rho, p = spearmanr(xs, ys)
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("directedness of the recording, 1 - symmetry of D(1)")
    ax.set_ylabel("closure advantage\ninference-RG - correlation-RG")
    ax.set_title(f"(a) data: Spearman rho = {rho:.2f}, p = {p:.4f}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # ---------------- (b) cultures: closure along RG ----------------
    ax = axs[0, 1]
    steps = np.arange(1, 9)
    for rule in ["inference", "inference_out", "inference_in", "correlation", "random"]:
        y, e = msem([pad(cl[f"cultures/{n}/{rule}/closure"]) for n in CULT])
        ax.errorbar(steps, y, e, fmt="o-", color=COL[rule], ms=3, capsize=2, label=LAB[rule])
    ax.axhline(1, color="0.6", ls=":")
    ax.set_xlabel("RG step")
    ax.set_ylabel("closure  R2(y|y) / R2(y|x), held-out")
    ax.set_title("(b) cultures, n = 9", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)

    # ---------------- (c) spiking: closure vs directedness ----------------
    ax = axs[0, 2]
    dirx = np.array([sw[f"{c}/R_asym"] for c in CHAINS])
    for rule in ["inference", "inference_out", "oneshot", "correlation", "random"]:
        A = np.stack([sw[f"{c}/{rule}/closure4"] for c in CHAINS])
        ax.errorbar(dirx, A.mean(1), A.std(1), fmt="o-", color=COL[rule], ms=3, capsize=2, label=LAB[rule])
    ax.plot(dirx, [sw[f"{c}/pca/closure4"] for c in CHAINS], "o-", color=COL["pca"], ms=3, label=LAB["pca"])
    ax.axhline(1, color="0.6", ls=":")
    ax.set_xlabel("directedness of the network\n(asymmetry of the response to stimulation)")
    ax.set_ylabel("closure at 32 blocks")
    ax.set_title("(c) spiking E/I networks, ground truth", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)

    # ---------------- (d) spiking: perturbation transfer ----------------
    ax = axs[1, 0]
    for rule in ["inference", "inference_out", "oneshot", "correlation", "random"]:
        A = np.stack([sw[f"{c}/{rule}/pert"] for c in CHAINS])
        ax.errorbar(dirx, A.mean(1), A.std(1), fmt="o-", color=COL[rule], ms=3, capsize=2, label=LAB[rule])
    A = np.stack([sw[f"{c}/micro/pert"] for c in CHAINS])
    ax.errorbar(dirx, A.mean(1), A.std(1), fmt="k--", capsize=2, label="micro model (no reduction)")
    ax.plot(dirx, [sw[f"{c}/ceiling"] for c in CHAINS], "k:", label="split-half ceiling")
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("directedness of the network")
    ax.set_ylabel("perturbation transfer, 64 blocks\ncorr(R_model, R_full), off-diagonal")
    ax.set_title("(d) response to stimulation of the reduced model", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)

    # ---------------- (e) iterative vs one-shot at strong reductions ----------------
    ax = axs[1, 1]
    lv = np.arange(1, 8)
    for rule in ["inference", "inference_out", "oneshot", "correlation", "random"]:
        y, e = msem([deep[f"cultures/{n}/{rule}"] for n in CULT])
        ax.errorbar(lv, y, e, fmt="o-", color=COL[rule], ms=3, capsize=2, label=LAB[rule])
        ys = np.nanmean([deep[f"spiking/{n}/{rule}"] for n in ("chain0.2", "chain0")], 0)
        ax.plot(lv, ys, "--", color=COL[rule], lw=1)
    ax.axhline(1, color="0.6", ls=":")
    ax.set_xlabel("RG step (256 -> 4 blocks)")
    ax.set_ylabel("closure")
    ax.set_title("(e) iterative vs one-shot: 9 cultures (solid), spiking (dashed)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)

    # ---------------- (f) specific predictability ----------------
    ax = axs[1, 2]
    for grp, lst, ls in [("cultures", CULT, "-"), ("mice", MICE, "--")]:
        for rule in ["inference", "correlation", "random"]:
            y, e = msem([cm[grp][f"{n}/{rule}/real/r2"] - cm[grp][f"{n}/{rule}/real/r2_ag"] for n in lst])
            L = min(len(y), 8)
            y, e = y[:L], e[:L]
            ax.errorbar(np.arange(1, L + 1), y, e, fmt="o" + ls, color=COL[rule], ms=3, capsize=2,
                        label=f"{LAB[rule]}, {grp}" if grp == "cultures" else None)
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("RG step")
    ax.set_ylabel("specific predictability  R2 - R2(auto + global)")
    ax.set_title("(f) coarse-grained dynamics: cultures (solid), mice (dashed)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig("Figures/Fig4_new.png", dpi=200)
    fig.savefig("Figures/Fig4_new.pdf")
    print("saved Figures/Fig4_new.png / .pdf")


if __name__ == "__main__":
    main()
