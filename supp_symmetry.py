"""
Supplementary figure: symmetry of the inferred couplings along the RG, with the
time-shifted surrogate as baseline.  Replaces the earlier figure on the culture of
Nieus et al. 2015 (not public) with the public data used in the paper.

Columns: (A) spiking E/I network with directed chain and axonal delays (ground truth,
10 ms bins, 512 units; symmetry of the true couplings coarse-grained along the same
pairing shown in black), (B) 9 cultures (Lonardoni 2017), (C) 3 mice (Stringer 2019).
Top row: symmetry index of the inferred J (solid: data, dashed: shifted surrogate).
Bottom row: excess symmetry, data minus surrogate.

Inputs: results/cultures_metrics.npz (excess_symmetry_cultures.py),
results/stringer_metrics.npz (invivo_stringer.py), sim_cache spikes of the delayed
chain network (spiking_ei_rg.py 5 0.5 0.15 3000 1.9).  The spiking part is computed
here and cached in results/supsym_spiking.npz.
Output: Figures/Supp_symmetry.png / .pdf

Run:  python supp_symmetry.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spiking_ei_rg as spk
from rg_metrics import data_vs_surrogate, RULES, COLORS
from repro_figures_v2 import infer_J
from rg_metrics import pairs_for

CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
MICE = ["Krebs", "Waksman", "Robbins"]
TAG = "d5j0.5_s0.15_Nref3000_wclu1.9"
L_SPK = 9
L_SHOW = 7          # spiking panel stops at 8 blocks (symmetry is meaningless with fewer)


def true_symmetry_along_rg(B, W, rule, n_levels, seed=0):
    """Symmetry of the true couplings summed over the blocks of the same pairing."""
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    M = np.eye(S.shape[1], dtype=np.float32)
    out = []
    for k in range(n_levels):
        out.append(spk.symmetry(M.T @ W @ M))
        if k == n_levels - 1:
            break
        J = infer_J(S, k)
        p = pairs_for(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return np.array(out)


def spiking_results(path="results/supsym_spiking.npz"):
    if os.path.exists(path):
        return dict(np.load(path))
    d = np.load(f"sim_cache/chain_{TAG}_spikes_T600.npz")
    B = spk.bin_spikes(d["ts"], d["ids"], 10.0)
    r = np.random.default_rng(0)
    sub = np.sort(np.r_[r.choice(spk.NE, 400, replace=False), spk.NE + r.choice(spk.NI, 112, replace=False)])
    B = B[:, sub]
    W = d["W"][np.ix_(sub, sub)]
    res = data_vs_surrogate(B, L_SPK)
    out = {}
    for rule in RULES:
        out[f"{rule}/real"] = res[rule]["real"]["sym_J"]
        out[f"{rule}/surr"] = res[rule]["surr"]["sym_J"]
        out[f"{rule}/true"] = true_symmetry_along_rg(B, W, rule, L_SPK)
    os.makedirs("results", exist_ok=True)
    np.savez(path, **out)
    return out


def msem(A):
    n = np.sum(~np.isnan(A), 0)
    return np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1))


def main():
    sp = spiking_results()
    cm = np.load("results/cultures_metrics.npz")
    sm = np.load("results/stringer_metrics.npz")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for rule in RULES:
        c = COLORS[rule]
        # (A) spiking
        n = L_SHOW
        x = np.arange(1, n + 1)
        axs[0, 0].plot(x, sp[f"{rule}/real"][:n], "o-", color=c, ms=3, label=rule)
        axs[0, 0].plot(x, sp[f"{rule}/surr"][:n], "--", color=c, lw=1)
        axs[0, 0].plot(x, sp[f"{rule}/true"][:n], ":", color=c, lw=1.5)
        axs[1, 0].plot(x, (sp[f"{rule}/real"] - sp[f"{rule}/surr"])[:n], "o-", color=c, ms=3)
        # (B) cultures, (C) mice
        for col, (store, names) in [(1, (cm, CULT)), (2, (sm, MICE))]:
            R = np.stack([store[f"{n}/{rule}/real/sym_J"] for n in names])
            Sg = np.stack([store[f"{n}/{rule}/surr/sym_J"] for n in names])
            x = np.arange(1, R.shape[1] + 1)
            y, e = msem(R)
            axs[0, col].errorbar(x, y, e, fmt="o-", color=c, ms=3, capsize=2, label=rule)
            axs[0, col].plot(x, np.nanmean(Sg, 0), "--", color=c, lw=1)
            y, e = msem(R - Sg)
            axs[1, col].errorbar(x, y, e, fmt="o-", color=c, ms=3, capsize=2)
    titles = ["(A) spiking E/I network (ground truth)", "(B) cultures, n = 9", "(C) mice, n = 3"]
    for col in range(3):
        axs[0, col].set_title(titles[col], fontsize=10, loc="left")
        axs[0, col].set_ylabel("symmetry of inferred J")
        axs[1, col].set_ylabel("excess symmetry (data - surrogate)")
        axs[1, col].set_xlabel("RG step")
        for row in range(2):
            axs[row, col].axhline(0, color="0.6", ls=":")
    axs[0, 0].plot([], [], "k--", lw=1, label="shifted surrogate")
    axs[0, 0].plot([], [], "k:", lw=1.5, label="true couplings")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_symmetry.png", dpi=170)
    fig.savefig("Figures/Supp_symmetry.pdf")
    print("saved Figures/Supp_symmetry.png / .pdf")


if __name__ == "__main__":
    main()
