"""
Does partial observation explain why, in vivo, the advantage of inference-based
coarse-graining does not grow with directedness?

Spiking E/I networks with known connectivity (N = 2048: clustered, chain 0.1, chain 0.2;
axonal delays, 10 ms bins, 600 s; spiking_ei_rg.py, run tag d5j0.5_s0.15_Nref3000_wclu1.9).
We observe random subsets of n neurons (E and I in proportion, 5 random subsets per n)
and compute, per subset:
  directedness  1 - symmetry of the lagged covariance D(1) (model-free)
  advantage     closure advantage of inference (in+out, output-only) over correlation,
                mean over RG steps 2..L-1
  coupling fidelity  Pearson correlation between the inferred couplings among the
                observed neurons and their true synaptic weights
  advantage at steps 2-3 only (same RG depth for every n, controls for the shorter
                trajectories of small subsets)
as a function of the observed fraction n / 2048.
Results: results/partial_observation.npz; figure Figures/Partial_observation.png.
Run from the folder holding sim_cache/ and results/:
    python partial_observation.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spiking_ei_rg as spk
from repro_figures_v2 import infer_J
from closure_test import rg_closure
from rg_metrics import lagged_cov

TAG = "d5j0.5_s0.15_Nref3000_wclu1.9"
NETS = {"clustered": 0.0, "chain0.1": 0.1, "chain0.2": 0.2}
FILES = {"clustered": "clustered", "chain0.1": "chain", "chain0.2": "chain0.2"}
SIZES = [512, 256, 128, 64, 32]
SEEDS = range(5)
RULES = ["inference", "inference_out", "correlation"]


def subset(rng, n):
    nE = int(round(n * spk.NE / spk.N))
    return np.sort(np.r_[rng.choice(spk.NE, nE, replace=False), spk.NE + rng.choice(spk.NI, n - nE, replace=False)])


def evaluate(B, W):
    S = B.astype(np.float32)
    _, D = lagged_cov(S)
    direct = 1 - spk.symmetry(D)
    J = infer_J(S, 0)
    off = ~np.eye(len(W), dtype=bool)
    fid = np.corrcoef(J[off], W[off])[0, 1]
    L = int(np.log2(S.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    adv = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    # same depth for every n: RG steps 2-3 only (removes the confound of shallower trajectories)
    adv23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:2]]
    return np.r_[direct, adv, fid, adv23]


def main():
    os.makedirs("results", exist_ok=True)
    store = {}
    for name, chain in NETS.items():
        d = np.load(f"sim_cache/{FILES[name]}_{TAG}_spikes_T600.npz")
        B_all = spk.bin_spikes(d["ts"], d["ids"], 10.0)
        W_all = d["W"]
        for n in SIZES:
            t0 = time.time()
            vals = []
            for s in SEEDS:
                idx = subset(np.random.default_rng(1000 * n + s), n)
                vals.append(evaluate(B_all[:, idx], W_all[np.ix_(idx, idx)]))
            vals = np.array(vals)
            store[f"{name}/{n}"] = vals
            m = vals.mean(0)
            print(f"{name:10s} n={n:4d} ({100 * n / spk.N:4.1f}%): directedness {m[0]:.3f}  adv inference {m[1]:+.3f}  "
                  f"output {m[2]:+.3f}  corr(J,W) {m[3]:.3f}  | steps 2-3 only: inference {m[4]:+.3f} output {m[5]:+.3f}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/partial_observation.npz", **store)
    plot(store)


def plot(store):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = {"clustered": "#7f7f7f", "chain0.1": "#f08080", "chain0.2": "#d62728"}
    frac = np.array(SIZES) / spk.N * 100
    for name in NETS:
        A = np.stack([store[f"{name}/{n}"] for n in SIZES])
        for ax, j, lab in [(axs[0], 0, "directedness, 1 - sym D(1)"), (axs[1], 2, "closure advantage (output) vs correlation"),
                           (axs[2], 3, "corr(inferred J, true W)")]:
            ax.errorbar(frac, A[:, :, j].mean(1), A[:, :, j].std(1), fmt="o-", color=cols[name], ms=4, capsize=2, label=name)
            ax.set_ylabel(lab)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("observed fraction of neurons (%)")
    axs[1].axhline(0, color="0.6", ls=":")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Partial_observation.png", dpi=170)
    print("saved Figures/Partial_observation.png")


if __name__ == "__main__":
    main()
