"""
Control for Fig. 3: is inference needed, or would a model-free grouping on the lag-1
cross-correlation R(1) separate the two sub-populations 2a and 2b as well?

Sequence network of Fig. 3 (N = 1024, T = 1e5, sim_cache/seq1024_T100000.npy), four RG
levels (1024 -> 64 blocks), rules: Pearson correlation; rows+columns of R(1) ("lagged");
columns only ("lagged_out"); R(1) denoised at the data-driven gap rank or at a fixed rank
4 / 5 (the ranks the inference rule selects), then rows+columns.  Reported per level:
block purity and the 2a/2b separation |c_2a - c_2b| / (c_2a + c_2b) of the blocks
dominated by pattern 2 (1 = perfectly separated), as in repro_figure3_ml.py, where the
ML-inferred couplings give separation 1.00 at every level.
Results: results/fig3_lagged.npz.  Run from the folder holding sim_cache/.
"""

import time
import numpy as np

import repro_figure3 as F
from repro_figures_v2 import greedy_pairs, synaptic_similarity
from pairing_tests import lowrank, lowrank_gap, labels_seq
from closure_test import lagged_corr

F.N = 1024
RULES = ["correlation", "lagged", "lagged_out", "lagged_gaprank", "lagged_rank4", "lagged_rank5"]


def similarity(rule, S):
    if rule == "correlation":
        return np.nan_to_num(np.corrcoef(S.T))
    R = lagged_corr(S)
    np.fill_diagonal(R, 0)
    if rule == "lagged_gaprank":
        R, _ = lowrank_gap(R)
    elif rule.startswith("lagged_rank"):
        R = lowrank(R, int(rule[-1]))
    if rule == "lagged_out":
        V = R.T / (np.linalg.norm(R.T, axis=1, keepdims=True) + 1e-12)
        return V @ V.T
    return synaptic_similarity(R)


def main():
    S0 = np.load("sim_cache/seq1024_T100000.npy")
    lab, _ = labels_seq(F.N)
    store = {}
    for rule in RULES:
        t0 = time.time()
        perm = np.random.default_rng(1).permutation(F.N)
        S, comp = S0[:, perm].astype(np.float32), np.eye(7)[lab[perm]]
        pur, sep = [], []
        for k in range(F.LEVEL):
            G = similarity(rule, S)
            np.fill_diagonal(G, -np.inf)
            p = greedy_pairs(G)
            S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
            comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
            t = comp[:, 1] + comp[:, 5]
            pur.append(comp.max(1).mean())
            sep.append(np.mean(np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]))
        store[f"{rule}/purity"], store[f"{rule}/separation"] = np.array(pur), np.array(sep)
        print(f"{rule:15s} purity {np.round(pur, 3)}  2a/2b separation {np.round(sep, 2)}  [{time.time() - t0:.0f} s]", flush=True)
    np.savez("results/fig3_lagged.npz", **store)


if __name__ == "__main__":
    main()
