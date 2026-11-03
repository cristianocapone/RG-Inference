"""
Control for Fig. 3: is re-inference at every level needed to keep 2a and 2b separated,
or does hierarchical pairing on the level-0 couplings (inferred once by exact ML and
denoised at the gap rank, as in repro_figure3_ml.py) do as well?
One-shot rules: rows+columns of J0 ("oneshot") and columns only ("oneshot_out"); at each
level blocks are paired by the mean level-0 similarity between their members.
Reported per level: purity and 2a/2b separation (iterative ML inference: 1.00 at every level).
Results: results/fig3_oneshot.npz.  Run from the folder holding sim_cache/.
"""

import time
import numpy as np

import repro_figure3 as F
from repro_figures_v2 import greedy_pairs, synaptic_similarity
from pairing_tests import lowrank_gap, labels_seq
from ml_kinetic_ising import fit_kinetic_ising

F.N = 1024


def main():
    S0 = np.load("sim_cache/seq1024_T100000.npy")
    lab, _ = labels_seq(F.N)
    perm = np.random.default_rng(1).permutation(F.N)
    S, comp0 = S0[:, perm].astype(np.float32), np.eye(7)[lab[perm]]
    t0 = time.time()
    J0 = fit_kinetic_ising(S, nlev=1, l2=1e-3, maxiter=300)[0]
    Jl, r = lowrank_gap(J0)
    print(f"level-0 ML fit done, gap rank {r}  [{time.time() - t0:.0f} s]", flush=True)
    store = {"rank": r}
    for rule in ["oneshot", "oneshot_out"]:
        if rule == "oneshot_out":
            V = Jl.T / (np.linalg.norm(Jl.T, axis=1, keepdims=True) + 1e-12)
            G0 = V @ V.T
        else:
            G0 = synaptic_similarity(Jl)
        M = np.eye(F.N, dtype=np.float32)
        pur, sep = [], []
        for k in range(F.LEVEL):
            Mn = M / M.sum(0)
            G = Mn.T @ G0 @ Mn
            np.fill_diagonal(G, -np.inf)
            p = greedy_pairs(G)
            M = M[:, p[:, 0]] + M[:, p[:, 1]]
            comp = (M / M.sum(0)).T @ comp0
            t = comp[:, 1] + comp[:, 5]
            pur.append(comp.max(1).mean())
            sep.append(np.mean(np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]))
        store[f"{rule}/purity"], store[f"{rule}/separation"] = np.array(pur), np.array(sep)
        print(f"{rule:12s} purity {np.round(pur, 3)}  2a/2b separation {np.round(sep, 2)}", flush=True)
    np.savez("results/fig3_oneshot.npz", **store)


if __name__ == "__main__":
    main()
