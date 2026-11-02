"""
Figure 3 at the paper's stated size, N = 1024, T = 1e5, with exact maximum-likelihood
inference (ml_kinetic_ising.py) at every RG level.

Same network and protocol as repro_figure3.py (5 patterns of 128 units, 2a -> 3,
2b -> 4, alpha = 0.2, beta = 55, h = -2), but:
  * inference rule: couplings re-inferred by exact ML (L2 = 1e-3) at every level and
    pairing on the rank-8 denoised synaptic similarity (fig3_ml_rg.py shows that this
    keeps 2a and 2b perfectly separated down to 64 blocks, whereas mean-field mixes them);
  * correlation rule: greedy pairing on the Pearson correlation, as before;
  * the coarse-grained model at RG step 5 (64 blocks) is fitted by exact ML for both
    rules, so that the comparison isolates the effect of the grouping.
Output: Figures/Figure3_repro_N1024.png; results/fig3_ml.npz

Run:  env -i /opt/homebrew/bin/python3 repro_figure3_ml.py
"""

import os
import shutil
import numpy as np

import repro_figure3 as F
from repro_figures_v2 import greedy_pairs, synaptic_similarity
from ml_kinetic_ising import fit_kinetic_ising
from pairing_tests import lowrank

F.N = 1024
T = 100_000
L2 = 1e-3
RANK = 8
CACHE = "sim_cache/seq1024_T100000.npy"


def rg_ml(S0, lab, rule, rng):
    perm = rng.permutation(S0.shape[1])
    S = S0[:, perm].astype(np.float32)
    comp = np.eye(7)[lab[perm]]
    for k in range(F.LEVEL):
        if rule == "inference":
            J = fit_kinetic_ising(S, nlev=2 ** k, l2=L2, maxiter=300)[0]
            G = synaptic_similarity(lowrank(J, RANK))
        else:
            G = np.nan_to_num(np.corrcoef(S.T))
        np.fill_diagonal(G, -np.inf)
        p = greedy_pairs(G)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
        print(f"  {rule} level {k + 1}: {S.shape[1]} blocks, purity {comp.max(1).mean():.3f}", flush=True)
    J, h, _ = fit_kinetic_ising(S, nlev=2 ** F.LEVEL, l2=L2, maxiter=300)
    return S, J, h, comp


def main():
    J, xi, lab = F.seq_network()
    if os.path.exists(CACHE):
        S0 = np.load(CACHE)
    else:
        os.makedirs("sim_cache", exist_ok=True)
        S0 = F.glauber(J, np.full(F.N, F.H_EXT / F.BETA), F.BETA, T + 500,
                       np.random.default_rng(0), S0=xi[1])[500:]
        np.save(CACHE, S0)
    res = {}
    for rule in ["correlation", "inference"]:
        S, Jk, hk, comp = rg_ml(S0, lab, rule, np.random.default_rng(1))
        o = F.block_order(comp)
        Jk, hk, comp = Jk[np.ix_(o, o)], hk[o], comp[o]
        t = comp[:, 1] + comp[:, 5]
        sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
        print(f"{rule}: purity {comp.max(1).mean():.3f}, 2a/2b separation per block {np.round(sep, 2)}", flush=True)
        g, cv = F.tune_gain(Jk, hk, comp, np.random.default_rng(2), np.linspace(0.4, 1.4, 11))
        print(f"  gain* = {g:.2f}, CV = {cv:.2f}", flush=True)
        Is = np.linspace(0, 8.0, 21)
        p3, p4, rasters = [], [], {}
        for I in Is:
            Sg = F.block_dynamics(Jk, hk, g, 20000, np.random.default_rng(3), I=F.stim_vector(comp, I))
            c = F.cycles(F.overlaps(Sg, comp))
            p3.append(c[:, 1].mean() if len(c) else np.nan)
            p4.append(c[:, 2].mean() if len(c) else np.nan)
            rasters[round(I, 2)] = Sg[:800]
        print("  P(3):", np.round(p3, 2))
        print("  P(4):", np.round(p4, 2), flush=True)
        res[rule] = dict(J=Jk, comp=comp, Is=Is, p3=np.array(p3), p4=np.array(p4), rasters=rasters)
    os.makedirs("results", exist_ok=True)
    np.savez("results/fig3_ml.npz", **{f"{r}/{k}": v for r in res for k, v in res[r].items() if k != "rasters"})
    F.plot(J, S0, res)
    shutil.move("Figures/Figure3_repro.png", "Figures/Figure3_repro_N1024.png")
    print("saved Figures/Figure3_repro_N1024.png")


if __name__ == "__main__":
    main()
