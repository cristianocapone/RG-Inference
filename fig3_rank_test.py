"""
Data-driven rank for the denoised synaptic similarity of Fig. 3: at every RG level the
rank is set at the largest relative gap of the singular-value spectrum of the inferred
couplings (pairing_tests.gap_rank), instead of the number of stored patterns.
Reports, per level, the chosen rank, block purity and 2a/2b separation.
Run from the folder holding sim_cache/:  python fig3_rank_test.py
"""
import time
import numpy as np
from ml_kinetic_ising import fit_kinetic_ising
from repro_figures_v2 import greedy_pairs, synaptic_similarity
from pairing_tests import labels_seq, lowrank_gap

S0 = np.load("sim_cache/seq1024_T100000.npy").astype(np.float32)
N = S0.shape[1]
lab, names = labels_seq(N)
perm = np.random.default_rng(1).permutation(N)
S, comp = S0[:, perm], np.eye(7)[lab[perm]]
t0 = time.time()
for k in range(4):
    J = fit_kinetic_ising(S, nlev=2 ** k, l2=1e-3, maxiter=300)[0]
    sv = np.linalg.svd(J, compute_uv=False)
    Jl, r = lowrank_gap(J)
    p = greedy_pairs(synaptic_similarity(Jl))
    S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
    comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
    t = comp[:, 1] + comp[:, 5]
    sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
    print(f"level {k + 1}: rank {r} (top sv ratios {np.round(sv[:12] / sv[1:13], 2)}), {S.shape[1]} blocks, "
          f"purity {comp.max(1).mean():.3f}, 2a/2b separation mean {sep.mean():.2f} min {sep.min():.2f}  [{time.time() - t0:.0f} s]",
          flush=True)
