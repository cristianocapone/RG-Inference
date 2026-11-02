"""
Fig. 3 at the paper's stated size (N = 1024, T = 1e5) with exact-ML inference at every
RG level (ml_kinetic_ising.py) and low-rank (r = 8) denoised synaptic similarity:
the 2a / 2b sub-populations stay perfectly separated down to 64 blocks (purity 1.000,
2a/2b separation 1.00 at levels 1-4), whereas with nMF the pairing mixes them (0.38
at level 0).  Variants: lowrank_all (denoised similarity at every level), lowrank_L0
(level 0 only), raw (cosine similarity on the ML J).
Input: the N = 1024 activity file produced by repro_figure3.glauber (see pairing_tests.py)
and, optionally, a cached level-0 ML coupling matrix.

Run:  env -i /opt/homebrew/bin/python3 fig3_ml_rg.py <S0seq1024.npy> [Jml_level0.npy]
"""
import sys, time, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from ml_kinetic_ising import fit_kinetic_ising
from repro_figures_v2 import greedy_pairs, synaptic_similarity
from pairing_tests import labels_seq, lowrank
SP = None
S0 = np.load(sys.argv[1]).astype(np.float32); N = S0.shape[1]; lab, names = labels_seq(N)
perm = np.random.default_rng(0).permutation(N); S0, lab = S0[:, perm], lab[perm]
J0 = np.load(sys.argv[2]) if len(sys.argv) > 2 else fit_kinetic_ising(S0, nlev=1, l2=1e-3, maxiter=300)[0]
def run(variant, n_levels=4):
    S, comp = S0, np.eye(7)[lab]; t0 = time.time()
    for k in range(n_levels):
        J = J0 if k == 0 else fit_kinetic_ising(S, nlev=2 ** k, l2=1e-3, maxiter=200)[0]
        use_lr = (variant == 'lowrank_all') or (variant == 'lowrank_L0' and k == 0)
        G = synaptic_similarity(lowrank(J, 8) if use_lr else J)
        p = greedy_pairs(G)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]]); comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
        t = comp[:, 1] + comp[:, 5]; sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
        print(f"  {variant:12s} level {k+1}: {S.shape[1]} blocks, purity {comp.max(1).mean():.3f}, 2a/2b sep mean {sep.mean():.2f} min {sep.min():.2f}  [{time.time()-t0:.0f} s]", flush=True)
for v in ['lowrank_L0', 'raw', 'lowrank_all']:
    run(v)
