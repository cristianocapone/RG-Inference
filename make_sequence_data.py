"""
Generate the activity of the asymmetric (sequence) Hopfield network of Fig. 3 used as
input by several scripts:
  sim_cache/seq1024_T100000.npy   N = 1024, T = 1e5  (fig3_ml_rg.py, pairing_tests.py,
                                                     repro_figure3_ml.py)
  sim_cache/seq512_T300000.npy    N = 512,  T = 3e5  (closure_test.py, excess_symmetry_sim.py)
Network: 5 patterns of N/8 units, 2a -> 3, 2b -> 4, alpha = 0.2, beta = 55, h = -2,
synchronous Glauber dynamics started in pattern 2, first 500 steps discarded.

Run:  python make_sequence_data.py
"""

import os
import numpy as np

import repro_figure3 as F


def make(N, T, path):
    if os.path.exists(path):
        print(f"{path} exists")
        return
    F.N = N
    J, xi, lab = F.seq_network()
    S = F.glauber(J, np.full(N, F.H_EXT / F.BETA), F.BETA, T + 500, np.random.default_rng(0), S0=xi[1])[500:]
    np.save(path, S)
    print(f"saved {path}: {S.shape}")


if __name__ == "__main__":
    os.makedirs("sim_cache", exist_ok=True)
    make(1024, 100_000, "sim_cache/seq1024_T100000.npy")
    make(512, 300_000, "sim_cache/seq512_T300000.npy")
