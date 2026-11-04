"""
Closure of the coarse-grained dynamics for reductions that do not produce groups of
units, at the same reduced dimension as the RG levels, on the cultures, the mice and the
spiking networks (512-unit subsets as in closure_test.py):
  PCA   top-k principal components of the microscopic activity (momentum-space RG)
  RRR   reduced-rank regression: the k-dimensional linear subspace that best predicts the
        next microscopic state (upper bound on the one-step closure of any linear
        k-dimensional reduction; the subspace is the top-k right singular space of the
        ridge one-step predictor, fitted on the training part)
  NMF   non-negative matrix factorisation with k components (parts-based, the closest
        non-grouping analogue of blocks), fitted on the training part
For every reduction y = S P the closure is R^2[y(t+1)|y(t)] / R^2[y(t+1)|S(t)] on the
held-out 20%, as for the grouping rules, whose values are read from results/closure.npz.
Results: results/dimred_comparison.npz.  Run from the folder holding data/ and sim_cache/.
"""

import time
import numpy as np

from closure_test import ridge_r2, FRAC, datasets
try:
    from sklearn.decomposition import NMF
except ImportError:
    NMF = None

LEVELS = [3, 4, 5]
RULES = ["inference", "inference_out", "lagged_out", "correlation", "random"]


def closure_proj(S0, P, Tt):
    dX = S0 - S0[:Tt].mean(0)
    dY = dX @ P
    r2c = ridge_r2(dY[:Tt - 1], dY[1:Tt], dY[Tt:-1], dY[Tt + 1:])
    r2m = ridge_r2(dX[:Tt - 1], dY[1:Tt], dX[Tt:-1], dY[Tt + 1:])
    return r2c / r2m


def pca(S0, k, Tt):
    dX = S0[:Tt] - S0[:Tt].mean(0)
    return np.linalg.svd(dX, full_matrices=False)[2][:k].T


def rrr(S0, k, Tt, ridge=1e-2):
    dX = S0[:Tt] - S0[:Tt].mean(0)
    X, Y = dX[:-1], dX[1:]
    C = X.T @ X / len(X)
    C += ridge * np.trace(C) / len(C) * np.eye(len(C), dtype=C.dtype)
    B = np.linalg.solve(C, X.T @ Y / len(X))           # OLS one-step predictor
    Yhat = X @ B
    return np.linalg.svd(Yhat, full_matrices=False)[2][:k].T   # top-k predicted directions


def nmf(S0, k, Tt):
    m = NMF(n_components=k, init="nndsvda", max_iter=300, random_state=0)
    m.fit(S0[:Tt])
    H = m.components_                                   # k x N
    return np.linalg.pinv(H)                            # N x k: y = S H^+


def main():
    cl = np.load("results/closure.npz")
    store = {}
    for group, name, B in datasets("sim_cache/seq512_T300000.npy"):
        if group == "hopfield":
            continue
        S0 = B.astype(np.float32)
        N = S0.shape[1]
        Tt = int(FRAC * len(S0))
        for lev in LEVELS:
            k = N // 2 ** lev
            t0 = time.time()
            vals = {"pca": closure_proj(S0, pca(S0, k, Tt), Tt), "rrr": closure_proj(S0, rrr(S0, k, Tt), Tt)}
            if NMF is not None:
                vals["nmf"] = closure_proj(S0, nmf(S0, k, Tt), Tt)
            for r in RULES:
                v = cl[f"{group}/{name}/{r}/closure"]
                vals[r] = float(v[lev]) if lev < len(v) else np.nan
            for key, v in vals.items():
                store[f"{group}/{name}/{lev}/{key}"] = v
            print(f"{group}/{name} level {lev} (k={k}): " + "  ".join(f"{key} {v:.3f}" for key, v in vals.items()) + f"  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/dimred_comparison.npz", **store)


if __name__ == "__main__":
    main()
