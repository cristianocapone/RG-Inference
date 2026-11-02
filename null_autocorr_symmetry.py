"""
Control: does the symmetry of the inferred J rise under coarse-graining for
INDEPENDENT but autocorrelated units?

With C = diag(v) + noise and D(1) = diag(a) + noise (a = lag-1 autocovariance), the
nMF inversion J = D C^{-1} / f' contains the term -a_i C_ij / (v_i v_j f'_i): the
equal-time sampling noise C_ij enters symmetrically and is weighted by the
autocorrelation.  Averaging units into blocks removes the fast noise and keeps the
slow component, so a/v grows and J looks increasingly symmetric even though nothing
is coupled.  Here each unit is an independent two-state Markov chain whose rate and
lag-1 autocorrelation are drawn to match the channels of culture d3.

Run:  env -i /opt/homebrew/bin/python3 null_autocorr_symmetry.py
"""

import numpy as np
from real_data_rg import load_culture, DT
from real_data_surrogate import rg_track
from spiking_ei_rg import symmetry


def telegraph(T, p1, rho, rng):
    """Stationary 2-state Markov chain with P(1) = p1 and lag-1 autocorrelation rho."""
    # transition probs: P(1->1) = p1 + rho (1 - p1), P(0->1) = p1 (1 - rho)
    p11 = p1 + rho * (1 - p1)
    p01 = p1 * (1 - rho)
    x = np.empty(T, np.uint8)
    s = rng.random() < p1
    u = rng.random(T)
    for t in range(T):
        s = u[t] < (p11 if s else p01)
        x[t] = s
    return x


def main(N=512, n_levels=8):
    B, rates, _ = load_culture("d3", N)
    T = B.shape[0]
    m = B.mean(0)
    rho = np.array([np.corrcoef(B[:-1, i], B[1:, i])[0, 1] for i in range(N)])
    print(f"d3 channels: rate {np.median(rates):.2f} Hz, lag-1 autocorr median {np.median(rho):.3f}"
          f" (10-90%: {np.percentile(rho, 10):.3f}-{np.percentile(rho, 90):.3f})")
    rng = np.random.default_rng(0)
    X = np.stack([telegraph(T, m[i], max(rho[i], 0.0), rng) for i in range(N)], 1)
    rho_x = np.median([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(N)])
    print(f"telegraph null: T={T}, rate {np.median(X.mean(0)) / DT:.2f} Hz, lag-1 autocorr median {rho_x:.3f}")
    for rule in ["inference", "correlation", "random"]:
        Js, ex = rg_track(X, rule, n_levels)
        print(f"  {rule:12s} symmetry {np.round([symmetry(J) for J in Js], 2)}   exps {np.round(ex, 2)}")
    # same, with autocorrelation removed (iid Bernoulli with the same rates)
    Y = (rng.random((T, N)) < m).astype(np.uint8)
    for rule in ["inference", "random"]:
        Js, ex = rg_track(Y, rule, n_levels)
        print(f"  iid {rule:8s} symmetry {np.round([symmetry(J) for J in Js], 2)}")


if __name__ == "__main__":
    main()
