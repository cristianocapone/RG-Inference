"""
Directedness of the recorded dynamics, corrected for sampling noise.

The raw index 1 - symmetry(D(1)) is maximal for independent units: sampling noise
contributes equally to the symmetric and antisymmetric parts of the lagged covariance,
so weak or noisy cross-covariances look "directional".  Here the lagged correlation
R_ij = D(1)_ij / (sigma_i sigma_j) (off-diagonal) is compared with circular-shift
surrogates, which keep the autocorrelation and the rate of every unit and destroy the
cross-correlations.  The noise power of each part is subtracted,
    A = |R_a|^2 - <|R_a,surr|^2>,   Y = |R_s|^2 - <|R_s,surr|^2>,
and directedness = A / (A + Y), the fraction of the excess lagged cross-correlation
power that is antisymmetric (0: time-reversible, 1: purely directional).  NaN if the
excess power is not positive.
"""

import numpy as np

from rg_metrics import shift_surrogate


def _parts(S, lags):
    d = S - S.mean(0)
    sd = d.std(0) + 1e-12
    off = ~np.eye(S.shape[1], dtype=bool)
    a = y = 0.0
    for k in lags:
        R = (d[k:].T @ d[:-k]) / (len(S) - k) / np.outer(sd, sd)
        Rs, Ra = 0.5 * (R + R.T), 0.5 * (R - R.T)
        a, y = a + float((Ra[off] ** 2).sum()), y + float((Rs[off] ** 2).sum())
    return a, y


def directedness(B, lags=(1,), n_surr=3, seed=0):
    """B: T x N activity.  Returns (directedness, raw index 1 - symmetry(D(1)))."""
    S = B.astype(np.float32)
    S = S[:, S.std(0) > 0]
    a, y = _parts(S, lags)
    rng = np.random.default_rng(seed)
    sa, sy = np.mean([_parts(shift_surrogate(S, rng), lags) for _ in range(n_surr)], 0)
    A, Y = a - sa, y - sy
    raw = 1 - (np.sqrt(y) - np.sqrt(a)) / (np.sqrt(y) + np.sqrt(a))
    return (A / (A + Y) if A + Y > 0 else np.nan), raw
