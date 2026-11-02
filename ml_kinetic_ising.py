"""
Exact maximum-likelihood inference of the kinetic Ising model (replaces the
mean-field inversion of repro_figures_v2.infer_J where accuracy matters).

Model at RG level k (block variables y in {0, 1/n, ..., 1}, n = 2^k; n = 1 is the
binary model):
    P(y_i(t+1) = m/n | y(t)) = exp(m/n H_i(t)) / Z_n(H_i(t)),   H = J y(t) + h,
    Z_n(H) = sum_{m=0}^{n} exp(m H / n),   f_n(H) = d log Z_n / dH.
The log-likelihood is separable over i; we maximise it jointly for (J, h) with
L-BFGS (scipy), analytic gradient, warm start from the nMF solution, optional L2
penalty on J.  Cost per iteration: two T x N x N matrix products.

    J, h, info = fit_kinetic_ising(S, nlev=1, l2=1e-4, maxiter=300)
"""

import time
import numpy as np
from scipy.optimize import minimize

from repro_figures_v2 import infer_J, _FG, _HG


def _logZ_f(H, n):
    """log Z_n(H) and f_n(H) for H of any shape, numerically stable."""
    if n == 1:
        lz = np.logaddexp(0.0, H)
        return lz, 1.0 / (1.0 + np.exp(-H))
    m = np.arange(n + 1, dtype=np.float32) / n
    A = H[..., None] * m                                   # ... x (n+1)
    Amax = A.max(-1, keepdims=True)
    E = np.exp(A - Amax)
    Zs = E.sum(-1)
    lz = np.log(Zs) + Amax[..., 0]
    f = (E * m).sum(-1) / Zs
    return lz, f


def fit_kinetic_ising(S, nlev=1, l2=1e-4, maxiter=300, J0=None, h0=None, verbose=False, chunk=20000):
    S = np.asarray(S, np.float32)
    T, N = S.shape
    Sp, Sn = S[:-1], S[1:]
    if J0 is None:
        J0 = infer_J(S, 0 if nlev == 1 else 1)
    m = np.clip(S.mean(0), 1e-4, 1 - 1e-4)
    if h0 is None:
        Hbar = np.log(m / (1 - m)) if nlev == 1 else np.interp(m, _FG, _HG)
        h0 = Hbar - J0 @ m
    x0 = np.concatenate([J0.ravel(), h0]).astype(np.float64)
    mask = ~np.eye(N, dtype=bool)
    nb = [(a, min(a + chunk, T - 1)) for a in range(0, T - 1, chunk)]

    def fg(x):
        J = x[:N * N].reshape(N, N).astype(np.float32)
        J[~mask] = 0
        h = x[N * N:].astype(np.float32)
        L = 0.0
        gJ = np.zeros((N, N), np.float32)
        gh = np.zeros(N, np.float32)
        for a, b in nb:                                    # chunked over time
            H = Sp[a:b] @ J.T + h
            lz, f = _logZ_f(H, nlev)
            L += float((Sn[a:b] * H - lz).sum())
            R = Sn[a:b] - f
            gJ += R.T @ Sp[a:b]
            gh += R.sum(0)
        L /= (T - 1)
        gJ /= (T - 1)
        gh /= (T - 1)
        L -= 0.5 * l2 * float((J ** 2).sum())
        gJ -= l2 * J
        gJ[~mask] = 0
        return -L, -np.concatenate([gJ.ravel(), gh]).astype(np.float64)

    t0 = time.time()
    hist = []
    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-10, gtol=1e-6),
                   callback=(lambda xk: hist.append(fg(xk)[0])) if verbose else None)
    J = res.x[:N * N].reshape(N, N)
    J[~mask] = 0
    h = res.x[N * N:]
    info = dict(logL_nmf=-fg(x0)[0], logL=-res.fun, nit=res.nit, time=time.time() - t0, message=str(res.message))
    if verbose:
        print(f"  ML kinetic Ising: <logL> {info['logL_nmf']:.5f} -> {info['logL']:.5f} in {res.nit} it, {info['time']:.0f} s")
    return J, h, info
