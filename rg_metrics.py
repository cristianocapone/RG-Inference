"""
Per-level metrics along an RG trajectory, for data and for shifted surrogates.

For every level k of a trajectory (pairing rule: inference / correlation / random):
  sym_J    symmetry of the inferred kinetic-Ising coupling J^(k)   (as in Fig. 4)
  sym_D    symmetry of the lagged covariance D^(k)(1) of the coarse-grained activity:
           a model-free "time-reversibility" index (1 = reversible, <1 = directed)
  r2       held-out one-step linear predictability of the coarse-grained activity,
           delta S(t+1) ~ J_lin delta S(t) with J_lin = D C^-1 fitted on the first 80%
           of the recording, R^2 averaged over block neurons on the last 20%
  r2_auto  same, with the diagonal (autoregressive) predictor only; r2 - r2_auto is the
           cross-unit predictability retained by the coarse-graining
  r2_ag    autoregressive + global predictor (own past and the population mean);
           r2 - r2_ag is the predictability beyond a single global mode ("specific")
  exp      local variance-scaling exponent
The "excess" of a metric is its value on the data minus its value on a surrogate in
which every unit is circularly shifted in time (autocorrelations kept, all
cross-correlations destroyed): this removes the part of sym_J that is produced by
autocorrelation + finite sampling (see null_autocorr_symmetry.py).
"""

import numpy as np

from repro_figures_v2 import (infer_J, greedy_pairs, synaptic_similarity,
                              variance_scaling, local_exponent)
from spiking_ei_rg import symmetry

RULES = ["inference", "correlation", "random"]
COLORS = {"inference": "#d62728", "correlation": "#8c8c00", "random": "#4a7fd6"}


def shift_surrogate(B, rng, min_shift=100):
    T, N = B.shape
    out = np.empty_like(B)
    for i in range(N):
        out[:, i] = np.roll(B[:, i], int(rng.integers(min_shift, T - min_shift)))
    return out


def lagged_cov(S):
    T = len(S)
    d = S - S.mean(0)
    C = d[:-1].T @ d[:-1] / (T - 1)
    D = d[1:].T @ d[:-1] / (T - 1)
    return C, D


def heldout_r2(S, frac=0.8, ridge=1e-3):
    T = len(S)
    Tt = int(frac * T)
    tr, te = S[:Tt], S[Tt:]
    m = tr.mean(0)
    dtr, dte = tr - m, te - m
    C = dtr[:-1].T @ dtr[:-1] / (Tt - 1)
    D = dtr[1:].T @ dtr[:-1] / (Tt - 1)
    n = len(C)
    Cr = C + ridge * np.trace(C) / n * np.eye(n, dtype=C.dtype)
    Jl = np.linalg.solve(Cr.T, D.T).T
    tot = (dte[1:] ** 2).sum(0) + 1e-12
    res = dte[1:] - dte[:-1] @ Jl.T
    r2 = 1 - (res ** 2).sum(0) / tot
    a = np.diag(D) / (np.diag(C) + 1e-12)
    res_a = dte[1:] - dte[:-1] * a
    r2a = 1 - (res_a ** 2).sum(0) / tot
    # autoregressive + global mode: per-unit least squares on [own past, population mean]
    X1, Y = dtr[:-1], dtr[1:]
    X2 = X1.mean(1, keepdims=True)
    s11 = (X1 * X1).sum(0) + 1e-12
    s12 = (X1 * X2).sum(0)
    s22 = float((X2 * X2).sum()) + 1e-12
    y1, y2 = (X1 * Y).sum(0), (X2 * Y).sum(0)
    det = s11 * s22 - s12 ** 2 + 1e-12
    b1, b2 = (s22 * y1 - s12 * y2) / det, (s11 * y2 - s12 * y1) / det
    res_ag = dte[1:] - dte[:-1] * b1 - dte[:-1].mean(1, keepdims=True) * b2
    r2ag = 1 - (res_ag ** 2).sum(0) / tot
    return float(np.mean(r2)), float(np.mean(r2a)), float(np.mean(r2ag))


def pairs_for(rule, S, J, rng):
    if rule == "inference":
        return greedy_pairs(synaptic_similarity(J))
    if rule == "correlation":
        C = np.nan_to_num(np.corrcoef(S.T))
        np.fill_diagonal(C, -np.inf)
        return greedy_pairs(C)
    return rng.permutation(S.shape[1] // 2 * 2).reshape(-1, 2)


def rg_metrics(B, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    out = {k: [] for k in ["sym_J", "sym_D", "r2", "r2_auto", "r2_ag", "N"]}
    levels = [S]
    for k in range(n_levels):
        J = infer_J(S, k)
        _, D = lagged_cov(S)
        out["sym_J"].append(symmetry(J))
        out["sym_D"].append(symmetry(D))
        r2, r2a, r2ag = heldout_r2(S)
        out["r2"].append(r2)
        out["r2_auto"].append(r2a)
        out["r2_ag"].append(r2ag)
        out["N"].append(S.shape[1])
        if k == n_levels - 1 or S.shape[1] < 4:
            break
        p = pairs_for(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        levels.append(S)
    K, M2 = variance_scaling(levels)
    out["exp"] = local_exponent(K, M2)[1]
    return {k: np.array(v, float) for k, v in out.items()}


def data_vs_surrogate(B, n_levels, seed=0, n_surr=1):
    """Metrics for the three rules on the data and on n_surr shifted surrogates."""
    res = {}
    surrs = [shift_surrogate(B, np.random.default_rng(1000 + s)) for s in range(n_surr)]
    for rule in RULES:
        real = rg_metrics(B, rule, n_levels, seed)
        sur = [rg_metrics(Bs, rule, n_levels, seed) for Bs in surrs]
        res[rule] = {"real": real,
                     "surr": {k: np.mean([s[k] for s in sur], 0) for k in real}}
    return res


def pad(a, L):
    out = np.full(L, np.nan)
    out[:min(L, len(a))] = a[:L]
    return out
