"""
Inference aware of a shared unobserved input: does it recover the in-vivo advantage?

The decomposition (decomposition_inversion.py) shows that the off-diagonal part of C^-1
costs closure in vivo, where the common drive comes from outside the recorded population.
Here the couplings used for the pairing (output rule, columns) are estimated after
accounting for a shared input in three ways, and compared with the plain inversion and
with the model-free lagged rule, on the cultures and on the cached Allen sessions:
  lagged        columns of D / (sigma_i sigma_j)                         (model-free)
  nmf           columns of D C^-1 / f'                                   (the paper's rule)
  resid         population mean g(t) regressed out of every unit (fit on the training part),
                then D_r C_r^-1 on the residuals (no row weight; `resid_sig`: weight 1/sigma_i)
  lowrank1/3    D C^-1 with the top 1 / 3 eigen-directions of C removed from the inverse
                (the shared modes are not deconvolved, the rest is)
  popfield      kinetic-Ising-like regression of S(t+1) on [S(t), g(t+1)]: the population
                activity at the target time stands for the instantaneous shared input;
                J = the S(t) part of the regression (no row weight)
All variants use the same RG loop (pairing on the first 80%) and the same held-out closure
on the original block variables.
Results: results/latent_cultures.npz, results/latent_allen.npz; summary on stdout.
Run from the folder holding data/ and results/:  python latent_inference.py cultures|allen|summary
"""

import glob
import os
import sys
import numpy as np

from closure_test import closure, FRAC
from repro_figures_v2 import infer_J, greedy_pairs

KINDS = ["correlation", "lagged", "nmf", "resid", "resid_sig", "lowrank1", "lowrank3", "popfield"]


def cols_pairs(V):
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    G = V @ V.T
    np.fill_diagonal(G, -np.inf)
    return greedy_pairs(G)


def couplings(S, k, kind):
    S = np.asarray(S, np.float32)
    d = S - S.mean(0)
    T = len(S)
    sd = d.std(0) + 1e-12
    if kind == "nmf":
        return infer_J(S, k)
    if kind == "lagged":
        D = d[1:].T @ d[:-1] / (T - 1)
        np.fill_diagonal(D, 0)
        return D / np.outer(sd, sd)
    if kind.startswith("resid"):
        g = d.mean(1, keepdims=True)
        r = d - g @ ((g.T @ d) / float((g ** 2).sum()))
        D = r[1:].T @ r[:-1] / (T - 1)
        C = r.T @ r / T
        C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C), dtype=C.dtype)
        J = np.linalg.solve(C.T, D.T).T
        if kind == "resid_sig":
            J = J / (r.std(0) + 1e-12)[:, None]
        np.fill_diagonal(J, 0)
        return J
    if kind.startswith("lowrank"):
        nrem = int(kind[-1])
        D = d[1:].T @ d[:-1] / (T - 1)
        C = d.T @ d / T
        w, U = np.linalg.eigh(C)
        keep = np.argsort(w)[:-nrem]                       # drop the largest eigen-directions
        Cinv = (U[:, keep] / np.maximum(w[keep], 1e-6)) @ U[:, keep].T
        J = D @ Cinv
        np.fill_diagonal(J, 0)
        return J
    if kind == "popfield":
        g1 = d[1:].mean(1, keepdims=True)                  # population activity at the target time
        X = np.concatenate([d[:-1], g1], 1)
        C = X.T @ X / (T - 1)
        C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C), dtype=C.dtype)
        B = np.linalg.solve(C, X.T @ d[1:] / (T - 1))      # (N+1) x N regression coefficients
        J = B[:-1].T                                       # J_ij: effect of j(t) on i(t+1), given the population at t+1
        np.fill_diagonal(J, 0)
        return J
    raise ValueError(kind)


def rg(B, kind, L):
    S0 = B.astype(np.float32)
    S, M = S0, np.eye(S0.shape[1], dtype=np.float32)
    out = []
    for k in range(L):
        r2c, r2m = closure(S0, M)
        out.append(r2c / r2m)
        if k == L - 1 or S.shape[1] < 4:
            break
        Str = S[:int(FRAC * len(S))]
        if kind == "correlation":
            G = np.nan_to_num(np.corrcoef(Str.T))
            np.fill_diagonal(G, -np.inf)
            p = greedy_pairs(G)
        else:
            p = cols_pairs(couplings(Str, k, kind).T)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return np.array(out)


def evaluate(B):
    B = B[:, B.std(0) > 0]
    B = B[:, :B.shape[1] // 2 * 2]
    L = int(np.log2(B.shape[1])) - 1
    cl = {k: rg(B, k, L) for k in KINDS}
    return np.array([float(np.nanmean((cl[k] - cl["correlation"])[1:L - 1])) for k in KINDS[1:]])


def run_cultures():
    from real_data_rg import load_culture
    store = {}
    for c in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]:
        B_all, _, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        store[c] = evaluate(B_all[:, :N])
        print(f"{c:4s}: " + "  ".join(f"{k} {v:+.3f}" for k, v in zip(KINDS[1:], store[c])), flush=True)
        np.savez("results/latent_cultures.npz", **store)


def run_allen():
    from invivo_binsize import bin_condition, CACHE
    from allen_rest_vs_task import condition_bins
    store = {}
    for f in sorted(glob.glob(f"{CACHE}/*.npz")):
        ses = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        spikes = np.split(d["spikes"], d["index"][:-1])
        t_max = max(s[-1] for s in spikes if len(s))
        T_eq = int(float(d["T_eq_min"]) * 60 / 0.01)
        for cond in ["spontaneous", "natural_movie_three"]:
            if f"iv/{cond}" not in d:
                continue
            bins = condition_bins(d[f"iv/{cond}"], d["inv"], t_max, 0.01)[:T_eq]
            store[f"{ses}/{cond}"] = evaluate(bin_condition(spikes, bins, 0.01))
            print(f"{ses} {cond:20s}: " + "  ".join(f"{k} {v:+.3f}" for k, v in zip(KINDS[1:], store[f'{ses}/{cond}'])), flush=True)
            np.savez("results/latent_allen.npz", **store)


def summary():
    from scipy.stats import wilcoxon
    for name, f in [("cultures", "results/latent_cultures.npz"), ("Allen (512 most active units)", "results/latent_allen.npz")]:
        if not os.path.exists(f):
            continue
        A = np.array(list(dict(np.load(f)).values()))
        print(f"\n=== {name}, n = {len(A)}: closure advantage over correlation (mean; > 0; Wilcoxon) and difference from the lagged rule")
        for j, k in enumerate(KINDS[1:]):
            dl = A[:, j] - A[:, 0]
            print(f"  {k:10s} {A[:, j].mean():+.4f} ({np.sum(A[:, j] > 0)}/{len(A)}, p={wilcoxon(A[:, j]).pvalue:.3f})"
                  + ("" if j == 0 else f"   vs lagged {dl.mean():+.4f} ({np.sum(dl > 0)}/{len(A)}, p={wilcoxon(dl).pvalue:.3f})"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if what == "cultures":
        run_cultures()
    elif what == "allen":
        run_allen()
    summary()
