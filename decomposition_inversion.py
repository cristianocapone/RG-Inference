"""
What does the inversion add to the lagged statistics?  (decomposition of the inference-based
output rule into its ingredients, on the cultures and on the Allen sessions)

The output rule pairs units by the cosine similarity of the columns of J = D C^-1 / f'; the
model-free lagged rule uses the columns of R = D / (sigma_i sigma_j).  For a column cosine only the
row weighting matters, so the two differ by (i) the row weight (1/sigma_i for R, 1/sigma_i^2 for
J with binary units, none for D C^-1) and (ii) the off-diagonal part of C^-1 (the deconvolution of
common causes).  Variants compared, all with the same RG loop and held-out closure:
  lagged    columns of D / (sigma_i sigma_j)
  diag      columns of D / (sigma_i^2 sigma_j^2)      (C^-1 replaced by its diagonal)
  nmf_nof   columns of D C^-1                          (full inversion, no row weight)
  nmf_sig   columns of D C^-1 / sigma_i                (full inversion, the row weight of R)
  nmf       columns of D C^-1 / f'_i                   (the inference-based output rule)
Results: results/decomp_cultures.npz (9 cultures), results/decomp_allen.npz (10 cached Allen
sessions, spontaneous and natural movie three, 512 most active units; the two "subsets" coincide).
Run from the folder holding data/ and results/:  python decomposition_inversion.py cultures|allen
"""

import sys, glob, os
import numpy as np
from closure_test import closure, FRAC
from repro_figures_v2 import infer_J, greedy_pairs
from invivo_binsize import bin_condition
from allen_rest_vs_task import condition_bins

def cols_pairs(V):
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12); G = V @ V.T; np.fill_diagonal(G, -np.inf); return greedy_pairs(G)

def Jvariant(S, k, kind):
    d = S - S.mean(0); T = len(S)
    D = d[1:].T @ d[:-1] / (T - 1); np.fill_diagonal(D, 0)
    sd = d.std(0) + 1e-12
    if kind == "lagged":   return D / np.outer(sd, sd)
    if kind == "diag":     return D / (sd ** 2)[:, None] / (sd ** 2)[None, :]
    if kind == "nmf":      return infer_J(S, k)
    if kind == "nmf_nof":
        C = d.T @ d / T; J = np.linalg.solve(C.T, D.T).T; np.fill_diagonal(J, 0); return J
    if kind == "nmf_sig":                                   # full inversion, row weight 1/sigma_i (as the lagged rule)
        C = d.T @ d / T; J = np.linalg.solve(C.T, D.T).T / sd[:, None]; np.fill_diagonal(J, 0); return J

def rg(B, kind, L):
    S0 = B.astype(np.float32); S = S0; M = np.eye(S.shape[1], dtype=np.float32); out = []
    for k in range(L):
        r2c, r2m = closure(S0, M); out.append(r2c / r2m)
        if k == L - 1 or S.shape[1] < 4: break
        Str = S[:int(FRAC * len(S))]
        if kind == "correlation":
            G = np.nan_to_num(np.corrcoef(Str.T)); np.fill_diagonal(G, -np.inf); p = greedy_pairs(G)
        else:
            p = cols_pairs(Jvariant(Str, k, kind).T)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]]); M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return np.array(out)

KINDS = ["correlation", "lagged", "diag", "nmf_nof", "nmf_sig", "nmf"]


def run_allen():
    store = {}
    for f in sorted(glob.glob("data/allen_cache/*.npz")):
        ses = os.path.basename(f).replace(".npz", ""); d = np.load(f)
        spikes = np.split(d["spikes"], d["index"][:-1]); t_max = max(s[-1] for s in spikes if len(s)); T_eq = int(float(d["T_eq_min"]) * 60 / 0.01)
        for cond in ["spontaneous", "natural_movie_three"]:
            if f"iv/{cond}" not in d: continue
            bins = condition_bins(d[f"iv/{cond}"], d["inv"], t_max, 0.01)[:T_eq]
            for sd in range(2):
                rnd = np.sort(np.random.default_rng(sd).choice(len(spikes), 512, replace=False))
                B = bin_condition([spikes[i] for i in rnd], bins, 0.01); B = B[:, B.std(0) > 0]; B = B[:, :B.shape[1] // 2 * 2]; L = int(np.log2(B.shape[1])) - 1
                cl = {k: rg(B, k, L) for k in KINDS}
                adv = {k: float(np.nanmean((cl[k] - cl["correlation"])[1:L - 1])) for k in KINDS[1:]}
                store[f"{ses}/{cond}/{sd}"] = np.array([adv[k] for k in KINDS[1:]])
                print(f"{ses} {cond:20s} sub{sd}: " + "  ".join(f"{k} {v:+.3f}" for k, v in adv.items()), flush=True)
        np.savez("results/decomp_allen.npz", **store)
    A = np.array(list(store.values()))
    print("mean over sessions x conditions:", dict(zip(KINDS[1:], np.round(A.mean(0), 4))))
    print("fraction > 0:", (A > 0).mean(0))


def run_cultures():
    from real_data_rg import load_culture
    store = {}
    for c in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]:
        B_all, _, _ = load_culture(c, 10000); N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        B = B_all[:, :N]; B = B[:, B.std(0) > 0]; B = B[:, :B.shape[1] // 2 * 2]; L = int(np.log2(B.shape[1])) - 1
        cl = {k: rg(B, k, L) for k in KINDS}
        adv = {k: float(np.nanmean((cl[k] - cl["correlation"])[1:L - 1])) for k in KINDS[1:]}
        store[c] = np.array([adv[k] for k in KINDS[1:]])
        print(f"{c:4s}: " + "  ".join(f"{k} {v:+.3f}" for k, v in adv.items()), flush=True)
        np.savez("results/decomp_cultures.npz", **store)
    A = np.array(list(store.values()))
    print("mean over 9 cultures:", dict(zip(KINDS[1:], np.round(A.mean(0), 4))), "| >0:", dict(zip(KINDS[1:], (A > 0).sum(0))))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cultures"
    run_cultures() if what == "cultures" else run_allen()
