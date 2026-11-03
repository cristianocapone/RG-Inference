"""
Closure (lumpability) test of the coarse-graining rules.

Pairing units with similar input AND output coupling vectors is the lumpability
criterion: if J_i. = J_j. and J_.i = J_.j, merging i and j is an exact reduction of
the dynamics.  The right metric is therefore how self-contained the coarse-grained
variables are.  At each level k, with block variables y = S0 M / |M|:
  r2_coarse  held-out R^2 of predicting dy(t+1) from the block state dy(t)
  r2_micro   held-out R^2 of predicting dy(t+1) from the full microscopic state dx(t)
  closure    r2_coarse / r2_micro  (1 = nothing lost by coarse-graining)
Rules: inference (in+out similarity), inference_in (rows of J only), inference_out
(columns only), correlation, random.  Data sets: 9 cultures, 3 mice, 2 spiking E/I
networks (delayed, 10 ms), Hopfield sequence network.
Figure: Figures/Closure.png; results: results/closure.npz.

Run:  env -i /opt/homebrew/bin/python3 closure_test.py <S0seq512.npy>
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity
from rg_metrics import pairs_for
from real_data_rg import load_culture
from invivo_stringer import bin_mouse, MICE
import spiking_ei_rg as spk

RULES5 = ["inference", "inference_in", "inference_out", "correlation", "lagged", "lagged_out", "random"]
COL5 = {"inference": "#d62728", "inference_in": "#ff9896", "inference_out": "#8b0000",
        "correlation": "#8c8c00", "lagged": "#9467bd", "lagged_out": "#5e3c99", "random": "#4a7fd6"}
L = 8
FRAC = 0.8            # first 80% of the recording: pairing and fit; last 20%: evaluation
RIDGES = (1e-3, 1e-2, 1e-1, 1.0)


def lagged_corr(S):
    """Model-free directional statistic: lag-1 cross-correlation R_ij = <dS_i(t+1) dS_j(t)> / (s_i s_j)."""
    d = S - S.mean(0)
    sd = d.std(0) + 1e-12
    return (d[1:].T @ d[:-1]) / (len(S) - 1) / np.outer(sd, sd)


def pairs5(rule, S, J, rng):
    if rule in ("lagged", "lagged_out"):
        J = lagged_corr(S)
        np.fill_diagonal(J, 0)
        rule = "inference" if rule == "lagged" else "inference_out"
    if rule in ("inference_in", "inference_out"):
        V = J if rule == "inference_in" else J.T
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        G = V @ V.T
        np.fill_diagonal(G, -np.inf)
        return greedy_pairs(G)
    return pairs_for(rule, S, J, rng)


def _fit(Xtr, Ytr, ridge):
    C = Xtr.T @ Xtr / len(Xtr)
    C += ridge * np.trace(C) / len(C) * np.eye(len(C), dtype=C.dtype)
    return np.linalg.solve(C, Xtr.T @ Ytr / len(Xtr))


def _r2(Y, Yhat):
    return float(np.mean(1 - ((Y - Yhat) ** 2).sum(0) / ((Y ** 2).sum(0) + 1e-12)))


def ridge_r2(Xtr, Ytr, Xte, Yte, ridges=RIDGES):
    """Held-out R^2; the ridge is chosen on the last quarter of the training set."""
    v = int(0.75 * len(Xtr))
    best = max(ridges, key=lambda r: _r2(Ytr[v:], Xtr[v:] @ _fit(Xtr[:v], Ytr[:v], r)))
    return _r2(Yte, Xte @ _fit(Xtr, Ytr, best))


def closure(S0, M, frac=FRAC):
    """S0: T x N micro (float32), M: N x Nk membership."""
    T = len(S0)
    Tt = int(frac * T)
    Y = S0 @ (M / M.sum(0))
    mx, my = S0[:Tt].mean(0), Y[:Tt].mean(0)
    dX, dY = S0 - mx, Y - my
    r2c = ridge_r2(dY[:Tt - 1], dY[1:Tt], dY[Tt:-1], dY[Tt + 1:])
    r2m = ridge_r2(dX[:Tt - 1], dY[1:Tt], dX[Tt:-1], dY[Tt + 1:])
    return r2c, r2m


def rg_closure(B, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S0 = B.astype(np.float32)
    S = S0
    N = S.shape[1]
    M = np.eye(N, dtype=np.float32)
    out = {"r2_coarse": [], "r2_micro": [], "closure": []}
    for k in range(n_levels):
        r2c, r2m = closure(S0, M)
        out["r2_coarse"].append(r2c)
        out["r2_micro"].append(r2m)
        out["closure"].append(r2c / r2m if r2m > 0 else np.nan)
        if k == n_levels - 1 or S.shape[1] < 4:
            break
        Str = S[:int(FRAC * len(S))]                      # pairing from the training part only
        J = infer_J(Str, k) if rule.startswith("inference") else None
        p = pairs5(rule, Str, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return {k: np.pad(np.array(v, float), (0, n_levels - len(v)), constant_values=np.nan) for k, v in out.items()}


def datasets(seq_path):
    for c in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]:
        B_all, rates, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        yield "cultures", c, B_all[:, :N]
    for mouse, tstart in MICE.items():
        B, *_ = bin_mouse(f"data/stringer2019_neuropixels/spks{mouse}_Feb18.mat", tstart)
        yield "mice", mouse, B
    for name in ["clustered", "chain"]:
        d = np.load(f"sim_cache/{name}_d5j0.5_s0.15_Nref3000_wclu1.9_spikes_T600.npz")
        B = spk.bin_spikes(d["ts"], d["ids"], 10.0)
        rng = np.random.default_rng(0)                       # 512 units: 400 E + 112 I
        idx = np.sort(np.r_[rng.choice(spk.NE, 400, replace=False), spk.NE + rng.choice(spk.NI, 112, replace=False)])
        yield "spiking", name, B[:, idx]
    if seq_path:
        yield "hopfield", "seq512", np.load(seq_path).astype(np.uint8)


def main(seq_path=None):
    os.makedirs("results", exist_ok=True)
    store, groups = {}, {}
    for group, name, B in datasets(seq_path):
        t0 = time.time()
        n_levels = min(L, int(np.log2(B.shape[1])) - 1)
        line = []
        for rule in RULES5:
            o = rg_closure(B, rule, n_levels)
            for m in o:
                store[f"{group}/{name}/{rule}/{m}"] = o[m]
            line.append(f"{rule:13s} {np.round(o['closure'], 2)}")
        groups.setdefault(group, []).append(name)
        print(f"=== {group}/{name}: T={B.shape[0]}, N={B.shape[1]}  [{time.time() - t0:.0f} s]\n  closure: " + "\n  closure: ".join(line), flush=True)
    np.savez("results/closure.npz", **store)
    plot(store, groups)


def plot(store, groups):
    steps = np.arange(1, L + 1)
    fig, axs = plt.subplots(2, len(groups), figsize=(4.2 * len(groups), 7), squeeze=False)
    for col, (group, names) in enumerate(groups.items()):
        for row, m in enumerate(["closure", "r2_coarse"]):
            ax = axs[row, col]
            for rule in RULES5:
                A = np.stack([np.pad(store[f"{group}/{n}/{rule}/{m}"], (0, L - len(store[f"{group}/{n}/{rule}/{m}"])),
                                     constant_values=np.nan) for n in names])
                y = np.nanmean(A, 0)
                e = np.nanstd(A, 0) / np.sqrt(np.maximum(np.sum(~np.isnan(A), 0), 1))
                ax.errorbar(steps, y, e, fmt="o-", color=COL5[rule], ms=3, capsize=2, label=rule)
            ax.set_title(f"{group} (n={len(names)}): {m}", fontsize=9)
            ax.set_xlabel("RG step")
            if m == "closure":
                ax.axhline(1, color="0.6", ls=":")
        axs[0, col].legend(fontsize=7, frameon=False)
    fig.suptitle("closure = R2(y(t+1)|y(t)) / R2(y(t+1)|x(t)), held-out; mean +/- sem", fontsize=10)
    fig.tight_layout()
    fig.savefig("Figures/Closure.png", dpi=170)
    print("saved Figures/Closure.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        st = dict(np.load("results/closure.npz"))
        gr = {}
        for k in st:
            g, n = k.split("/")[:2]
            gr.setdefault(g, [])
            if n not in gr[g]:
                gr[g].append(n)
        plot(st, gr)
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
