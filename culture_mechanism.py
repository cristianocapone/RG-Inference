"""
Mechanism of the advantage in the cultures (referee controls): is the closure advantage
of the inference-based rule a by-product of block geometry on the 64 x 64 electrode
grid, of the network bursts, or of the global mode?

For every culture (electrode coordinates from the channel names ChRR_CC, pitch units):
  compactness   mean pairwise distance between the electrodes of a block, at every RG
                level, for each rule (inference in+out, inference output, correlation,
                random, and a spatial rule that pairs the closest blocks: Kadanoff blocks)
  closure       held-out closure along the RG for every rule including the spatial one
                (groupings on the first 80%, evaluation on the last 20%, as in the paper)
  bursts        closure evaluated separately on the held-out transitions inside and
                outside network bursts (population activity above mean + 2 s.d. of its
                smoothed value, 50-ms window)
  global mode   closure computed after regressing the population mean activity out of
                every unit and block variable (residual dynamics)
Results: results/culture_mechanism.npz; figure Figures/Supp_mechanism.png; table on stdout.
Run from the folder holding data/ and results/:  python culture_mechanism.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from closure_test import pairs5, infer_J_est, ridge_r2, FRAC, L
from repro_figures_v2 import greedy_pairs
from real_data_rg import load_culture

RULES = ["inference", "inference_out", "correlation", "spatial", "random"]
COL = {"inference": "#d62728", "inference_out": "#8b0000", "correlation": "#8c8c00", "spatial": "#2a9d8f", "random": "#4a7fd6"}
LAB = {"inference": "inference (in+out)", "inference_out": "inference (output)", "correlation": "correlation", "spatial": "spatial (nearest blocks)", "random": "random"}
CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
OUT = "results/culture_mechanism.npz"


def coords(chans):
    return np.array([[int(c[2:4]), int(c[5:7])] for c in chans], float)


def pair_spatial(M, xy):
    cen = (M.T @ xy) / M.sum(0)[:, None]                      # block centroids
    D = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return greedy_pairs(-D)


def compactness(M, xy):
    v = []
    for b in range(M.shape[1]):
        idx = np.flatnonzero(M[:, b])
        if len(idx) < 2:
            continue
        P = xy[idx]
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        v.append(D[np.triu_indices(len(idx), 1)].mean())
    return float(np.mean(v)) if v else np.nan


def memberships(B, rule, xy, seed=0):
    """List of membership matrices (N x N/2^k) for k = 0..L-1, pairing on the training part only."""
    rng = np.random.default_rng(seed)
    S = B[:int(FRAC * len(B))].astype(np.float32)
    N = S.shape[1]
    M = np.eye(N, dtype=np.float32)
    Ms = [M]
    for k in range(L - 1):
        if S.shape[1] < 4:
            break
        if rule == "spatial":
            p = pair_spatial(M, xy)
        else:
            J = infer_J_est(S, k) if rule.startswith("inference") else None
            p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
        Ms.append(M)
    return Ms


def closure_masked(S0, M, test_mask=None, residual=False):
    """Closure as in closure_test.closure; test_mask selects held-out transitions (on t);
    residual=True regresses the population mean out of units and blocks (fit on training)."""
    T = len(S0)
    Tt = int(FRAC * T)
    Y = S0 @ (M / M.sum(0))
    dX, dY = S0 - S0[:Tt].mean(0), Y - Y[:Tt].mean(0)
    if residual:
        g = dX.mean(1, keepdims=True)
        gg = float((g[:Tt] ** 2).sum())
        dX = dX - g @ ((g[:Tt].T @ dX[:Tt]) / gg)
        dY = dY - g @ ((g[:Tt].T @ dY[:Tt]) / gg)
    Xte, Yte_prev, Yte = dX[Tt:-1], dY[Tt:-1], dY[Tt + 1:]
    if test_mask is not None:
        m = test_mask[Tt:-1]
        Xte, Yte_prev, Yte = Xte[m], Yte_prev[m], Yte[m]
    r2c = ridge_r2(dY[:Tt - 1], dY[1:Tt], Yte_prev, Yte)
    r2m = ridge_r2(dX[:Tt - 1], dY[1:Tt], Xte, Yte)
    return r2c / r2m if r2m > 0 else np.nan


def burst_mask(B, win=5, nsd=2.0):
    pop = B.mean(1)
    sm = np.convolve(pop, np.ones(win) / win, mode="same")
    return sm > sm.mean() + nsd * sm.std()


def main():
    os.makedirs("results", exist_ok=True)
    store = dict(np.load(OUT)) if os.path.exists(OUT) else {}
    for c in CULT:
        if f"{c}/random/closure" in store:
            continue
        t0 = time.time()
        B_all, rates, chans = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        B, xy = B_all[:, :N], coords(chans[:N])
        S0 = B.astype(np.float32)
        bm = burst_mask(B)
        store[f"{c}/burst_fraction"] = bm.mean()
        for rule in RULES:
            Ms = memberships(B, rule, xy)
            cl = np.full(L, np.nan); cp = np.full(L, np.nan); cin = np.full(L, np.nan); cout = np.full(L, np.nan); cres = np.full(L, np.nan)
            for k, M in enumerate(Ms):
                cl[k] = closure_masked(S0, M)
                cp[k] = compactness(M, xy)
                cin[k] = closure_masked(S0, M, bm)
                cout[k] = closure_masked(S0, M, ~bm)
                cres[k] = closure_masked(S0, M, residual=True)
            for q, v in [("closure", cl), ("compact", cp), ("closure_burst", cin), ("closure_quiet", cout), ("closure_resid", cres)]:
                store[f"{c}/{rule}/{q}"] = v
        np.savez(OUT, **store)
        adv = lambda q, r: np.nanmean((store[f"{c}/{r}/{q}"] - store[f"{c}/correlation/{q}"])[1:7])
        print(f"{c}: N {N}, bursts {bm.mean():.2f} of bins | compactness step 5 (electrodes): "
              + " ".join(f"{r[:9]} {store[f'{c}/{r}/compact'][4]:.1f}" for r in RULES)
              + f" | adv output vs corr: all {adv('closure', 'inference_out'):+.3f} burst {adv('closure_burst', 'inference_out'):+.3f} quiet {adv('closure_quiet', 'inference_out'):+.3f} resid {adv('closure_resid', 'inference_out'):+.3f}"
              + f" | spatial vs corr {adv('closure', 'spatial'):+.3f}  [{time.time() - t0:.0f} s]", flush=True)
    summary(store)
    plot(store)


def summary(store):
    dr = np.load("results/directedness.npz")
    x = np.array([float(dr[f"cultures/{c}/lag1"]) for c in CULT])
    print("\n=== mean over cultures (steps 2-7); Spearman with lag-1 directedness")
    for q in ["closure", "closure_burst", "closure_quiet", "closure_resid"]:
        for r in ["inference_out", "inference", "spatial", "random"]:
            a = np.array([np.nanmean((store[f"{c}/{r}/{q}"] - store[f"{c}/correlation/{q}"])[1:7]) for c in CULT])
            print(f"  {q:14s} {r[:13]:13s} vs correlation: {a.mean():+.4f} ({np.sum(a > 0)}/9, p={wilcoxon(a).pvalue:.3f}), rho(dir) {spearmanr(x, a)[0]:+.2f} p={spearmanr(x, a)[1]:.3f}")
    a = np.array([np.nanmean((store[f"{c}/inference_out/closure"] - store[f"{c}/spatial/closure"])[1:7]) for c in CULT])
    print(f"  output vs spatial: {a.mean():+.4f} ({np.sum(a > 0)}/9, p={wilcoxon(a).pvalue:.3f}), rho(dir) {spearmanr(x, a)[0]:+.2f}")
    print("\n=== compactness (mean within-block electrode distance) at steps 3, 5, 7, mean over cultures")
    for r in RULES:
        v = np.nanmean([store[f"{c}/{r}/compact"] for c in CULT], 0)
        print(f"  {r[:13]:13s} " + " ".join(f"step {k + 1}: {v[k]:.1f}" for k in (2, 4, 6)))


def plot(store):
    dr = np.load("results/directedness.npz")
    x = np.array([float(dr[f"cultures/{c}/lag1"]) for c in CULT])
    steps = np.arange(1, L + 1)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    ax = axs[0]
    for r in RULES:
        v = np.array([store[f"{c}/{r}/compact"] for c in CULT])
        ax.errorbar(steps, np.nanmean(v, 0), np.nanstd(v, 0) / 3, fmt="o-", color=COL[r], ms=3, capsize=2, label=LAB[r])
    ax.set_xlabel("RG step"); ax.set_ylabel("within-block electrode distance (pitch units)"); ax.set_title("(a) block compactness", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    for r in RULES:
        v = np.array([store[f"{c}/{r}/closure"] for c in CULT])
        ax.errorbar(steps, np.nanmean(v, 0), np.nanstd(v, 0) / 3, fmt="o-", color=COL[r], ms=3, capsize=2, label=LAB[r])
    ax.axhline(1, color="0.6", ls=":"); ax.set_xlabel("RG step"); ax.set_ylabel("closure"); ax.set_title("(b) closure, incl. spatial rule (9 cultures)", fontsize=9, loc="left")
    ax = axs[2]
    for q, mk, lab in [("closure_burst", "o", "inside bursts"), ("closure", "x", "all transitions")]:
        a = [np.nanmean((store[f"{c}/inference_out/{q}"] - store[f"{c}/correlation/{q}"])[1:7]) for c in CULT]
        ax.scatter(x, a, marker=mk, s=30, label=lab, color={"o": "#8b0000", "x": "0.3"}[mk])
    ax.axhline(0, color="0.6", ls=":"); ax.set_xlabel("lag-1 directedness"); ax.set_ylabel("advantage of the output rule over correlation"); ax.set_title("(c) inside network bursts", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    ax = axs[3]
    for q, mk, lab in [("closure", "x", "original"), ("closure_resid", "o", "global mode removed")]:
        a = [np.nanmean((store[f"{c}/inference_out/{q}"] - store[f"{c}/correlation/{q}"])[1:7]) for c in CULT]
        ax.scatter(x, a, marker=mk, s=30, label=lab, color={"x": "0.3", "o": "#8b0000"}[mk])
    a = [np.nanmean((store[f"{c}/inference_out/closure"] - store[f"{c}/spatial/closure"])[1:7]) for c in CULT]
    ax.scatter(x, a, marker="^", s=30, label="output vs spatial rule", color="#2a9d8f")
    ax.axhline(0, color="0.6", ls=":"); ax.set_xlabel("lag-1 directedness"); ax.set_ylabel("advantage of the output rule"); ax.set_title("(d) global mode, spatial baseline", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_mechanism.png", dpi=170)
    fig.savefig("Figures/Supp_mechanism.pdf")
    print("saved Figures/Supp_mechanism.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        st = dict(np.load(OUT))
        summary(st)
        plot(st)
    else:
        main()
