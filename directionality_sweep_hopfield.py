"""
Directionality as a control parameter (Hopfield network with transitions).

J = J0 + alpha [ (1 - gamma) J_T + gamma (J_T + J_T^T)/2 ]: gamma = 0 is the directed
sequence network of Fig. 3 (1 -> 2 -> {3|4} -> 5 -> 1), gamma = 1 has the same
transition strength but bidirectional (undirected) transitions.  N = 512, beta = 55,
h = -2, T = 2e5, 5 patterns of 64 units, 2a/2b sub-populations.
For every gamma and every reduction rule we measure, at a fixed reduced dimension
(32 blocks, RG level 4):
  closure         R2(y(t+1)|y(t)) / R2(y(t+1)|x(t)), held-out (closure_test.py)
  transition fid. Pearson correlation between the pattern-to-pattern transition
                  matrix of the sampled reduced model and the one of the full
                  network (off-diagonal, 5 patterns)
Rules: inference (iterative, in+out), inference_out, oneshot (pairing from the
level-0 J only, never re-inferred), correlation, random, and PCA (32 components; for
closure only).  Ground-truth directedness of each network: 1 - symmetry(J) and the
time-irreversibility of the full dynamics (1 - symmetry of D(1)).
Figure: Figures/Directionality_sweep_hopfield.png; results: results/dir_sweep_hopfield.npz

Run:  env -i /opt/homebrew/bin/python3 directionality_sweep_hopfield.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import repro_figure3 as F3
from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity, _FG, _HG
from closure_test import pairs5, closure, ridge_r2
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

N, ETA, ALPHA, BETA, H_EXT = 512, 1 / 8, 0.2, 55.0, -2.0
LEVEL = 4
T = 200_000
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RULES = ["inference", "inference_out", "oneshot", "correlation", "random"]
COL = {"inference": "#d62728", "inference_out": "#8b0000", "oneshot": "#ff7f0e",
       "correlation": "#8c8c00", "lagged_out": "#5e3c99", "random": "#4a7fd6", "pca": "#7f7f7f"}


def seq_network(gamma):
    s = int(N * ETA)
    xi = np.zeros((5, N))
    for m in range(5):
        xi[m, m * s:(m + 1) * s] = 1
    sa, sb = np.zeros(N), np.zeros(N)
    sa[s:s + s // 2] = 1
    sb[s + s // 2:2 * s] = 1
    X = xi - ETA
    J0 = X.T @ X / N
    JT = (np.outer(xi[1], xi[0]) + np.outer(xi[4], xi[3]) + np.outer(xi[4], xi[2]) + np.outer(xi[0], xi[4])) / N
    JT += 2 * (np.outer(xi[2], sa) + np.outer(xi[3], sb)) / N
    JT = (1 - gamma) * JT + gamma * 0.5 * (JT + JT.T)
    J = J0 + ALPHA * JT
    np.fill_diagonal(J, 0)
    return J, xi


def pattern_transitions(act, thr=0.6, smooth=5):
    """act: T x 5 pattern activities -> 5 x 5 transition count matrix between
    consecutive distinct dominant patterns."""
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(act.astype(float), smooth, axis=0)
    dom = np.where(sm.max(1) > thr, sm.argmax(1), -1)
    seq = [d for i, d in enumerate(dom) if d >= 0 and (i == 0 or d != dom[i - 1])]
    seq = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
    C = np.zeros((5, 5))
    for a, b in zip(seq[:-1], seq[1:]):
        C[a, b] += 1
    return C / max(C.sum(), 1), len(seq)


def oneshot_membership(J0, level, out=False):
    """Hierarchical pairing from the level-0 similarity only (no re-inference):
    at each step pair blocks by the mean level-0 similarity between their members
    (rows+columns of J0, or columns only with out=True)."""
    if out:
        V = J0.T / (np.linalg.norm(J0.T, axis=1, keepdims=True) + 1e-12)
        G0 = V @ V.T
    else:
        G0 = synaptic_similarity(J0)
    n = len(J0)
    M = np.eye(n, dtype=np.float32)
    for k in range(level):
        Mn = M / M.sum(0)
        G = Mn.T @ G0 @ Mn
        np.fill_diagonal(G, -np.inf)
        p = greedy_pairs(G)
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return M


def membership(S0, rule, level, seed=0):
    rng = np.random.default_rng(seed)
    if rule == "oneshot":
        return oneshot_membership(infer_J(S0, 0), level)
    S = S0
    M = np.eye(S0.shape[1], dtype=np.float32)
    for k in range(level):
        J = infer_J(S, k)
        p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return M


def pca_closure(S0, k, frac=0.8):
    Tt = int(frac * len(S0))
    mx = S0[:Tt].mean(0)
    dX = S0 - mx
    U, s, Vt = np.linalg.svd(dX[:Tt], full_matrices=False)
    P = Vt[:k].T
    dY = dX @ P
    r2c = ridge_r2(dY[:Tt - 1], dY[1:Tt], dY[Tt:-1], dY[Tt + 1:])
    r2m = ridge_r2(dX[:Tt - 1], dY[1:Tt], dX[Tt:-1], dY[Tt + 1:])
    return r2c / r2m


def reduced_model_transitions(S0, M, level, xi, rng, T_sim=60_000):
    y = S0 @ (M / M.sum(0))
    J, h = fit(y, level)
    # gain calibrated on the mean block activity
    best, berr = 1.0, np.inf
    for g in (0.6, 0.8, 1.0, 1.2):
        ys = F3.block_dynamics(J, h, g, 6000, rng, nlev=2 ** level)
        e = abs(ys.mean() - y.mean()) / y.mean() + abs(ys.std(0).mean() - y.std(0).mean()) / y.std(0).mean()
        if e < berr:
            best, berr = g, e
    ys = F3.block_dynamics(J, h, best, T_sim, rng, nlev=2 ** level)
    Wp = (M.T @ xi.T) / M.sum(0)[:, None]                 # blocks x patterns membership
    Wp = Wp / (Wp.sum(0, keepdims=True) + 1e-12)
    act = ys @ Wp
    return pattern_transitions(act)


def fit(S, level):
    J = infer_J(S, level)
    m = np.clip(S.mean(0), 1e-4, 1 - 1e-4)
    Hbar = np.log(m / (1 - m)) if level == 0 else np.interp(m, _FG, _HG)
    return J, Hbar - J @ m


def main():
    os.makedirs("results", exist_ok=True)
    store = {}
    for gamma in GAMMAS:
        t0 = time.time()
        J, xi = seq_network(gamma)
        S0 = F3.glauber(J, np.full(N, H_EXT / BETA), BETA, T + 500, np.random.default_rng(0), S0=xi[1])[500:].astype(np.float32)
        _, D = lagged_cov(S0)
        act_full = S0 @ xi.T / xi.sum(1)
        P_full, n_tr = pattern_transitions(act_full)
        store[f"{gamma}/symJ"] = symmetry(J)
        store[f"{gamma}/symD"] = symmetry(D)
        print(f"=== gamma {gamma}: symmetry(J) {symmetry(J):.2f}, symmetry(D1) {symmetry(D):.2f}, "
              f"{n_tr} pattern transitions, P_full rows:\n{np.round(P_full, 2)}  [{time.time() - t0:.0f} s]", flush=True)
        off = ~np.eye(5, dtype=bool)
        for rule in RULES:
            t0 = time.time()
            M = membership(S0, rule, LEVEL)
            r2c, r2m = closure(S0, M)
            P_red, n_red = reduced_model_transitions(S0, M, LEVEL, xi, np.random.default_rng(1))
            fid = np.corrcoef(P_red[off], P_full[off])[0, 1] if P_red[off].std() > 0 else np.nan
            store[f"{gamma}/{rule}/closure"] = r2c / r2m
            store[f"{gamma}/{rule}/fidelity"] = fid
            print(f"  {rule:14s} closure {r2c / r2m:.3f}  transition fidelity {fid:+.2f} ({n_red} transitions)  [{time.time() - t0:.0f} s]", flush=True)
        store[f"{gamma}/pca/closure"] = pca_closure(S0, N // 2 ** LEVEL)
        print(f"  {'pca':14s} closure {store[f'{gamma}/pca/closure']:.3f}", flush=True)
    np.savez("results/dir_sweep_hopfield.npz", **store)
    plot(store)


def plot(store):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    g = np.array(GAMMAS)
    direct = [1 - store[f"{x}/symD"] for x in GAMMAS]
    axs[0].plot(g, [1 - store[f"{x}/symJ"] for x in GAMMAS], "ko-", label="1 - symmetry(J)")
    axs[0].plot(g, direct, "ks--", label="1 - symmetry(D1), dynamics")
    axs[0].set_xlabel("gamma (0 = directed transitions, 1 = bidirectional)")
    axs[0].set_ylabel("directedness")
    axs[0].legend(fontsize=8, frameon=False)
    for rule in RULES + ["pca"]:
        axs[1].plot(g, [store[f"{x}/{rule}/closure"] for x in GAMMAS], "o-", color=COL[rule], label=rule)
        if rule != "pca":
            axs[2].plot(g, [store[f"{x}/{rule}/fidelity"] for x in GAMMAS], "o-", color=COL[rule], label=rule)
    axs[1].set_ylabel("closure at 32 blocks")
    axs[2].set_ylabel("transition-matrix fidelity of reduced model")
    for ax in axs[1:]:
        ax.set_xlabel("gamma")
        ax.axhline(1 if ax is axs[1] else 0, color="0.6", ls=":")
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("Figures/Directionality_sweep_hopfield.png", dpi=170)
    print("saved Figures/Directionality_sweep_hopfield.png")


if __name__ == "__main__":
    main()
