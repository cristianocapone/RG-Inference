"""
Does a model with a temporal kernel recover, in the chain networks, the directional
advantage that the one-step model cannot see?

Kinetic Ising model with an exponential presynaptic kernel: S_i(t+1) is predicted from
G_j(t) = (1-a) sum_{k>=0} a^k S_j(t-k), a = exp(-1/tau) (tau in bins of 10 ms; tau = 0 is
the one-step model).  Couplings by mean-field inversion of the filtered statistics,
J = <dS(t+1) dG(t)> <dG dG>^-1 / f'.  Pairing rules as in closure_test (inference in+out,
inference output, model-free lagged-correlation output, correlation, random), formed on
the training part with the kernel couplings re-inferred at every level; closure evaluated
with the same kernel: R^2[y(t+1) | G_y(t)] / R^2[y(t+1) | G_S(t)] held-out, at 32 blocks.
Directedness of each network over the window of the kernel (lags 1..3 tau; lag 1 for the
one-step model).  Chain 0, 0.05, 0.1, 0.2; 512-unit subsets as in closure_test.py.
Results: results/kernel_closure_spiking.npz.  Run from the folder holding sim_cache/.
"""

import sys
import time
import numpy as np
from scipy.signal import lfilter

sys.argv = [sys.argv[0], "40", "0.5", "sweep"]
import spiking_ei_rg as spk
from directionality_sweep_spiking import cache_name, NET_KW, SIM, CHAINS
from repro_figures_v2 import greedy_pairs, synaptic_similarity, _FG, _HG, _FPG
from closure_test import ridge_r2, FRAC
from directedness import directedness

TAUS = [0, 3, 10, 30]
LEVEL = 4
RULES = ["inference", "inference_out", "lagged_out", "correlation", "random"]


def filt(S, tau):
    if tau == 0:
        return S
    a = np.exp(-1.0 / tau)
    return lfilter([1.0 - a], [1.0, -a], S, axis=0).astype(np.float32)


def infer_J_filtered(S, G, level):
    m = S.mean(0)
    dS, dG = S[1:] - m, G[:-1] - G[:-1].mean(0)
    C, D = dG.T @ dG / len(dS), dS.T @ dG / len(dS)
    mc = np.clip(m, 1e-4, 1 - 1e-4)
    fp = mc * (1 - mc) if level == 0 else np.interp(np.interp(mc, _FG, _HG), _HG, _FPG)
    C += 1e-6 * np.trace(C) / len(C) * np.eye(len(C), dtype=C.dtype)
    J = np.linalg.solve(C.T, D.T).T / fp[:, None]
    np.fill_diagonal(J, 0)
    return J


def lagged_corr_filtered(S, G):
    dS, dG = S[1:] - S.mean(0), G[:-1] - G[:-1].mean(0)
    R = dS.T @ dG / len(dS) / np.outer(dS.std(0) + 1e-12, dG.std(0) + 1e-12)
    np.fill_diagonal(R, 0)
    return R


def pairs(rule, S, G, level, rng):
    if rule == "random":
        return rng.permutation(S.shape[1] // 2 * 2).reshape(-1, 2)
    if rule == "correlation":
        Gm = np.nan_to_num(np.corrcoef(S.T))
    else:
        J = infer_J_filtered(S, G, level) if rule.startswith("inference") else lagged_corr_filtered(S, G)
        if rule.endswith("_out"):
            V = J.T / (np.linalg.norm(J.T, axis=1, keepdims=True) + 1e-12)
            Gm = V @ V.T
        else:
            Gm = synaptic_similarity(J)
    np.fill_diagonal(Gm, -np.inf)
    return greedy_pairs(Gm)


def membership(S0, tau, rule, seed=0):
    rng = np.random.default_rng(seed)
    S = S0[:int(FRAC * len(S0))]
    G = filt(S, tau)
    M = np.eye(S.shape[1], dtype=np.float32)
    for k in range(LEVEL):
        p = pairs(rule, S, G, k, rng)
        S, G = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]]), 0.5 * (G[:, p[:, 0]] + G[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return M


def closure_tau(S0, M, tau):
    Tt = int(FRAC * len(S0))
    Y = S0 @ (M / M.sum(0))
    FY, FX = filt(Y, tau), filt(S0, tau)
    dY, dFY, dFX = Y - Y[:Tt].mean(0), FY - FY[:Tt].mean(0), FX - FX[:Tt].mean(0)
    r2c = ridge_r2(dFY[:Tt - 1], dY[1:Tt], dFY[Tt:-1], dY[Tt + 1:])
    r2m = ridge_r2(dFX[:Tt - 1], dY[1:Tt], dFX[Tt:-1], dY[Tt + 1:])
    return r2c / r2m


def main():
    store = {}
    for c in CHAINS:
        ts, ids, W, clu = spk.get_spikes(cache_name(c), c, 600, SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
        B = spk.bin_spikes(ts, ids, 10.0)
        rng0 = np.random.default_rng(0)
        sub = np.sort(np.r_[rng0.choice(spk.NE, 400, replace=False), spk.NE + rng0.choice(spk.NI, 112, replace=False)])
        S0 = B[:, sub].astype(np.float32)
        for tau in TAUS:
            t0 = time.time()
            store[f"{c}/{tau}/dir"] = directedness(S0, tuple(range(1, 3 * tau + 1)) if tau else (1,))[0]
            vals = {}
            for rule in RULES:
                v = [closure_tau(S0, membership(S0, tau, rule, seed=s), tau) for s in (range(3) if rule == "random" else [0])]
                vals[rule] = float(np.mean(v))
                store[f"{c}/{tau}/{rule}"] = vals[rule]
            print(f"chain {c:4} tau {tau:2d}: dir {store[f'{c}/{tau}/dir']:.3f} | " + "  ".join(f"{r} {vals[r]:.3f}" for r in RULES)
                  + f" | adv out {vals['inference_out'] - vals['correlation']:+.3f} in+out {vals['inference'] - vals['correlation']:+.3f}"
                  f" lagged {vals['lagged_out'] - vals['correlation']:+.3f}  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/kernel_closure_spiking.npz", **store)


if __name__ == "__main__":
    main()
