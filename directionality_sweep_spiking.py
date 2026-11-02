"""
Directionality as a control parameter (spiking E/I network).

Clustered LIF E/I network with delays (as in spiking_ei_rg.py, run tag
d5j0.5_s0.15_Nref3000_wclu1.9) with an extra E->E projection from cluster c to
cluster c+1 of relative strength `chain` in {0, 0.05, 0.1, 0.2}: chain = 0 is the
symmetric (competitive) network, larger chain = more directed propagation.
For every network, on a 512-unit subset (400 E + 112 I, 10 ms bins):
  directedness   1 - symmetry of D(1) of the activity; asymmetry of the full-network
                 perturbation-response matrix R_full (|R - R^T| / |R + R^T|, off-diag)
  closure        at RG levels 3 and 4, rules inference / inference_out / oneshot /
                 correlation / random, plus PCA at the same dimension
  perturbation   correlation (off-diagonal) between the response matrix of the
                 reduced model (level 3) and R_full, 5 seeds, plus the micro model
Figure: Figures/Directionality_sweep_spiking.png; results: results/dir_sweep_spiking.npz

Run:  env -i /opt/homebrew/bin/python3 directionality_sweep_spiking.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = [sys.argv[0], "40", "0.5", "sweep"]
import spiking_ei_rg as spk
import perturbation_transfer as P
from repro_figures_v2 import infer_J
from closure_test import pairs5, closure
from directionality_sweep_hopfield import oneshot_membership, pca_closure, COL
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

CHAINS = [0.0, 0.05, 0.1, 0.2]
RULES = ["inference", "inference_out", "oneshot", "correlation", "random"]
NET_KW = dict(Nref=3000, wclu=1.9)
SIM = dict(delay_ms=5.0, delay_jitter=0.5, sigma=0.15)
SEEDS = range(5)


def cache_name(c):
    return {0.0: "clustered", 0.1: "chain"}.get(c, f"chain{c:g}")


def membership(S0, rule, level, seed=0):
    rng = np.random.default_rng(seed)
    if rule == "oneshot":
        return oneshot_membership(infer_J(S0, 0), level)
    S, M = S0, np.eye(S0.shape[1], dtype=np.float32)
    for k in range(level):
        J = infer_J(S, k)
        p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return M


def main():
    os.makedirs("results", exist_ok=True)
    store = {}
    off16 = ~np.eye(spk.NCLU, dtype=bool)
    for c in CHAINS:
        t0 = time.time()
        ts, ids, W, clu = spk.get_spikes(cache_name(c), c, 600, SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
        B = spk.bin_spikes(ts, ids, P.BIN)
        rng0 = np.random.default_rng(0)
        sub = np.sort(np.r_[rng0.choice(spk.NE, 400, replace=False), spk.NE + rng0.choice(spk.NI, 112, replace=False)])
        S0 = B[:, sub].astype(np.float32)
        clu_sub = np.r_[clu, np.full(spk.NI, -1)][sub]
        Wc0 = np.stack([(clu_sub == k).astype(np.float32) for k in range(spk.NCLU)], 1)
        _, D = lagged_cov(S0)
        store[f"{c}/symD"] = symmetry(D)
        # full-network perturbation
        R1 = P.run_full(W, clu, np.random.default_rng(3))
        R2 = P.run_full(W, clu, np.random.default_rng(4))
        R_full = 0.5 * (R1 + R2)
        Ra, Rs = 0.5 * (R_full - R_full.T), 0.5 * (R_full + R_full.T)
        store[f"{c}/R_asym"] = np.linalg.norm(Ra[off16]) / np.linalg.norm(Rs[off16])
        store[f"{c}/ceiling"] = P.corr(R1, R2, True)
        self_target = float(np.mean(np.diag(R_full)))
        print(f"=== chain {c}: symmetry(D1) {symmetry(D):.2f}, response asymmetry {store[f'{c}/R_asym']:.2f}, "
              f"ceiling {store[f'{c}/ceiling']:.2f}  [{time.time() - t0:.0f} s]", flush=True)
        J0, h0 = P.fit_model(S0, 0)
        v = [P.corr(P.model_response(J0, h0, 1, Wc0, S0, np.random.default_rng(s), self_target)[0], R_full, True) for s in SEEDS]
        store[f"{c}/micro/pert"] = np.array(v)
        print(f"  micro model: perturbation corr {np.mean(v):.3f} +/- {np.std(v):.3f}", flush=True)
        for level in (3, 4):
            store[f"{c}/pca/closure{level}"] = pca_closure(S0, S0.shape[1] // 2 ** level)
        for rule in RULES:
            t0 = time.time()
            cl, pert = {3: [], 4: []}, []
            for s in SEEDS:
                for level in (3, 4):
                    M = membership(S0, rule, level, seed=s)
                    r2c, r2m = closure(S0, M)
                    cl[level].append(r2c / r2m)
                    if level == 3:
                        y = S0 @ (M / M.sum(0))
                        J, h = P.fit_model(y, level)
                        R = P.model_response(J, h, 2 ** level, M.T @ Wc0, y, np.random.default_rng(10 + s), self_target)[0]
                        pert.append(P.corr(R, R_full, True))
            for level in (3, 4):
                store[f"{c}/{rule}/closure{level}"] = np.array(cl[level])
            store[f"{c}/{rule}/pert"] = np.array(pert)
            print(f"  {rule:14s} closure L3 {np.mean(cl[3]):.3f}+/-{np.std(cl[3]):.3f}  L4 {np.mean(cl[4]):.3f}+/-{np.std(cl[4]):.3f}"
                  f"  perturbation L3 {np.mean(pert):.3f}+/-{np.std(pert):.3f}  [{time.time() - t0:.0f} s]", flush=True)
        print(f"  {'pca':14s} closure L3 {store[f'{c}/pca/closure3']:.3f}  L4 {store[f'{c}/pca/closure4']:.3f}", flush=True)
        np.savez("results/dir_sweep_spiking.npz", **store)
    plot(store)


def plot(store):
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    x = np.array(CHAINS)
    axs[0].plot(x, [1 - store[f"{c}/symD"] for c in CHAINS], "ks--", label="1 - symmetry(D1)")
    axs[0].plot(x, [store[f"{c}/R_asym"] for c in CHAINS], "ko-", label="asymmetry of R_full")
    axs[0].set_ylabel("directedness")
    axs[0].legend(fontsize=8, frameon=False)
    for rule in RULES:
        for ax, key in [(axs[1], "closure3"), (axs[2], "closure4"), (axs[3], "pert")]:
            A = np.stack([store[f"{c}/{rule}/{key}"] for c in CHAINS])
            ax.errorbar(x, A.mean(1), A.std(1), fmt="o-", color=COL[rule], ms=3, capsize=2, label=rule)
    for ax, lv in [(axs[1], 3), (axs[2], 4)]:
        ax.plot(x, [store[f"{c}/pca/closure{lv}"] for c in CHAINS], "o-", color=COL["pca"], ms=3, label="pca")
        ax.set_ylabel(f"closure, level {lv}")
        ax.axhline(1, color="0.6", ls=":")
    A = np.stack([store[f"{c}/micro/pert"] for c in CHAINS])
    axs[3].errorbar(x, A.mean(1), A.std(1), fmt="k--", capsize=2, label="micro model")
    axs[3].plot(x, [store[f"{c}/ceiling"] for c in CHAINS], "k:", label="ceiling")
    axs[3].set_ylabel("perturbation transfer, level 3 (corr off-diag)")
    for ax in axs:
        ax.set_xlabel("chain strength (0 = symmetric clusters)")
    axs[1].legend(fontsize=7, frameon=False)
    axs[3].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("Figures/Directionality_sweep_spiking.png", dpi=170)
    print("saved Figures/Directionality_sweep_spiking.png")


if __name__ == "__main__":
    main()
