"""
Does the reduced model, run freely, reproduce the sequence of cluster activations of the
spiking network?  Spiking analogue of the routing test of Fig. 3, along the chain sweep.

For each network (chain 0, 0.05, 0.1, 0.2) and reduction rule, at 32 blocks (level 4):
  * the reduced kinetic Ising model is fitted by exact maximum likelihood on the
    spontaneous block activity (512-unit subset, training part) and run freely for T_SIM
    bins, with no free parameter (gain 1: the fitted model as it is);
  * the cluster activity of the model is read from the blocks through their cluster
    composition; the dominant cluster per bin (z-scored, smoothed, above threshold) gives
    a sequence of cluster activations and a 16 x 16 transition matrix;
  * the same is done on the full spiking network (all excitatory neurons) and on the
    unreduced inferred model (512 units).
Metrics: Pearson correlation of the off-diagonal transition probabilities between model and
network; "sequentiality" P(c -> c+1) of the model and of the network (chance 1/15).
Results: results/directional_sequence_spiking.npz.  Run from the folder holding sim_cache/:
    python directional_sequence_spiking.py [chain ...]
"""

import sys
import time
import numpy as np

ARGS = [float(a) for a in sys.argv[1:]]
sys.argv = [sys.argv[0], "40", "0.5", "sweep"]
import spiking_ei_rg as spk
import perturbation_transfer as P
from directionality_sweep_spiking import membership, cache_name, NET_KW, SIM, CHAINS
from closure_test import FRAC
from ml_kinetic_ising import fit_kinetic_ising

RULES = ["inference", "inference_out", "oneshot", "lagged_out", "correlation", "random"]
LEVEL = 4
T_SIM = 60_000
SEEDS = range(5)
GAIN = 1.0            # ML-fitted models are run as fitted


def sequence_matrix(A, smooth=5, z_thr=1.0, min_run=3):
    """A: T x n_clusters activity -> row-normalised transition matrix between dominant clusters."""
    k = np.ones(smooth) / smooth
    As = np.stack([np.convolve(A[:, c], k, mode="same") for c in range(A.shape[1])], 1)
    Z = (As - As.mean(0)) / (As.std(0) + 1e-12)
    dom = np.where(Z.max(1) > z_thr, Z.argmax(1), -1)
    runs = []
    t = 0
    while t < len(dom):
        u = t
        while u < len(dom) and dom[u] == dom[t]:
            u += 1
        if dom[t] >= 0 and u - t >= min_run and (not runs or runs[-1] != dom[t]):
            runs.append(dom[t])
        t = u
    n = A.shape[1]
    Tm = np.zeros((n, n))
    for a, b in zip(runs[:-1], runs[1:]):
        Tm[a, b] += 1
    return Tm / np.maximum(Tm.sum(1, keepdims=True), 1), len(runs)


def seq_index(Tm):
    n = len(Tm)
    return float(np.mean([Tm[c, (c + 1) % n] for c in range(n)]))


def calibrate(J, h, nlev, Wn, rate_target, rng, T_cal=6000):
    """Gain at which the model's rate of cluster activations (per bin) matches the network's."""
    gains = np.arange(0.5, 1.61, 0.1)
    err = []
    for g in gains:
        y = P.sample(J, h, g, nlev, T_cal, rng)
        _, nr = sequence_matrix(y @ Wn)
        err.append(abs(nr / T_cal - rate_target) / rate_target)
    return gains[int(np.argmin(err))]


def fit(S, nlev):
    J, h, _ = fit_kinetic_ising(S, nlev=nlev, l2=1e-3, maxiter=300)
    return J, h


def main(chains):
    store = {}
    off = ~np.eye(spk.NCLU, dtype=bool)
    for c in chains:
        ts, ids, W, clu = spk.get_spikes(cache_name(c), c, 600, SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
        B = spk.bin_spikes(ts, ids, P.BIN)
        A_full = np.stack([B[:, :spk.NE][:, clu == k].mean(1) for k in range(spk.NCLU)], 1)
        T_full, n_full = sequence_matrix(A_full)
        store[f"{c}/full/T"] = T_full
        rate_target = n_full / len(B)
        print(f"=== chain {c}: network sequentiality P(c->c+1) = {seq_index(T_full):.3f} ({n_full} activations in 600 s)", flush=True)
        rng0 = np.random.default_rng(0)
        sub = np.sort(np.r_[rng0.choice(spk.NE, 400, replace=False), spk.NE + rng0.choice(spk.NI, 112, replace=False)])
        S0 = B[:, sub].astype(np.float32)
        Str = S0[:int(FRAC * len(S0))]
        clu_sub = np.r_[clu, np.full(spk.NI, -1)][sub]
        Wc0 = np.stack([(clu_sub == k).astype(np.float32) for k in range(spk.NCLU)], 1)
        # unreduced inferred model
        J0, h0 = fit(Str, 1)
        Wn0 = Wc0 / (Wc0.sum(0) + 1e-12)
        vals = []
        for s in SEEDS:
            rng = np.random.default_rng(100 + s)
            g = GAIN
            y = P.sample(J0, h0, g, 1, T_SIM, rng)
            Tm, nr = sequence_matrix(y @ Wn0)
            vals.append([np.corrcoef(Tm[off], T_full[off])[0, 1], seq_index(Tm), nr])
        v = np.array(vals); store[f"{c}/micro/seq"] = v
        print(f"  {'micro model':14s} corr(T) {v[:, 0].mean():.3f}+/-{v[:, 0].std():.3f}  P(c->c+1) {v[:, 1].mean():.3f}  activations {v[:, 2].mean():.0f}", flush=True)
        for rule in RULES:
            t0 = time.time()
            vals = []
            for s in SEEDS:
                M = membership(S0, rule, LEVEL, seed=s)
                ytr = Str @ (M / M.sum(0))
                J, h = fit(ytr, 2 ** LEVEL)
                Wc = M.T @ Wc0
                Wn = Wc / (Wc.sum(0) + 1e-12)
                rng = np.random.default_rng(200 + s)
                g = GAIN
                y = P.sample(J, h, g, 2 ** LEVEL, T_SIM, rng)
                Tm, nr = sequence_matrix(y @ Wn)
                vals.append([np.corrcoef(Tm[off], T_full[off])[0, 1], seq_index(Tm), nr, g])
            v = np.array(vals); store[f"{c}/{rule}/seq"] = v
            print(f"  {rule:14s} corr(T) {v[:, 0].mean():.3f}+/-{v[:, 0].std():.3f}  P(c->c+1) {v[:, 1].mean():.3f}  activations {v[:, 2].mean():.0f}  gain {v[:, 3].mean():.2f}  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/directional_sequence_spiking.npz", **store)


if __name__ == "__main__":
    main(ARGS if ARGS else CHAINS)
