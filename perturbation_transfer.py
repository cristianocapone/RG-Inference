"""
Perturbation-transfer test: does the coarse-grained model predict the response of the
full network to a stimulus?  (Fig. 3 generalised and quantified.)

Ground truth: the clustered spiking E/I network with delays (same run as the RG
analyses, 10 ms bins).  Protocol in the full LIF network: for each of the 16 E
clusters, 20 trials of a 100 ms extra drive to all its E units; response of every
cluster = E rate in [0, 150] ms after onset minus the pre-stimulus baseline, averaged
over trials -> R_full (16 x 16, row = stimulated cluster).
Models: on a 512-unit subset (400 E + 112 I) of the spontaneous recording, the
kinetic-Ising model is inferred at RG level k (blocks of 2^k units) for every pairing
rule; the coarse model is sampled (2^k + 1 activity levels) with the same protocol,
the stimulus being delivered to each block in proportion to its membership of the
stimulated cluster; block responses are mapped back to clusters by membership ->
R_model.  Gain of the model is calibrated on the spontaneous block activity, the
stimulus amplitude so that the mean self-response matches the full network.
Metric: Pearson correlation between R_model and R_full, over all entries and over
off-diagonal entries only (network propagation of the perturbation).
Reference: the micro model (level 0, 512 units, binary Glauber).
Figure: Figures/Perturbation_transfer.png; results: results/perturbation.npz.

Run:  env -i /opt/homebrew/bin/python3 perturbation_transfer.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spiking_ei_rg as spk
from repro_figures_v2 import infer_J, _FG, _HG
from closure_test import pairs5, RULES5, COL5

TAG = "d5j0.5_s0.15_Nref3000_wclu1.9"
SIM_KW = dict(delay_ms=5.0, delay_jitter=0.5, sigma=0.15)
BIN = 10.0
import sys
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
A_FULL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3    # extra drive in the LIF (threshold units)
NET = sys.argv[3] if len(sys.argv) > 3 else "clustered"
ON_MS, OFF_MS = 100.0, 400.0
LEVELS = (3, 4)


# --------------------------------------------------------------------------- #
#  Full network
# --------------------------------------------------------------------------- #
def protocol(n_clusters, n_trials, on_ms, off_ms, dt):
    """Returns (onset steps, stimulated cluster) and total duration in ms."""
    period = on_ms + off_ms
    order = np.tile(np.arange(n_clusters), n_trials)
    onsets = (1000.0 + np.arange(len(order)) * period) / dt
    return onsets.astype(int), order, 1000.0 + len(order) * period + off_ms


def response_matrix(B_cl, onsets_bins, order, on_bins, n_clusters, win=15, base=15):
    R = np.zeros((n_clusters, n_clusters))
    for c in range(n_clusters):
        rows = []
        for o in onsets_bins[order == c]:
            if o + win < len(B_cl) and o - base >= 0:
                rows.append(B_cl[o:o + win].mean(0) - B_cl[o - base:o].mean(0))
        R[c] = np.mean(rows, 0)
    return R


def run_full(W, clu, rng):
    onsets, order, T_ms = protocol(spk.NCLU, N_TRIALS, ON_MS, OFF_MS, spk.DT)
    on_steps = int(ON_MS / spk.DT)
    drive = np.zeros((spk.NCLU, spk.N))
    for c in range(spk.NCLU):
        drive[c, :spk.NE][clu == c] = A_FULL
    idx = 0

    def stim(step):
        nonlocal idx
        while idx < len(onsets) and step >= onsets[idx] + on_steps:
            idx += 1
        if idx < len(onsets) and onsets[idx] <= step < onsets[idx] + on_steps:
            return drive[order[idx]]
        return None

    t0 = time.time()
    ts, ids = spk.simulate(W, T_ms, rng, stim=stim, **SIM_KW)
    print(f"  full network stimulated for {T_ms / 1000:.0f} s in {time.time() - t0:.0f} s", flush=True)
    B = spk.bin_spikes(ts, ids, BIN, skip_ms=0.0)
    B_cl = np.stack([B[:, :spk.NE][:, clu == c].mean(1) for c in range(spk.NCLU)], 1)
    onsets_bins = (onsets * spk.DT / BIN).astype(int)
    return response_matrix(B_cl, onsets_bins, order, int(ON_MS / BIN), spk.NCLU)


# --------------------------------------------------------------------------- #
#  Coarse models
# --------------------------------------------------------------------------- #
def fit_model(S, level):
    J = infer_J(S, level)
    m = np.clip(S.mean(0), 1e-4, 1 - 1e-4)
    Hbar = np.log(m / (1 - m)) if level == 0 else np.interp(m, _FG, _HG)
    return J, Hbar - J @ m


def sample(J, h, gain, nlev, T, rng, I=None):
    """Kinetic Ising with 2^k + 1 levels; I: callable bin -> extra field or None."""
    n = len(J)
    v = np.arange(nlev + 1) / nlev
    S = np.zeros(n)
    out = np.empty((T, n), np.float32)
    for t in range(T):
        H = gain * (J @ S + h)
        if I is not None:
            e = I(t)
            if e is not None:
                H = H + e
        if nlev == 1:
            S = (rng.random(n) < 1 / (1 + np.exp(-H))).astype(float)
        else:
            Lg = np.outer(H, v)
            Lg -= Lg.max(1, keepdims=True)
            p = np.exp(Lg)
            c = np.cumsum(p / p.sum(1, keepdims=True), 1)
            S = v[np.minimum((rng.random(n)[:, None] > c).sum(1), nlev)]
        out[t] = S
    return out


def rg_membership(S0, rule, level, seed=0):
    rng = np.random.default_rng(seed)
    S = S0
    M = np.eye(S0.shape[1], dtype=np.float32)
    for k in range(level):
        J = infer_J(S, k)
        p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return S, M


def model_response(J, h, nlev, Wc, y_data, rng, R_self_target):
    """Wc: blocks x clusters membership (E units). Calibrates gain on the spontaneous
    block activity, then the stimulus amplitude on the mean self-response."""
    T_spont = 6000
    gains = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    err = []
    for g in gains:
        y = sample(J, h, g, nlev, T_spont, rng)
        err.append(abs(y.mean() - y_data.mean()) / y_data.mean() + abs(y.std(0).mean() - y_data.std(0).mean()) / y_data.std(0).mean())
    gain = gains[int(np.argmin(err))]
    on_b, off_b = int(ON_MS / BIN), int(OFF_MS / BIN)
    order = np.tile(np.arange(spk.NCLU), N_TRIALS)
    onsets = 100 + np.arange(len(order)) * (on_b + off_b)
    T = onsets[-1] + on_b + off_b
    Wn = Wc / (Wc.sum(0, keepdims=True) + 1e-12)          # blocks x clusters, columns sum to 1

    def run(a):
        def I(t):
            i = np.searchsorted(onsets, t, side="right") - 1
            if i >= 0 and t < onsets[i] + on_b:
                return a * Wc[:, order[i]]
            return None
        y = sample(J, h, gain, nlev, T, rng, I)
        y_cl = y @ Wn                                       # cluster activity from blocks
        return response_matrix(y_cl, onsets, order, on_b, spk.NCLU, win=15, base=10)

    amps = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    Rs = [run(a) for a in amps]
    selfs = np.array([np.mean(np.diag(R)) for R in Rs])
    i = int(np.argmin(np.abs(selfs - R_self_target)))
    return Rs[i], gain, amps[i], selfs[i]


def corr(R1, R2, off=False):
    m = ~np.eye(len(R1), dtype=bool) if off else np.ones_like(R1, bool)
    return np.corrcoef(R1[m], R2[m])[0, 1]


def main():
    os.makedirs("results", exist_ok=True)
    d = np.load(f"sim_cache/{NET}_{TAG}_spikes_T600.npz")
    W, clu = d["W"], d["clu"]
    R1 = run_full(W, clu, np.random.default_rng(3))
    R2 = run_full(W, clu, np.random.default_rng(4))
    R_full = 0.5 * (R1 + R2)
    self_target = float(np.mean(np.diag(R_full)))
    off = ~np.eye(spk.NCLU, dtype=bool)
    print(f"  R_full ({N_TRIALS} trials x 2 runs, amp {A_FULL}): mean self-response {self_target:.3f}, "
          f"off-diag mean {R_full[off].mean():+.4f} std {R_full[off].std():.4f}; split-half reliability of off-diag: "
          f"{corr(R1, R2, True):.2f} (ceiling), all entries {corr(R1, R2):.2f}", flush=True)

    B = spk.bin_spikes(d["ts"], d["ids"], BIN)
    sub = np.sort(np.r_[np.random.default_rng(0).choice(spk.NE, 400, replace=False),
                        spk.NE + np.random.default_rng(0).choice(spk.NI, 112, replace=False)])
    S0 = B[:, sub].astype(np.float32)
    clu_sub = np.r_[clu, np.full(spk.NI, -1)][sub]
    Wc0 = np.stack([(clu_sub == c).astype(np.float32) for c in range(spk.NCLU)], 1)   # units x clusters

    res = {"R_full": R_full, "R_full_1": R1, "R_full_2": R2}
    # micro model reference (level 0)
    J0, h0 = fit_model(S0, 0)
    R0, g0, a0, s0 = model_response(J0, h0, 1, Wc0, S0, np.random.default_rng(1), self_target)
    res["micro"] = R0
    print(f"  micro model (N=512): gain {g0}, amp {a0}, self {s0:.3f}  corr all {corr(R0, R_full):.2f}  off-diag {corr(R0, R_full, True):.2f}", flush=True)

    for level in LEVELS:
        for rule in RULES5:
            t0 = time.time()
            S, M = rg_membership(S0, rule, level)
            J, h = fit_model(S, level)
            Wc = M.T @ Wc0                                   # blocks x clusters (counts of E units)
            R, g, a, s = model_response(J, h, 2 ** level, Wc, S, np.random.default_rng(1), self_target)
            res[f"L{level}/{rule}"] = R
            print(f"  level {level} ({S.shape[1]} blocks) {rule:13s}: gain {g}, amp {a}, self {s:.3f}  "
                  f"corr all {corr(R, R_full):.2f}  off-diag {corr(R, R_full, True):.2f}  [{time.time() - t0:.0f} s]", flush=True)
    np.savez(f"results/perturbation_{NET}_t{N_TRIALS}_a{A_FULL:g}.npz", **res)
    plot(res)


def plot(res):
    R_full = res["R_full"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    vmax = np.abs(R_full).max()
    axs[0, 0].imshow(R_full, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[0, 0].set_title("full LIF network", fontsize=9)
    axs[0, 1].imshow(res["micro"], cmap="RdBu_r", vmin=-np.abs(res["micro"]).max(), vmax=np.abs(res["micro"]).max())
    axs[0, 1].set_title(f"micro model, corr off-diag {corr(res['micro'], R_full, True):.2f}", fontsize=9)
    for j, rule in enumerate(["inference", "correlation", "random"]):
        R = res[f"L{LEVELS[-1]}/{rule}"]
        ax = axs[0, 2 + j] if j < 2 else axs[1, 0]
        ax.imshow(R, cmap="RdBu_r", vmin=-np.abs(R).max(), vmax=np.abs(R).max())
        ax.set_title(f"level {LEVELS[-1]} {rule}, corr off-diag {corr(R, R_full, True):.2f}", fontsize=9)
    for ax in axs.ravel()[:5]:
        ax.set_xlabel("responding cluster")
        ax.set_ylabel("stimulated cluster")
    ax = axs[1, 1]
    x = np.arange(len(RULES5))
    for i, level in enumerate(LEVELS):
        ax.bar(x + 0.35 * i, [corr(res[f"L{level}/{r}"], R_full, True) for r in RULES5], 0.35,
               color=[COL5[r] for r in RULES5], alpha=0.5 + 0.5 * i, label=f"level {level}")
    ax.axhline(corr(res["micro"], R_full, True), color="k", ls="--", label="micro model")
    ax.set_xticks(x + 0.17)
    ax.set_xticklabels(RULES5, rotation=30, fontsize=8)
    ax.set_ylabel("corr(R_model, R_full), off-diagonal")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 2]
    for i, level in enumerate(LEVELS):
        ax.bar(x + 0.35 * i, [corr(res[f"L{level}/{r}"], R_full) for r in RULES5], 0.35,
               color=[COL5[r] for r in RULES5], alpha=0.5 + 0.5 * i)
    ax.axhline(corr(res["micro"], R_full), color="k", ls="--")
    ax.set_xticks(x + 0.17)
    ax.set_xticklabels(RULES5, rotation=30, fontsize=8)
    ax.set_ylabel("corr(R_model, R_full), all entries")
    axs[1, 3].axis("off")
    fig.tight_layout()
    fig.savefig(f"Figures/Perturbation_transfer_{NET}_t{N_TRIALS}_a{A_FULL:g}.png", dpi=170)
    print(f"saved Figures/Perturbation_transfer_{NET}_t{N_TRIALS}_a{A_FULL:g}.png")


if __name__ == "__main__":
    main()
