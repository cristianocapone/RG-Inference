"""
Experimental prediction test: does the model inferred on spontaneous (basal)
activity predict the response of the same culture to a pharmacological perturbation?

Lonardoni et al. 2017 (Zenodo 4977727) recorded the same cultures before and after
bicuculline (BIC, GABA_A block; d1-d4) or APV (NMDA block; d1-d2).  Protocol:
  1. bin basal and drug recordings at dt = 10 ms on the common channels, keep the
     N most active basal channels (N <= T/100);
  2. infer the kinetic Ising model (J, h) on basal; calibrate a global gain so that
     the sampled model reproduces the basal mean rate;
  3. perturb the model in silico: BIC -> remove negative couplings (J_ij < 0 := 0),
     APV -> scale positive couplings by (1 - beta); sample; predicted change of
     each channel = log(rate_pred / rate_basal_model);
  4. compare with the observed per-channel change log(rate_drug / rate_basal):
     Spearman correlation, and partial Spearman controlling for the basal log-rate
     (low-rate channels increase more under BIC for trivial reasons);
  5. null: remove the same number of couplings chosen at random (sign-agnostic),
     20 repetitions -> z-score of the real prediction;
  6. same at the coarse-grained level: RG with inference / inference_out /
     correlation / random pairing to 64 and 32 blocks, coarse model fitted on basal
     blocks, perturbed in silico, predicted vs observed per-block change
     (observed = membership-weighted channel changes).  This is the paper's claim
     in experimental form: which reduction keeps the information needed to predict
     the response to a real perturbation?
Results: results/drug_prediction.npz; figure Figures/Drug_prediction.png.

Run:  env -i /opt/homebrew/bin/python3 drug_prediction.py <dir with BIC/ and APV/ folders>
"""

import os
import sys
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import infer_J, _FG, _HG
from ml_kinetic_ising import fit_kinetic_ising
from closure_test import pairs5
from rg_metrics import COLORS

DT = 0.010
PAIRS = [("BIC", "d3"), ("BIC", "d4"), ("BIC", "d1"), ("APV", "d1")]
RULES = ["inference", "inference_out", "correlation", "random"]
COL = dict(COLORS, inference_out="#8b0000")
N_NULL = 20
LEVELS = (3, 4)
USE_ML = True            # exact ML with L2 instead of nMF (nMF J saturates the sampled model on these sparse data)
L2 = 1e-3
N_MAX = 300
BETAS = (0.5, 1.0)


# --------------------------------------------------------------------------- #
def load_pair(root, cond, cult, dt=DT):
    a = sio.loadmat(f"{root}/{cond}/{cult}_basal.mat")
    b = sio.loadmat(f"{root}/{cond}/{cult}_{cond}.mat")
    chans = sorted(set(k for k in a if k.startswith("Ch")) & set(k for k in b if k.startswith("Ch")))
    fs = float(a["SamplingFrequency"].squeeze()) if "SamplingFrequency" in a else 7022.0

    def binarise(d):
        n_frames = int(d["StopFrame"].squeeze() - d["StartFrame"].squeeze()) if "StopFrame" in d else d[chans[0]].shape[0]
        T = int(n_frames / fs / dt)
        B = np.zeros((T, len(chans)), np.uint8)
        for c, k in enumerate(chans):
            x = d[k]
            frames = x.tocoo().row if sp.issparse(x) else np.flatnonzero(np.asarray(x).ravel())
            bb = (frames / fs / dt).astype(int)
            B[bb[bb < T], c] = 1
        return B

    return binarise(a), binarise(b), chans


def fit(S, level):
    if USE_ML:
        J, h, info = fit_kinetic_ising(S, nlev=2 ** level, l2=L2, maxiter=300)
        return J, h
    J = infer_J(S, level)
    m = np.clip(S.mean(0), 1e-4, 1 - 1e-4)
    Hbar = np.log(m / (1 - m)) if level == 0 else np.interp(m, _FG, _HG)
    return J, Hbar - J @ m


def sample(J, h, gain, nlev, T, rng):
    n = len(J)
    v = np.arange(nlev + 1) / nlev
    S = np.zeros(n)
    out = np.zeros(n)
    for t in range(T):
        H = gain * (J @ S + h)
        if nlev == 1:
            S = (rng.random(n) < 1 / (1 + np.exp(-H))).astype(float)
        else:
            L = np.outer(H, v)
            L -= L.max(1, keepdims=True)
            p = np.exp(L)
            c = np.cumsum(p / p.sum(1, keepdims=True), 1)
            S = v[np.minimum((rng.random(n)[:, None] > c).sum(1), nlev)]
        if t >= 500:
            out += S
    return out / (T - 500)


def calibrate(J, h, nlev, target_mean, rng, T=4000):
    best, berr = 1.0, np.inf
    for g in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5):
        r = sample(J, h, g, nlev, T, rng).mean()
        e = abs(np.log((r + 1e-4) / (target_mean + 1e-4)))
        if e < berr:
            best, berr = g, e
    return best


def perturb(J, cond, beta=1.0):
    Jp = J.copy()
    if cond == "BIC":
        Jp[Jp < 0] *= (1 - beta)
    else:
        Jp[Jp > 0] *= (1 - beta)
    return Jp


def perturb_random(J, cond, rng, beta=1.0):
    """Remove the same number of couplings, chosen at random regardless of sign."""
    Jp = J.copy()
    n_rm = int((J < 0).sum() if cond == "BIC" else (J > 0).sum())
    idx = rng.choice(J.size, n_rm, replace=False)
    Jp.ravel()[idx] *= (1 - beta)
    return Jp


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        A = np.c_[np.ones_like(b), b]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]


def predict(J, h, nlev, cond, gain, T, rng, rate_model_basal):
    rp = sample(perturb(J, cond), h, gain, nlev, T, rng)
    return np.log((rp + 1e-3) / (rate_model_basal + 1e-3))


def evaluate(pred, obs, basal_log_rate):
    rho = spearmanr(pred, obs)[0]
    prho = partial_spearman(pred, obs, basal_log_rate)
    return rho, prho


# --------------------------------------------------------------------------- #
def run_pair(root, cond, cult, store):
    t0 = time.time()
    Ba, Bb, chans = load_pair(root, cond, cult)
    Ta, Tb = Ba.shape[0], Bb.shape[0]
    rates_a, rates_b = Ba.mean(0) / DT, Bb.mean(0) / DT
    N = min(N_MAX, Ta // 200, len(chans)) // 2 * 2
    keep = np.argsort(-rates_a)[:N]
    Ba, Bb = Ba[:, keep], Bb[:, keep]
    ra, rb = rates_a[keep], rates_b[keep]
    obs = np.log((rb + 1e-3) / (ra + 1e-3))
    lra = np.log(ra + 1e-3)
    tag = f"{cond}/{cult}"
    print(f"=== {tag}: {len(chans)} common channels, N={N}, basal {Ta * DT / 60:.0f} min, drug {Tb * DT / 60:.0f} min; "
          f"observed change mean {obs.mean():+.2f}, corr(basal log-rate, change) {np.corrcoef(lra, obs)[0, 1]:+.2f}  [{time.time() - t0:.0f} s]", flush=True)
    rng = np.random.default_rng(0)
    S = Ba.astype(np.float32)
    T_sim = 20000

    # ---- micro model ----
    J, h = fit(S, 0)
    gain = calibrate(J, h, 1, S.mean(), rng)
    r0 = sample(J, h, gain, 1, T_sim, rng)
    off = ~np.eye(N, dtype=bool)
    print(f"  micro model: frac J<0 {(J[off] < 0).mean():.2f}, mean|J| {np.abs(J[off]).mean():.3f}; gain {gain}; "
          f"basal model mean {r0.mean() / DT:.2f} Hz (data {S.mean() / DT:.2f}), Spearman(model rate, data rate) {spearmanr(r0, ra)[0]:+.2f}", flush=True)
    inh = J.clip(max=0).sum(1)
    print(f"  direct predictors: Spearman(-inhibitory input, obs) {spearmanr(-inh, obs)[0]:+.3f}, partial {partial_spearman(-inh, obs, lra):+.3f}", flush=True)
    for beta in BETAS:
        rp = sample(perturb(J, cond, beta), h, gain, 1, T_sim, rng)
        pred = np.log((rp + 1e-3) / (r0 + 1e-3))
        rho, prho = evaluate(pred, obs, lra)
        null = []
        for s_ in range(N_NULL):
            rpn = sample(perturb_random(J, cond, np.random.default_rng(100 + s_), beta), h, gain, 1, 6000, rng)
            null.append(evaluate(np.log((rpn + 1e-3) / (r0 + 1e-3)), obs, lra)[0])
        null = np.array(null)
        z = (rho - null.mean()) / (null.std() + 1e-12)
        store[f"{tag}/micro/b{beta}/rho"], store[f"{tag}/micro/b{beta}/prho"], store[f"{tag}/micro/b{beta}/z"] = rho, prho, z
        print(f"  micro beta {beta}: perturbed mean {rp.mean() / DT:.2f} Hz, predicted change {pred.mean():+.2f} (obs {obs.mean():+.2f}), "
              f"frac saturated {np.mean(rp > 0.95):.2f}; Spearman {rho:+.3f}, partial {prho:+.3f}; null {null.mean():+.3f} +/- {null.std():.3f}, z = {z:+.1f}", flush=True)
    beta = BETAS[-1]
    pred = np.log((sample(perturb(J, cond, beta), h, gain, 1, T_sim, rng) + 1e-3) / (r0 + 1e-3))
    store[f"{tag}/micro/rho"], store[f"{tag}/micro/prho"] = store[f"{tag}/micro/b{beta}/rho"], store[f"{tag}/micro/b{beta}/prho"]
    # trivial baseline: predict from basal rate only
    print(f"  baseline Spearman(-basal rate, obs) {spearmanr(-lra, obs)[0]:+.3f}", flush=True)

    # ---- coarse models ----
    for level in LEVELS:
        nb = N // 2 ** level
        for rule in RULES:
            t1 = time.time()
            Sk, M = S, np.eye(N, dtype=np.float32)
            rr = np.random.default_rng(1)
            for k in range(level):
                Jk = infer_J(Sk, k)
                p = pairs5(rule, Sk, Jk, rr)
                Sk = 0.5 * (Sk[:, p[:, 0]] + Sk[:, p[:, 1]])
                M = M[:, p[:, 0]] + M[:, p[:, 1]]
            Mn = M / M.sum(0)
            obs_b = obs @ Mn                                # membership-weighted observed change per block
            lra_b = lra @ Mn
            if nb < 16:
                continue
            Jk, hk = fit(Sk, level)
            g = calibrate(Jk, hk, 2 ** level, Sk.mean(), rng)
            r0b = sample(Jk, hk, g, 2 ** level, T_sim, rng)
            pred_b = np.log((sample(perturb(Jk, cond, BETAS[-1]), hk, g, 2 ** level, T_sim, rng) + 1e-3) / (r0b + 1e-3))
            rho_b, prho_b = evaluate(pred_b, obs_b, lra_b)
            # also: micro prediction aggregated to the same blocks (upper reference)
            rho_micro_b = spearmanr(pred @ Mn, obs_b)[0]
            store[f"{tag}/L{level}/{rule}/rho"], store[f"{tag}/L{level}/{rule}/prho"] = rho_b, prho_b
            store[f"{tag}/L{level}/{rule}/rho_micro_agg"] = rho_micro_b
            print(f"  level {level} ({nb} blocks) {rule:13s}: Spearman {rho_b:+.3f}  partial {prho_b:+.3f}  "
                  f"(micro aggregated to these blocks {rho_micro_b:+.3f})  [{time.time() - t1:.0f} s]", flush=True)


def main(root):
    os.makedirs("results", exist_ok=True)
    store = {}
    for cond, cult in PAIRS:
        try:
            run_pair(root, cond, cult, store)
        except Exception as e:                              # keep going on a bad file
            print(f"  !! {cond}/{cult} failed: {e!r}", flush=True)
        np.savez("results/drug_prediction.npz", **store)
    plot(store)


def plot(store):
    tags = sorted({k.split("/L")[0].split("/micro")[0] for k in store})
    fig, axs = plt.subplots(1, len(LEVELS) + 1, figsize=(4.5 * (len(LEVELS) + 1), 4))
    x = np.arange(len(tags))
    ax = axs[0]
    ax.bar(x - 0.2, [store.get(f"{t}/micro/rho", np.nan) for t in tags], 0.4, color="k", label="Spearman")
    ax.bar(x + 0.2, [store.get(f"{t}/micro/prho", np.nan) for t in tags], 0.4, color="0.6", label="partial | basal rate")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8)
    ax.set_title("micro model: predicted vs observed per-channel change", fontsize=9)
    ax.axhline(0, color="0.6", ls=":")
    ax.legend(fontsize=8, frameon=False)
    for j, level in enumerate(LEVELS):
        ax = axs[j + 1]
        w = 0.8 / len(RULES)
        for i, rule in enumerate(RULES):
            ax.bar(x + (i - 1.5) * w, [store.get(f"{t}/L{level}/{rule}/rho", np.nan) for t in tags], w, color=COL[rule], label=rule)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=8)
        ax.set_title(f"reduced models, level {level}: Spearman(pred, obs) per block", fontsize=9)
        ax.axhline(0, color="0.6", ls=":")
        if j == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("Figures/Drug_prediction.png", dpi=170)
    print("saved Figures/Drug_prediction.png")


if __name__ == "__main__":
    main(sys.argv[1])
