"""
RG observables of real-space coarse-graining under the different pairing rules
(referee request: do the scaling observables of Meshulam et al. depend on the rule?).

For every recording (9 cultures, 3 mice) and every rule (inference in+out, inference
output, correlation, random) the RG is run on the whole recording (these are
descriptive observables, not predictions) and at every level k (blocks of K = 2^k
units) we compute, as in Meshulam et al. 2019:
  M2(K)   variance of the summed block activity, mean over blocks   -> alpha  (M2 ~ K^alpha)
  F(K)    -log P(block silent), mean over blocks                     -> beta   (F ~ K^beta)
  tau(K)  autocorrelation time of the block activity (first lag at which the
          autocorrelation falls below 1/e, mean over blocks)         -> z      (tau ~ K^z)
  spectrum: eigenvalues of the covariance of the block variables, rank-ordered,
          against rank / number of blocks                            -> collapse across levels
Exponents are least-squares slopes in log-log over levels 0..5 (K = 1..32).
Results: results/rg_observables.npz; figure Figures/Supp_rgobs.png; table on stdout.
Run from the folder holding data/ and results/:  python rg_observables.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from closure_test import pairs5, infer_J_est
from real_data_rg import load_culture
from invivo_stringer import bin_mouse, MICE

RULES = ["inference", "inference_out", "correlation", "random"]
COL = {"inference": "#d62728", "inference_out": "#8b0000", "correlation": "#8c8c00", "random": "#4a7fd6"}
LAB = {"inference": "inference (in+out)", "inference_out": "inference (output)", "correlation": "correlation", "random": "random"}
CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
N_LEVELS = 7
FIT_LEVELS = slice(0, 6)
OUT = "results/rg_observables.npz"


def levels(B, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    out = [S]
    for k in range(n_levels - 1):
        if S.shape[1] < 4:
            break
        J = infer_J_est(S, k) if rule.startswith("inference") else None
        p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        out.append(S)
    return out


def autocorr_time(S, max_lag=100):
    d = S - S.mean(0)
    v = (d ** 2).mean(0) + 1e-12
    taus = []
    for j in range(S.shape[1]):
        for lag in range(1, max_lag + 1):
            if (d[lag:, j] * d[:-lag, j]).mean() / v[j] < np.exp(-1):
                taus.append(lag)
                break
        else:
            taus.append(max_lag)
    return float(np.mean(taus))


def observables(lv):
    K = 2.0 ** np.arange(len(lv))
    M2 = np.array([np.var(S * k, axis=0).mean() for S, k in zip(lv, K)])
    P0 = np.array([np.clip((S == 0).mean(0), 1e-6, 1).mean() for S in lv])
    tau = np.array([autocorr_time(S) for S in lv])
    spec = [np.sort(np.linalg.eigvalsh(np.cov(S.T)))[::-1] for S in lv[:6]]
    return K, M2, -np.log(P0), tau, spec


def slope(K, y, sl=FIT_LEVELS):
    x, y = np.log(K[sl]), np.log(y[sl])
    return float(np.polyfit(x, y, 1)[0])


def datasets():
    for c in CULT:
        B_all, _, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        yield "cultures", c, B_all[:, :N]
    for mouse, tstart in MICE.items():
        B, *_ = bin_mouse(f"data/stringer2019_neuropixels/spks{mouse}_Feb18.mat", tstart)
        yield "mice", mouse, B


def main():
    os.makedirs("results", exist_ok=True)
    store = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
    for grp, name, B in datasets():
        t0 = time.time()
        for rule in RULES:
            key = f"{grp}/{name}/{rule}"
            if f"{key}/M2" in store:
                continue
            K, M2, F, tau, spec = observables(levels(B, rule, N_LEVELS))
            store[f"{key}/K"], store[f"{key}/M2"], store[f"{key}/F"], store[f"{key}/tau"] = K, M2, F, tau
            for k, s in enumerate(spec):
                store[f"{key}/spec{k}"] = s
        np.savez(OUT, **store)
        line = "  ".join(f"{r[:9]} a={slope(store[f'{grp}/{name}/{r}/K'], store[f'{grp}/{name}/{r}/M2']):.2f} "
                         f"b={slope(store[f'{grp}/{name}/{r}/K'], store[f'{grp}/{name}/{r}/F']):.2f} "
                         f"z={slope(store[f'{grp}/{name}/{r}/K'], store[f'{grp}/{name}/{r}/tau']):.2f}" for r in RULES)
        print(f"{grp}/{name}: {line}  [{time.time() - t0:.0f} s]", flush=True)
    summary(store)
    plot(store)


def summary(store):
    print("\n=== exponents (levels 0-5), mean +/- s.e.m. over recordings; paired Wilcoxon vs correlation")
    for grp, names in [("cultures", CULT), ("mice", list(MICE))]:
        for q, lab in [("M2", "alpha"), ("F", "beta"), ("tau", "z")]:
            vals = {r: np.array([slope(store[f"{grp}/{n}/{r}/K"], store[f"{grp}/{n}/{r}/{q}"]) for n in names]) for r in RULES}
            txt = "  ".join(f"{r[:9]} {v.mean():.3f}+/-{v.std(ddof=1) / np.sqrt(len(v)):.3f}" for r, v in vals.items())
            pv = {r: (wilcoxon(vals[r] - vals["correlation"]).pvalue if len(names) >= 5 and r != "correlation" else np.nan) for r in RULES}
            print(f"  {grp:9s} {lab:5s}: {txt} | p vs corr: " + " ".join(f"{r[:9]} {p:.2g}" for r, p in pv.items() if r != "correlation"))


def plot(store):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for row, (grp, names) in enumerate([("cultures", CULT), ("mice", list(MICE))]):
        for rule in RULES:
            K = store[f"{grp}/{names[0]}/{rule}/K"]
            for j, (q, ylab) in enumerate([("M2", "variance of block activity"), ("F", "-log P(silent)"), ("tau", "autocorrelation time (bins)")]):
                Y = np.array([store[f"{grp}/{n}/{rule}/{q}"][:len(K)] for n in names])
                m = np.exp(np.nanmean(np.log(Y), 0))
                e = np.nanstd(np.log(Y), 0) / np.sqrt(len(names))
                axs[row, j].errorbar(K, m, yerr=[m - m * np.exp(-e), m * np.exp(e) - m], fmt="o-", color=COL[rule], ms=3, capsize=2, label=LAB[rule])
                axs[row, j].set_xscale("log", base=2); axs[row, j].set_yscale("log")
                axs[row, j].set_xlabel("block size K"); axs[row, j].set_ylabel(ylab)
            ax = axs[row, 3]
            for k in range(6):
                s = store[f"{grp}/{names[0]}/{rule}/spec{k}"]
                s = s[s > 0]
                ax.plot(np.arange(1, len(s) + 1) / len(s), s / s[0], "-", color=COL[rule], lw=0.8, alpha=0.4 + 0.1 * k)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rank / number of blocks"); ax.set_ylabel("eigenvalue / largest")
        axs[row, 0].set_title(f"{grp}: variance scaling", fontsize=9, loc="left")
        axs[row, 1].set_title("silence", fontsize=9, loc="left")
        axs[row, 2].set_title("dynamics", fontsize=9, loc="left")
        axs[row, 3].set_title(f"covariance spectrum along the RG ({names[0]})", fontsize=9, loc="left")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_rgobs.png", dpi=170)
    fig.savefig("Figures/Supp_rgobs.pdf")
    print("saved Figures/Supp_rgobs.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        st = dict(np.load(OUT))
        summary(st)
        plot(st)
    else:
        main()
