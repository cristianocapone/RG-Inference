"""
Fig. 3 stimulation test, non-degenerate version (referee control).

In repro_figure3*.py the stimulus delivered to the blocks is I (c_2a - c_2b): a block that
mixes 2a and 2b receives no net input by construction, so the correlation-based model
cannot route.  Here the stimulus is delivered to 2a only: in the full network, a field I
on the units of 2a; in a reduced model, a field I c_2a on every block (c_2a = fraction of
2a units in the block), which is what the block receives from a microscopic stimulation
of 2a.  The reference is the response of the full network (same protocol, true
couplings): the probability that the cycle continues with pattern 3 rather than 4.
Reduced models: correlation, inference (in+out), inference (output), exact ML with
rank-truncated similarity as in repro_figure3_ml.py (N = 1024, T = 1e5), level 4
(64 blocks).  Robustness to the gain: every curve is computed at the tuned gain and at
the tuned gain +/- 0.1.
Results: results/fig3_fair.npz; figure Figures/Fig3_fair.png.
Run from the folder holding sim_cache/ and results/:  python fig3_fair.py [long|plot]
(`long`: 2e5 steps and 3 seeds per intensity, i.e. about 30x more cycles; reduced models cached in results/fig3_fair_models.npz)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import repro_figure3 as F
from repro_figure3_ml import L2, CACHE                 # also sets F.N = 1024
from pairing_tests import lowrank_gap
from ml_kinetic_ising import fit_kinetic_ising
from repro_figures_v2 import synaptic_similarity, greedy_pairs

RULES = ["correlation", "inference", "inference_out"]
IS = np.linspace(0, 12.0, 25)          # field on the blocks (level 4)
IS_FULL = np.linspace(0, 1.6, 17)      # field on the units of 2a in the full network (scale of beta J S)
T_STIM = 20000
MIN_CYCLES = 20
MODELS = "results/fig3_fair_models.npz"       # cached reduced models (ML fits are the expensive part)
LONG = dict(T=200000, seeds=3)                  # `long` mode: 10x more cycles per intensity, 3 sampling seeds


def similarity(J, rule):
    if rule == "inference":
        return synaptic_similarity(J)
    V = J.T / (np.linalg.norm(J.T, axis=1, keepdims=True) + 1e-12)      # columns of J: outputs
    return V @ V.T


def rg_ml(S0, lab, rule, rng):
    perm = rng.permutation(S0.shape[1])
    S = S0[:, perm].astype(np.float32)
    comp = np.eye(7)[lab[perm]]
    for k in range(F.LEVEL):
        if rule.startswith("inference"):
            J = fit_kinetic_ising(S, nlev=2 ** k, l2=L2, maxiter=300)[0]
            Jl, r = lowrank_gap(J)
            G = similarity(Jl, rule)
        else:
            G = np.nan_to_num(np.corrcoef(S.T))
        np.fill_diagonal(G, -np.inf)
        p = greedy_pairs(G)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
        print(f"  {rule} level {k + 1}: {S.shape[1]} blocks, purity {comp.max(1).mean():.3f}", flush=True)
    J, h, _ = fit_kinetic_ising(S, nlev=2 ** F.LEVEL, l2=L2, maxiter=300)
    return S, J, h, comp


def p34(S, comp):
    """P(3), P(4) over the cycles of the run; undefined when the stimulus stalls the cycle."""
    c = F.cycles(F.overlaps(S, comp))
    return (c[:, 1].mean(), c[:, 2].mean(), len(c)) if len(c) >= MIN_CYCLES else (np.nan, np.nan, len(c))


def best_sep(p3, p4):
    d = np.asarray(p3) - np.asarray(p4)
    return np.nanmax(d) if np.any(~np.isnan(d)) else np.nan


def main():
    J, xi, lab = F.seq_network()
    S0 = np.load(CACHE)
    comp0 = np.eye(7)[lab]
    res = {"Is": IS, "Is_full": IS_FULL}
    # ---- reference: the full network stimulated on 2a
    t0 = time.time()
    long = len(sys.argv) > 1 and sys.argv[1] == "long"
    p3, p4, ncyc = [], [], []
    for I in IS_FULL:
        a_, b_, n_ = [], [], 0
        for sd in range(LONG["seeds"] if long else 1):
            Sg = F.glauber(J, np.full(F.N, F.H_EXT / F.BETA), F.BETA, LONG["T"] // 2 if long else T_STIM, np.random.default_rng(3 + sd), S0=xi[1], I=I * (lab == 1))
            c = F.cycles(F.overlaps(Sg, comp0))
            if len(c):
                a_ += list(c[:, 1]); b_ += list(c[:, 2]); n_ += len(c)
        ok = n_ >= MIN_CYCLES * (10 if long else 1)
        p3.append(np.mean(a_) if ok else np.nan); p4.append(np.mean(b_) if ok else np.nan); ncyc.append(n_)
    res["full/p3"], res["full/p4"], res["full/n"] = np.array(p3), np.array(p4), np.array(ncyc)
    print(f"full network: P(3) {np.round(p3, 2)}\n              P(4) {np.round(p4, 2)}\n              max P(3)-P(4) {best_sep(p3, p4):.2f}  [{time.time() - t0:.0f} s]", flush=True)
    # ---- reduced models
    models = dict(np.load(MODELS)) if os.path.exists(MODELS) else {}
    long = len(sys.argv) > 1 and sys.argv[1] == "long"
    T_run, seeds = (LONG["T"], LONG["seeds"]) if long else (T_STIM, 1)
    for rule in RULES:
        t0 = time.time()
        if f"{rule}/J" not in models:
            S, Jk, hk, comp = rg_ml(S0, lab, rule, np.random.default_rng(1))
            o = F.block_order(comp)
            Jk, hk, comp = Jk[np.ix_(o, o)], hk[o], comp[o]
            g, cv = F.tune_gain(Jk, hk, comp, np.random.default_rng(2), np.linspace(0.4, 1.4, 11))
            models.update({f"{rule}/J": Jk, f"{rule}/h": hk, f"{rule}/comp": comp, f"{rule}/gain": g, f"{rule}/cv": cv})
            np.savez(MODELS, **models)
        Jk, hk, comp, g = models[f"{rule}/J"], models[f"{rule}/h"], models[f"{rule}/comp"], float(models[f"{rule}/gain"])
        t = comp[:, 1] + comp[:, 5]
        sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
        res[f"{rule}/sep"] = float(sep.mean()) if len(sep) else np.nan
        res[f"{rule}/gain"] = g
        print(f"{rule}: 2a/2b separation {res[f'{rule}/sep']:.2f}, gain* {g:.2f}", flush=True)
        for dg in (-0.1, 0.0, 0.1):
            p3, p4, ncyc = [], [], []
            for I in IS:
                a_, b_, n_ = [], [], 0
                for sd in range(seeds):
                    Sg = F.block_dynamics(Jk, hk, g + dg, T_run, np.random.default_rng(3 + sd), I=I * comp[:, 1])
                    c = F.cycles(F.overlaps(Sg, comp))
                    if len(c):
                        a_ += list(c[:, 1]); b_ += list(c[:, 2]); n_ += len(c)
                ok = n_ >= MIN_CYCLES * (10 if long else 1)
                p3.append(np.mean(a_) if ok else np.nan); p4.append(np.mean(b_) if ok else np.nan); ncyc.append(n_)
            res[f"{rule}/g{dg:+.1f}/p3"], res[f"{rule}/g{dg:+.1f}/p4"], res[f"{rule}/g{dg:+.1f}/n"] = np.array(p3), np.array(p4), np.array(ncyc)
            print(f"  gain {g + dg:.2f}: P(3) {np.round(p3, 2)}\n             P(4) {np.round(p4, 2)}\n             cycles {ncyc}\n             max P(3)-P(4) {best_sep(p3, p4):.2f}", flush=True)
        print(f"  [{time.time() - t0:.0f} s]", flush=True)
    os.makedirs("results", exist_ok=True)
    np.savez("results/fig3_fair_long.npz" if long else "results/fig3_fair.npz", **res)
    plot(res)


def plot(res):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    col = {"full": "k", "correlation": "#8c8c00", "inference": "#d62728", "inference_out": "#8b0000"}
    ax = axs[0]
    ax.plot(res["Is_full"], res["full/p3"], "o-", color="k", ms=3, label="P(3)")
    ax.plot(res["Is_full"], res["full/p4"], "s--", color="k", ms=3, label="P(4)")
    ax.set_title("full network (N = 1024)\nstimulus on the units of 2a", fontsize=9); ax.legend(frameon=False, fontsize=8)
    for ax, rule in zip(axs[1:], RULES):
        for dg, a in ((-0.1, 0.35), (0.0, 1.0), (0.1, 0.35)):
            ax.plot(res["Is"], res[f"{rule}/g{dg:+.1f}/p3"], "o-", color=col[rule], ms=3, alpha=a, label="P(3)" if dg == 0 else None)
            ax.plot(res["Is"], res[f"{rule}/g{dg:+.1f}/p4"], "s--", color=col[rule], ms=3, alpha=a, label="P(4)" if dg == 0 else None)
        ax.set_title(f"{rule.replace('inference_out', 'inference (output)').replace('inference', 'inference (in+out)') if rule != 'inference_out' else 'inference (output)'} (64 blocks)\n"
                     f"2a/2b separation {float(res[f'{rule}/sep']):.2f}, gain {float(res[f'{rule}/gain']):.2f} (faint: +/- 0.1)", fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.set_xlabel("stimulus on 2a, I (units of the model's field)"); ax.axhline(0.5, color="0.7", ls=":")
    axs[0].set_ylabel("probability that the cycle continues with 3 / 4")
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Fig3_fair_long.png" if (len(sys.argv) > 1 and sys.argv[1] == "long") else "Figures/Fig3_fair.png", dpi=170)
    print("saved Figures/Fig3_fair.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(dict(np.load("results/fig3_fair.npz")))
    else:
        main()
