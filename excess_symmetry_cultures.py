"""
Fig. 4B with statistics: excess symmetry (data minus shifted surrogate) and
model-free directedness / predictability along the RG, across the Lonardoni 2017
cultures (d1-d8, d15; d9-d14 contain non-binary values and are skipped).

For each culture: N = min(512, T/100, #channels) most active channels, dt = 10 ms,
levels = log2(N) - 1, three pairing rules, data vs one shifted surrogate.
Results saved to results/cultures_metrics.npz; figure Figures/Cultures_excess.png.

Run:  env -i /opt/homebrew/bin/python3 excess_symmetry_cultures.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data_rg import load_culture
from rg_metrics import data_vs_surrogate, pad, RULES, COLORS

CULTURES = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
L = 8
METRICS = ["sym_J", "sym_D", "r2", "r2_auto", "r2_ag", "exp"]


def main():
    os.makedirs("results", exist_ok=True)
    store = {}
    for c in CULTURES:
        t0 = time.time()
        B_all, rates, _ = load_culture(c, 10000)
        T = B_all.shape[0]
        N = min(512, T // 100, B_all.shape[1])
        N -= N % 2
        B = B_all[:, :N]
        n_levels = int(np.log2(N)) - 1
        res = data_vs_surrogate(B, n_levels)
        print(f"=== {c}: T={T} ({T / 6000:.0f} min), N={N}, levels={n_levels}, "
              f"rate median {np.median(rates[:N]):.2f} Hz  [{time.time() - t0:.0f} s]")
        for rule in RULES:
            r, s = res[rule]["real"], res[rule]["surr"]
            print(f"  {rule:12s} sym_J real {np.round(r['sym_J'], 2)}  surr {np.round(s['sym_J'], 2)}")
            print(f"  {'':12s} sym_D real {np.round(r['sym_D'], 2)}  surr {np.round(s['sym_D'], 2)}")
            print(f"  {'':12s} r2-r2auto real {np.round(r['r2'] - r['r2_auto'], 3)}  r2-r2ag {np.round(r['r2'] - r['r2_ag'], 3)}  exps {np.round(r['exp'], 2)}", flush=True)
            for m in METRICS:
                store[f"{c}/{rule}/real/{m}"] = pad(r[m], L)
                store[f"{c}/{rule}/surr/{m}"] = pad(s[m], L)
    np.savez("results/cultures_metrics.npz", **store)
    plot(store)


def agg(store, rule, kind, m):
    A = np.stack([store[f"{c}/{rule}/{kind}/{m}"] for c in CULTURES])
    n = np.sum(~np.isnan(A), 0)
    return np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1)), n


def plot(store):
    steps = np.arange(1, L + 1)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    panels = [
        (axs[0, 0], "sym_J", "real", "symmetry of inferred J (solid: data, dashed: surrogate)", True),
        (axs[0, 1], "sym_J", "excess", "excess symmetry of J (data - surrogate)", False),
        (axs[0, 2], "sym_D", "excess", "excess symmetry of lagged covariance D(1)", False),
        (axs[1, 0], "r2", "real", "held-out R2, one-step linear prediction (dotted: autoregressive only)", True),
        (axs[1, 1], "r2x", "real", "cross-unit predictability R2 - R2_auto (dashed: beyond global mode, R2 - R2_ag)", False),
        (axs[1, 2], "exp", "real", "variance scaling exponent (dashed: surrogate)", True),
    ]
    for ax, m, kind, title, show_surr in panels:
        for rule in RULES:
            if m == "r2x":
                mr, _, _ = agg(store, rule, "real", "r2")
                ma, _, _ = agg(store, rule, "real", "r2_auto")
                y, e = mr - ma, 0 * mr
                A = np.stack([store[f"{c}/{rule}/real/r2"] - store[f"{c}/{rule}/real/r2_auto"] for c in CULTURES])
                e = np.nanstd(A, 0) / np.sqrt(np.sum(~np.isnan(A), 0))
                As = np.stack([store[f"{c}/{rule}/real/r2"] - store[f"{c}/{rule}/real/r2_ag"] for c in CULTURES])
                ax.plot(steps, np.nanmean(As, 0), "--", color=COLORS[rule], lw=1)
            elif kind == "excess":
                A = np.stack([store[f"{c}/{rule}/real/{m}"] - store[f"{c}/{rule}/surr/{m}"] for c in CULTURES])
                y, e = np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.sum(~np.isnan(A), 0))
            else:
                y, e, _ = agg(store, rule, "real", m)
            x = steps[1:] if m == "exp" else steps
            y, e = (y[:L - 1], e[:L - 1]) if m == "exp" else (y, e)
            ax.errorbar(x, y, e, fmt="o-", color=COLORS[rule], ms=3, capsize=2, label=rule)
            if show_surr and m != "r2":
                ys, _, _ = agg(store, rule, "surr", m)
                ax.plot(x, ys[:len(x)], "--", color=COLORS[rule], lw=1)
            if m == "r2":
                ya, _, _ = agg(store, rule, "real", "r2_auto")
                ax.plot(x, ya, ":", color=COLORS[rule], lw=1)
        ax.axhline(0, color="0.6", ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("RG step")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Lonardoni 2017 cultures, n = {len(CULTURES)}, mean +/- sem", fontsize=10)
    fig.tight_layout()
    fig.savefig("Figures/Cultures_excess.png", dpi=170)
    print("saved Figures/Cultures_excess.png")


if __name__ == "__main__":
    main()
