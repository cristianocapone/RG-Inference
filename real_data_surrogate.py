"""
Fig. 4B analogue on public HD-MEA culture data, with a surrogate control.

Same data and RG as real_data_rg.py (Lonardoni et al. 2017, dt = 10 ms, N most active
channels).  For each culture we run the three pairing rules on the real data and on a
surrogate in which every channel is circularly shifted in time by an independent
random offset (>= 1 s): autocorrelations are kept, equal-time and lagged
cross-correlations are destroyed, so the inferred J of the surrogate is pure noise.

At each RG level k we report
  * symmetry of the inferred J (real vs surrogate);
  * symmetry restricted to significant couplings: entries with |J| above the 99th
    percentile of |J_surrogate| at the same level (same rule), everything else set
    to 0;
  * fraction of significant couplings;
  * variance-scaling exponent (real vs surrogate).

Run:  env -i /opt/homebrew/bin/python3 real_data_surrogate.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import (infer_J, greedy_pairs, synaptic_similarity,
                              variance_scaling, local_exponent)
from spiking_ei_rg import symmetry
from real_data_rg import load_culture, DT


def shift_surrogate(B, rng, min_shift_s=1.0, dt=DT):
    T, N = B.shape
    lo = int(min_shift_s / dt)
    out = np.empty_like(B)
    for i in range(N):
        out[:, i] = np.roll(B[:, i], rng.integers(lo, T - lo))
    return out


def rg_track(B, rule, n_levels, seed=0):
    """Returns per-level inferred J list and variance exponents."""
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    Js, levels = [], [S]
    for k in range(n_levels):
        J = infer_J(S, k)
        Js.append(J)
        if k == n_levels - 1:
            break
        if rule == "inference":
            pairs = greedy_pairs(synaptic_similarity(J))
        elif rule == "correlation":
            C = np.nan_to_num(np.corrcoef(S.T))
            np.fill_diagonal(C, -np.inf)
            pairs = greedy_pairs(C)
        else:
            pairs = rng.permutation(S.shape[1] // 2 * 2).reshape(-1, 2)
        S = 0.5 * (S[:, pairs[:, 0]] + S[:, pairs[:, 1]])
        levels.append(S)
    K, M2 = variance_scaling(levels)
    return Js, local_exponent(K, M2)[1]


def significant_symmetry(J, Jsur, q=99):
    off = ~np.eye(len(J), dtype=bool)
    thr = np.percentile(np.abs(Jsur[off]), q)
    mask = np.abs(J) > thr
    Jsig = np.where(mask, J, 0.0)
    frac = mask[off].mean()
    return (symmetry(Jsig) if frac > 0 else np.nan), frac


def main(cultures=("d3", "d5"), N=512, n_levels=8):
    rules = ["inference", "correlation", "random"]
    colors = {"inference": "#d62728", "correlation": "#8c8c00", "random": "#4a7fd6"}
    fig, axs = plt.subplots(4, len(cultures), figsize=(4.6 * len(cultures), 11), squeeze=False)
    steps = np.arange(1, n_levels + 1)
    for c, name in enumerate(cultures):
        B, rates, _ = load_culture(name, N)
        Bs = shift_surrogate(B, np.random.default_rng(7))
        print(f"=== {name}: T={B.shape[0]} bins, N={N}, mean rate {rates.mean():.2f} Hz")
        for rule in rules:
            Js, ex = rg_track(B, rule, n_levels)
            Jsur, exs = rg_track(Bs, rule, n_levels)
            sym = [symmetry(J) for J in Js]
            syms = [symmetry(J) for J in Jsur]
            sig = [significant_symmetry(J, Jsu) for J, Jsu in zip(Js, Jsur)]
            sym_sig, frac = [s[0] for s in sig], [s[1] for s in sig]
            print(f"  {rule:12s} symmetry real {np.round(sym, 2)}")
            print(f"  {'':12s} symmetry surr {np.round(syms, 2)}")
            print(f"  {'':12s} sym (signif.) {np.round(sym_sig, 2)}")
            print(f"  {'':12s} frac signif.  {np.round(frac, 3)}")
            print(f"  {'':12s} exps real {np.round(ex, 2)}  surr {np.round(exs, 2)}")
            axs[0, c].plot(steps, sym, "o-", color=colors[rule], ms=3, label=rule)
            axs[0, c].plot(steps, syms, "--", color=colors[rule], lw=1)
            axs[1, c].plot(steps, sym_sig, "o-", color=colors[rule], ms=3)
            axs[2, c].plot(steps, frac, "o-", color=colors[rule], ms=3)
            axs[3, c].plot(steps[1:], ex, "o-", color=colors[rule], ms=3)
            axs[3, c].plot(steps[1:], exs, "--", color=colors[rule], lw=1)
        axs[0, c].set_title(f"culture {name}, N={N}  (solid: data, dashed: shifted surrogate)", fontsize=9)
        axs[0, c].set_ylabel("symmetry of inferred J")
        axs[0, c].axhline(0, color="0.6", ls=":")
        axs[0, c].legend(fontsize=8, frameon=False)
        axs[1, c].set_ylabel("symmetry, significant couplings only")
        axs[1, c].axhline(0, color="0.6", ls=":")
        axs[2, c].set_ylabel("fraction of significant couplings")
        axs[2, c].axhline(0.01, color="0.6", ls=":")
        axs[2, c].set_yscale("log")
        axs[3, c].set_ylabel("variance scaling exponent")
        for ax in axs[:, c]:
            ax.set_xlabel("RG step")
    fig.tight_layout()
    fig.savefig("Figures/RealData_surrogate.png", dpi=170)
    print("saved Figures/RealData_surrogate.png")


if __name__ == "__main__":
    main()
