"""
Fig. 4B analogue on public HD-MEA culture data (the original Nieus 2015 recording is
not public): Lonardoni et al. 2017, rat hippocampal cultures on the same 3Brain 4096
chip, spontaneous activity (data/lonardoni2017_spont, see SOURCE.txt).

Channels (thresholded MUA) are binned at dt = 10 ms and binarised; the N most active
channels are kept so that T/N >~ 100. RG is run with the inference, correlation and
random pairing rules, tracking the symmetry of the inferred J and the variance-scaling
exponent at each step.

Run:  env -i /opt/homebrew/bin/python3 real_data_rg.py
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import (infer_J, greedy_pairs, synaptic_similarity,
                              variance_scaling, local_exponent)
from spiking_ei_rg import symmetry

DATA = "data/lonardoni2017_spont"
DT = 0.010


def load_culture(name, n_keep, dt=DT):
    d = sio.loadmat(f"{DATA}/{name}_spont.mat")
    fs = float(d["SamplingFrequency"].squeeze())
    n_frames = int(d["StopFrame"].squeeze() - d["StartFrame"].squeeze())
    T = int(n_frames / fs / dt)
    chans = sorted(k for k in d if k.startswith("Ch"))
    B = np.zeros((T, len(chans)), np.uint8)
    for c, k in enumerate(chans):
        frames = d[k].tocoo().row
        b = (frames / fs / dt).astype(int)
        B[b[b < T], c] = 1
    rates = B.mean(0) / dt
    keep = np.argsort(-rates)[:n_keep]
    return B[:, keep], rates[keep], [chans[i] for i in keep]


def rg_symmetry(B, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    sym, levels = [], [S]
    for k in range(n_levels):
        J = infer_J(S, k)
        sym.append(symmetry(J))
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
    return np.array(sym), local_exponent(K, M2)[1]


def main(cultures=("d3", "d5", "d6"), N=512, n_levels=8, n_random=5):
    rules = ["inference", "correlation", "random"]
    colors = {"inference": "#d62728", "correlation": "#8c8c00", "random": "#4a7fd6"}
    fig, axs = plt.subplots(2, len(cultures), figsize=(4.2 * len(cultures), 6.5), squeeze=False)
    steps = np.arange(1, n_levels + 1)
    for c, name in enumerate(cultures):
        B, rates, _ = load_culture(name, N)
        print(f"=== {name}: T={B.shape[0]} bins ({B.shape[0] * DT / 60:.0f} min), N={N}, "
              f"rate median {np.median(rates):.2f} Hz (min {rates.min():.2f})")
        for rule in rules:
            if rule == "random":
                runs = [rg_symmetry(B, rule, n_levels, seed=s) for s in range(n_random)]
                sym = np.array([r[0] for r in runs])
                ex = np.array([r[1] for r in runs])
                axs[0, c].fill_between(steps, sym.mean(0) - sym.std(0), sym.mean(0) + sym.std(0),
                                       color=colors[rule], alpha=0.25, lw=0)
                sym, ex = sym.mean(0), ex.mean(0)
            else:
                sym, ex = rg_symmetry(B, rule, n_levels)
            print(f"  {rule:12s} symmetry {np.round(sym, 2)}")
            print(f"  {'':12s} variance exps {np.round(ex, 2)}")
            axs[0, c].plot(steps, sym, "o-", color=colors[rule], ms=3, label=rule)
            axs[1, c].plot(steps[1:], ex, "o-", color=colors[rule], ms=3, label=rule)
        axs[0, c].axhline(0, color="0.6", ls=":")
        axs[0, c].set_title(f"culture {name} (Lonardoni 2017), N={N}", fontsize=9)
        axs[0, c].set_ylabel("symmetry of inferred J")
        axs[1, c].set_ylabel("variance scaling exponent")
        for ax in axs[:, c]:
            ax.set_xlabel("RG step")
        axs[0, c].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("Figures/RealData_RG.png", dpi=170)
    print("saved Figures/RealData_RG.png")


if __name__ == "__main__":
    main()
