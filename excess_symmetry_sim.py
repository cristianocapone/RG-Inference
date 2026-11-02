"""
Same metrics as excess_symmetry_cultures.py on simulations with ground truth:
  * spiking E/I networks (clustered, chain) with axonal delays, matched 10 ms bins
    (cached spikes from spiking_ei_rg.py, run tag d5j0.5_s0.15_Nref3000_wclu1.9);
  * the Hopfield sequence network of Fig. 3 (N = 512, T = 3e5), if the activity file
    is given as argv[1].
Figure: Figures/Sim_excess.png

Run:  env -i /opt/homebrew/bin/python3 excess_symmetry_sim.py [S0seq512.npy]
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spiking_ei_rg as spk
from rg_metrics import data_vs_surrogate, RULES, COLORS

TAG = "d5j0.5_s0.15_Nref3000_wclu1.9"


def main():
    sets = []
    for name in ["clustered", "chain"]:
        d = np.load(f"sim_cache/{name}_{TAG}_spikes_T600.npz")
        B = spk.bin_spikes(d["ts"], d["ids"], 10.0)
        sets.append((f"spiking {name} (10 ms)", B, 10))
    if len(sys.argv) > 1:
        S0 = np.load(sys.argv[1]).astype(np.uint8)
        sets.append(("Hopfield sequences N=512", S0, 8))

    fig, axs = plt.subplots(len(sets), 4, figsize=(15, 3.3 * len(sets)), squeeze=False)
    for row, (title, B, n_levels) in enumerate(sets):
        t0 = time.time()
        res = data_vs_surrogate(B, n_levels)
        print(f"=== {title}: T={B.shape[0]}, N={B.shape[1]}  [{time.time() - t0:.0f} s]")
        steps = np.arange(1, n_levels + 1)
        for rule in RULES:
            r, s = res[rule]["real"], res[rule]["surr"]
            print(f"  {rule:12s} sym_J real {np.round(r['sym_J'], 2)}  surr {np.round(s['sym_J'], 2)}")
            print(f"  {'':12s} sym_D real {np.round(r['sym_D'], 2)}  surr {np.round(s['sym_D'], 2)}")
            print(f"  {'':12s} r2-r2auto {np.round(r['r2'] - r['r2_auto'], 3)}", flush=True)
            n = len(r["sym_J"])
            axs[row, 0].plot(steps[:n], r["sym_J"], "o-", color=COLORS[rule], ms=3, label=rule)
            axs[row, 0].plot(steps[:n], s["sym_J"], "--", color=COLORS[rule], lw=1)
            axs[row, 1].plot(steps[:n], r["sym_J"] - s["sym_J"], "o-", color=COLORS[rule], ms=3)
            axs[row, 2].plot(steps[:n], r["sym_D"] - s["sym_D"], "o-", color=COLORS[rule], ms=3)
            axs[row, 3].plot(steps[:n], r["r2"] - r["r2_auto"], "o-", color=COLORS[rule], ms=3)
        for ax, t in zip(axs[row], ["symmetry of inferred J (dashed: surrogate)", "excess symmetry of J",
                                    "excess symmetry of D(1)", "cross-unit predictability R2 - R2_auto"]):
            ax.axhline(0, color="0.6", ls=":")
            ax.set_title(f"{title}: {t}", fontsize=8)
            ax.set_xlabel("RG step")
        axs[row, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("Figures/Sim_excess.png", dpi=170)
    print("saved Figures/Sim_excess.png")


if __name__ == "__main__":
    main()
