"""
Partial observation in the data: closure advantage of inference-based over
correlation-based coarse-graining on random subsets of electrodes of the cultures
(Lonardoni 2017).  If the advantage requires observing a sufficient fraction of the
circuit, it must decrease as fewer electrodes are kept.
For each culture, n = 512, 256, 128, 64, 32 random electrodes among the usable ones (rate
>= 0.2 Hz, n <= T/100; sizes above the available number are skipped), 5 subsets each; advantage averaged over
RG steps 2-3 (same depth for every n) and over steps 2..L-1.
Results: results/partial_observation_cultures.npz.  Run from the folder holding data/.
    python partial_observation_cultures.py
"""

import time
import numpy as np

from real_data_rg import load_culture
from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
SIZES = [512, 256, 128, 64, 32]
SEEDS = range(5)
RULES = ["inference", "inference_out", "correlation"]


def evaluate(B):
    _, D = lagged_cov(B.astype(np.float32))
    L = int(np.log2(B.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    full = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    d23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:2]]
    return np.r_[1 - symmetry(D), full, d23]


def main():
    store = {}
    for c in CULT:
        B_all, rates, _ = load_culture(c, 10000)
        ok = np.flatnonzero(rates >= 0.2)
        nmax = min(len(ok), B_all.shape[0] // 100)
        sizes = [n for n in SIZES if n <= nmax]
        if len(sizes) < 3:
            print(f"{c}: only {nmax} usable electrodes, skipped", flush=True)
            continue
        for n in sizes:
            t0 = time.time()
            vals = np.array([evaluate(B_all[:, np.sort(np.random.default_rng(1000 * n + s).choice(ok, n, replace=False))])
                             for s in SEEDS])
            store[f"{c}/{n}"] = vals
            m = vals.mean(0)
            print(f"{c:4s} n={n:4d}: directedness {m[0]:.3f}  adv (steps 2..L-1) inference {m[1]:+.3f} output {m[2]:+.3f}"
                  f"  | steps 2-3: inference {m[3]:+.3f} output {m[4]:+.3f}  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/partial_observation_cultures.npz", **store)


if __name__ == "__main__":
    main()
