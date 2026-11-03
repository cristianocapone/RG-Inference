"""
In-vivo test of the partial-observation prediction: does the advantage of
inference-based over correlation-based coarse-graining grow with the number of
recorded units?

Stringer et al. 2019, eight-probe Neuropixels, 3 mice (1462-2688 sorted units,
22-41 min of spontaneous activity).  For each mouse, random subsets of n units among
those with rate > 0.5 Hz, n = 64 ... up to min(#units, T/100), 3 subsets per n;
10-ms bins.  Per subset: directedness (1 - symmetry of D(1)), closure advantage of the
inference-based rule (in+out and output-only) over the correlation-based rule,
averaged over RG steps 2..L-1 and over steps 2-3 only (same depth for every n).
Results: results/invivo_density.npz.  Run from the folder holding data/ and results/:
    python invivo_density.py
"""

import os
import time
import numpy as np

from invivo_areas import load_units, MICE
from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

SIZES = [64, 128, 256, 512, 1024, 2048]
SEEDS = range(3)
RULES = ["inference", "inference_out", "correlation"]


def evaluate(B):
    S = B.astype(np.float32)
    _, D = lagged_cov(S)
    L = min(9, int(np.log2(B.shape[1])) - 1)
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    full = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    d23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:2]]
    return np.r_[1 - symmetry(D), full, d23]


def main():
    os.makedirs("results", exist_ok=True)
    out = "results/invivo_density.npz"
    store = dict(np.load(out)) if os.path.exists(out) else {}
    for mouse, tstart in MICE.items():
        B_all, rates, area, probe = load_units(mouse, tstart)
        ok = np.flatnonzero(rates > 0.5)
        nmax = min(len(ok), B_all.shape[0] // 100)
        print(f"=== {mouse}: {len(ok)} units > 0.5 Hz, T = {B_all.shape[0]}, max n = {nmax}", flush=True)
        for n in [n for n in SIZES if n <= nmax]:
            key = f"{mouse}/{n}"
            if key in store:
                continue
            t0 = time.time()
            vals = np.array([evaluate(B_all[:, np.sort(np.random.default_rng(1000 * n + s).choice(ok, n, replace=False))])
                             for s in SEEDS])
            store[key] = vals
            m = vals.mean(0)
            print(f"{mouse:8s} n={n:5d}: directedness {m[0]:.3f}  adv (2..L-1) inference {m[1]:+.4f} output {m[2]:+.4f}"
                  f"  | steps 2-3: inference {m[3]:+.4f} output {m[4]:+.4f}  [{time.time() - t0:.0f} s]", flush=True)
            np.savez(out, **store)


if __name__ == "__main__":
    main()
