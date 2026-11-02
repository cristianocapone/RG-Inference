"""
Is in-vivo spontaneous activity weakly directional only because the 512 units are
scattered over eight probes and many areas?  For every mouse and every probe (local
populations) we compute, at 5 and 10 ms bins:
  directedness   1 - symmetry index of the lagged covariance D(1) at level 0
  advantage      closure advantage of the inference-based (in+out, output-only) over
                 the correlation-based reduction, mean over RG steps 2..L-1
on the N = min(256, T/100) most active units of the probe (probes with >= 64 such
units).  Also the pooled 512-unit population of each mouse for reference.
Results: results/invivo_local.npz.  Run from the folder holding data/ and results/:

    python invivo_local.py
"""

import os
import time
import numpy as np

from invivo_areas import load_units, MICE, AREAS
from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

MIN_UNITS, MAX_UNITS = 64, 256
RULES = ["inference", "inference_out", "correlation"]


def evaluate(B):
    _, D = lagged_cov(B.astype(np.float32))
    direct = 1 - symmetry(D)
    n_levels = int(np.log2(B.shape[1])) - 2
    cl = {r: rg_closure(B, r, n_levels)["closure"] for r in RULES}
    adv = {r: float(np.nanmean((cl[r] - cl["correlation"])[1:n_levels - 1])) for r in RULES[:2]}
    return direct, adv, n_levels


def main():
    os.makedirs("results", exist_ok=True)
    store = {}
    for dt in (0.010, 0.005):
        for mouse, tstart in MICE.items():
            t0 = time.time()
            B_all, rates, area, probe = load_units(mouse, tstart, dt=dt)
            T = B_all.shape[0]
            for p in np.unique(probe):
                idx = np.flatnonzero((probe == p) & (rates > 0.5))
                idx = idx[np.argsort(-rates[idx])][:min(MAX_UNITS, T // 100)]
                if len(idx) < MIN_UNITS:
                    continue
                idx = np.sort(idx[:len(idx) // 2 * 2])
                areas = ",".join(sorted({AREAS[a] for a in area[idx] if a >= 0}))
                direct, adv, L = evaluate(B_all[:, idx])
                key = f"{dt * 1000:g}ms/{mouse}/probe{p}"
                store[key] = np.array([direct, adv["inference"], adv["inference_out"], len(idx)])
                print(f"{key:24s} N={len(idx):3d} [{areas}] directedness {direct:.3f}  "
                      f"adv inference {adv['inference']:+.3f}  output {adv['inference_out']:+.3f}", flush=True)
            idx = np.argsort(-rates)[:512]
            direct, adv, L = evaluate(B_all[:, np.sort(idx)])
            key = f"{dt * 1000:g}ms/{mouse}/pooled512"
            store[key] = np.array([direct, adv["inference"], adv["inference_out"], 512])
            print(f"{key:24s} N=512 directedness {direct:.3f}  adv inference {adv['inference']:+.3f}  "
                  f"output {adv['inference_out']:+.3f}  [{time.time() - t0:.0f} s]", flush=True)
            np.savez("results/invivo_local.npz", **store)


if __name__ == "__main__":
    main()
