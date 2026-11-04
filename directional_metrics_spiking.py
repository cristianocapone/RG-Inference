"""
Metrics sensitive to the directional part of the dynamics, along the chain sweep of the
spiking E/I networks (chain 0, 0.05, 0.1, 0.2; 512-unit subsets as in closure_test.py).
The one-step closure is dominated by the slow, symmetric co-fluctuations of the clusters
(symmetric lag-1 power doubles along the chain while the antisymmetric one grows by 40%),
so it cannot track the directional information the chain adds.  Here, at 32 blocks
(level 4), for the rules inference / inference_out / oneshot / lagged_out / correlation:
  specific predictability   held-out R^2 of y(t+1) beyond an autoregressive + global-mode
                            predictor (rg_metrics.heldout_r2), averaged over blocks
  multi-step closure        R^2[y(t+k) | y(t)] / R^2[y(t+k) | S(t)], k = 1, 3, 10, 30 bins
Groupings formed on the training part only.  Results: results/directional_metrics_spiking.npz
Run from the folder holding sim_cache/.
"""

import sys
import time
import numpy as np

sys.argv = [sys.argv[0], "40", "0.5", "sweep"]
import spiking_ei_rg as spk
from directionality_sweep_spiking import membership, cache_name, NET_KW, SIM, CHAINS
from closure_test import ridge_r2, FRAC
from rg_metrics import heldout_r2

RULES = ["inference", "inference_out", "oneshot", "lagged_out", "correlation"]
LAGS = [1, 3, 10, 30]
LEVEL = 4


def closure_k(S0, M, k, frac=FRAC):
    T = len(S0)
    Tt = int(frac * T)
    Y = S0 @ (M / M.sum(0))
    dX, dY = S0 - S0[:Tt].mean(0), Y - Y[:Tt].mean(0)
    r2c = ridge_r2(dY[:Tt - k], dY[k:Tt], dY[Tt:-k], dY[Tt + k:])
    r2m = ridge_r2(dX[:Tt - k], dY[k:Tt], dX[Tt:-k], dY[Tt + k:])
    return r2c / r2m, r2c


def main():
    store = {}
    for c in CHAINS:
        ts, ids, W, clu = spk.get_spikes(cache_name(c), c, 600, SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
        B = spk.bin_spikes(ts, ids, 10.0)
        rng = np.random.default_rng(0)
        sub = np.sort(np.r_[rng.choice(spk.NE, 400, replace=False), spk.NE + rng.choice(spk.NI, 112, replace=False)])
        S0 = B[:, sub].astype(np.float32)
        for rule in RULES:
            t0 = time.time()
            M = membership(S0, rule, LEVEL)
            Y = S0 @ (M / M.sum(0))
            r2, r2a, r2ag = heldout_r2(Y)
            store[f"{c}/{rule}/specific"] = r2 - r2ag
            store[f"{c}/{rule}/r2"] = r2
            for k in LAGS:
                store[f"{c}/{rule}/closure{k}"], store[f"{c}/{rule}/r2coarse{k}"] = closure_k(S0, M, k)
            print(f"chain {c:4}: {rule:14s} specific {r2 - r2ag:+.4f} (r2 {r2:.3f})  closure k=1,3,10,30: "
                  + " ".join(f"{store[f'{c}/{rule}/closure{k}']:.3f}" for k in LAGS) + f"  [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/directional_metrics_spiking.npz", **store)


if __name__ == "__main__":
    main()
