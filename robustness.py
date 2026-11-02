"""
Robustness of the data results (excess_symmetry_cultures.py, invivo_stringer.py) to
  * unit selection: 512 most active units vs 512 random units (rate > 0.2 Hz);
  * bin size: 5, 10, 20 ms.
For every configuration and every sample (9 cultures, 3 mice) the three rules are run
(data + shifted surrogate) and we keep, per level, the specific predictability
r2 - r2_ag and the excess symmetry of J.  Printed: inference - correlation specific
predictability (mean over samples, fraction > 0, pooled over steps 2-7) and the excess
symmetry of the three rules at step 6.  Results saved to results/robustness.npz.

Run:  env -i /opt/homebrew/bin/python3 robustness.py <dir with spks*.mat>
"""

import os
import sys
import time
import numpy as np

from real_data_rg import load_culture
from invivo_stringer import bin_mouse, MICE
from rg_metrics import data_vs_surrogate, pad, RULES

CULTURES = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
L = 8
N_KEEP = 512


def select(B_all, rates, mode, rng, min_rate=0.2):
    T = B_all.shape[0]
    n = min(N_KEEP, B_all.shape[1])
    if mode == "top":
        idx = np.arange(n)
    else:
        ok = np.flatnonzero(rates > min_rate)
        idx = np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))
    idx = idx[:len(idx) // 2 * 2]
    return B_all[:, idx]


def run_sample(B, tag, store):
    n_levels = min(L, int(np.log2(B.shape[1])) - 1)
    res = data_vs_surrogate(B, n_levels)
    for rule in RULES:
        r, s = res[rule]["real"], res[rule]["surr"]
        store[f"{tag}/{rule}/spec"] = pad(r["r2"] - r["r2_ag"], L)
        store[f"{tag}/{rule}/xsym"] = pad(r["sym_J"] - s["sym_J"], L)
        store[f"{tag}/{rule}/sym"] = pad(r["sym_J"], L)


def main(stringer_root):
    os.makedirs("results", exist_ok=True)
    store = {}
    configs = [(mode, dt) for dt in (0.005, 0.010, 0.020) for mode in ("top", "random")]
    for mode, dt in configs:
        t0 = time.time()
        for c in CULTURES:
            B_all, rates, _ = load_culture(c, 10000, dt=dt)
            n_max = min(N_KEEP, B_all.shape[0] // 100)
            B = select(B_all[:, :max(n_max, 2)] if mode == "top" else B_all, rates, mode, np.random.default_rng(5))
            if mode == "random":
                B = B[:, :n_max // 2 * 2]
            run_sample(B, f"cult/{mode}/{dt * 1000:g}/{c}", store)
        print(f"cultures {mode} {dt * 1000:g} ms done [{time.time() - t0:.0f} s]", flush=True)
        t0 = time.time()
        for mouse, tstart in MICE.items():
            path = next(os.path.join(dp, f) for dp, _, fs in os.walk(stringer_root) for f in fs
                        if f.startswith(f"spks{mouse}") and f.endswith(".mat"))
            B_all, rates, _, _ = bin_mouse(path, tstart, dt=dt, n_keep=100000)
            B = select(B_all, rates, mode, np.random.default_rng(5))
            run_sample(B, f"mouse/{mode}/{dt * 1000:g}/{mouse}", store)
        print(f"mice {mode} {dt * 1000:g} ms done [{time.time() - t0:.0f} s]", flush=True)
        np.savez("results/robustness.npz", **store)
    report(store)


def report(store):
    for group, samples in [("cult", CULTURES), ("mouse", list(MICE))]:
        print(f"\n=== {group} (n = {len(samples)})")
        print(f"{'sel':7s} {'dt':>5s} | spec. pred. inf-corr: mean steps2-7, frac>0 | inf-rand | excess sym @step6: inf / corr / rand")
        for dt in (5, 10, 20):
            for mode in ("top", "random"):
                g = lambda rule, m: np.stack([store[f"{group}/{mode}/{dt}/{s}/{rule}/{m}"] for s in samples])
                d_ic = (g("inference", "spec") - g("correlation", "spec"))[:, 1:7]
                d_ir = (g("inference", "spec") - g("random", "spec"))[:, 1:7]
                xs = [np.nanmean(g(r, "xsym")[:, 5]) for r in RULES]
                print(f"{mode:7s} {dt:5d} | {np.nanmean(d_ic):+.3f}  {np.nanmean(d_ic > 0):.2f}            "
                      f"| {np.nanmean(d_ir):+.3f}   | {xs[0]:+.2f} / {xs[1]:+.2f} / {xs[2]:+.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(dict(np.load("results/robustness.npz")))
    else:
        main(sys.argv[1])
