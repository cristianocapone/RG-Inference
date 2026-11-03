"""
In-vivo test of the joint dependence of the inference advantage on directedness and on
the number of observed units.  In the cultures the advantage grew with the number of
observed units only when the dynamics was directional; spontaneous activity in vivo is
weakly directional, so the test must be done where it is not: during visual stimulation.

Allen Brain Observatory, Visual Coding - Neuropixels (DANDI 000021), 32 sessions,
streamed.  Per session: all units with quality "good", presence ratio > 0.9 and rate
> 1 Hz; conditions spontaneous, drifting_gratings, natural_movie_three, truncated to
the same duration; for n = 64, 128, 256, 512 (n <= #units and n <= T/100), SEEDS random
subsets of units: directedness and closure advantage of inference (in+out, output) over
correlation, averaged over steps 2..L-1 and over steps 2-3.
Per-session cache results/allen_density/<session>.npz; summary printed at the end.
Run from the folder holding results/:  python allen_density.py [n_sessions]
"""

import glob
import os
import sys
import time
import numpy as np
import h5py
import remfile

from allen_rest_vs_task import session_urls, condition_bins, read_strings, DT
from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

CONDS = ["spontaneous", "drifting_gratings", "natural_movie_three"]
SIZES = [64, 128, 256, 512]
SEEDS = range(2)
RULES = ["inference", "inference_out", "correlation"]


def load_all(url):
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    q = read_strings(u["quality"])
    ok = np.flatnonzero((q == "good") & (u["presence_ratio"][()] > 0.9) & (u["firing_rate"][()] > 1.0))
    idx = u["spike_times_index"][()]
    starts = np.r_[0, idx[:-1]]
    spikes = [u["spike_times"][starts[i]:idx[i]] for i in ok]
    intervals = {c: np.c_[f[f"intervals/{c}_presentations"]["start_time"][()], f[f"intervals/{c}_presentations"]["stop_time"][()]]
                 for c in CONDS if f"intervals/{c}_presentations" in f}
    inv = np.c_[f["intervals/invalid_times/start_time"][()], f["intervals/invalid_times/stop_time"][()]] \
        if "intervals/invalid_times" in f else np.zeros((0, 2))
    f.close()
    return spikes, intervals, inv


def evaluate(B):
    _, D = lagged_cov(B.astype(np.float32))
    L = int(np.log2(B.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    full = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    d23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:2]]
    return np.r_[1 - symmetry(D), full, d23]


def run_session(url):
    spikes, intervals, inv = load_all(url)
    t_max = max(s[-1] for s in spikes if len(s))
    bins = {c: condition_bins(iv, inv, t_max) for c, iv in intervals.items()}
    if any(c not in bins for c in CONDS):
        return None
    T_eq = min(len(b) for b in bins.values())
    nb = int(t_max / DT) + 1
    full = np.zeros((nb, len(spikes)), np.uint8)
    for j, s in enumerate(spikes):
        k = (s / DT).astype(int)
        full[k[k < nb], j] = 1
    out = {"n_units": len(spikes), "T_eq_min": T_eq * DT / 60}
    for c in CONDS:
        B = full[bins[c][:T_eq]]
        active = np.flatnonzero(B.std(0) > 0)
        for n in [n for n in SIZES if n <= min(len(active), T_eq // 100)]:
            vals = [evaluate(B[:, np.sort(np.random.default_rng(100 * n + s).choice(active, n, replace=False))])
                    for s in SEEDS]
            out[f"{c}/{n}"] = np.mean(vals, 0)
    return out


def main(n_max=None):
    os.makedirs("results/allen_density", exist_ok=True)
    done = 0
    for path, url in session_urls():
        ses = os.path.basename(path).replace(".nwb", "")
        f = f"results/allen_density/{ses}.npz"
        if not os.path.exists(f):
            t0 = time.time()
            try:
                out = run_session(url)
            except Exception as e:
                print(f"{ses}: failed {e!r}", flush=True)
                continue
            np.savez(f, **({"skipped": True} if out is None else {k: np.asarray(v) for k, v in out.items()}))
            if out is not None:
                line = "  ".join(f"{k} {v[0]:.2f}/{v[2]:+.3f}" for k, v in out.items() if "/" in k)
                print(f"{ses}: {out['n_units']} units, {out['T_eq_min']:.1f} min | dir/adv(output): {line}  [{time.time() - t0:.0f} s]",
                      flush=True)
        done += 1
        if n_max and done >= n_max:
            break
    summarize()


def summarize():
    from scipy.stats import wilcoxon, spearmanr
    rows = [dict(np.load(f)) for f in sorted(glob.glob("results/allen_density/*.npz"))]
    rows = [r for r in rows if "skipped" not in r]
    print(f"\n{len(rows)} sessions")
    for c in CONDS:
        for n in SIZES:
            v = np.array([r[f"{c}/{n}"] for r in rows if f"{c}/{n}" in r])
            if len(v):
                print(f"  {c:20s} n={n:4d} (sessions {len(v):2d}): directedness {v[:, 0].mean():.3f}  "
                      f"adv output {v[:, 2].mean():+.4f}  (steps 2-3 {v[:, 4].mean():+.4f})")
        # within-session change of the advantage from the smallest to the largest common size
        pairs = [(r[f"{c}/64"][2], r[f"{c}/256"][2]) for r in rows if f"{c}/64" in r and f"{c}/256" in r]
        if len(pairs) >= 6:
            d = np.array([b - a for a, b in pairs])
            print(f"  {c:20s} adv(256) - adv(64): {d.mean():+.4f}, {np.sum(d > 0)}/{len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.3g}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
