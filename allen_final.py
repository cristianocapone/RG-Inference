"""
Allen Brain Observatory (Visual Coding - Neuropixels, DANDI 000021, streamed), analysed
with the final pipeline in a single pass per session (about 1 GB of spike times streamed
per session):
  noise-corrected lag-1 directedness (directedness.py), groupings formed on the training
  part only, closure advantage over the correlation-based rule for the inference-based
  (full and output similarity) and the model-free lagged-correlation rules.
Two analyses per session:
  A. rest vs stimuli: the 256 most active good units (quality "good", presence ratio
     > 0.9, rate > 1 Hz), the same for all conditions, each condition truncated to the
     same duration (>= 8 min);
  B. number of observed units: all good units, random subsets of 64 ... 512 units
     (2 per size), for spontaneous activity, drifting gratings and natural movie three.
Per-session cache results/allen_final/<session>.npz; summary printed at the end.
Run from the folder holding results/:  python allen_final.py [n_sessions]
"""

import glob
import os
import sys
import time
import numpy as np
import h5py
import remfile

from allen_rest_vs_task import session_urls, condition_bins, read_strings, DT, MIN_MIN
from closure_test import rg_closure
from directedness import directedness
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

CONDS = ["spontaneous", "natural_movie_three", "natural_movie_one", "natural_scenes",
         "drifting_gratings", "static_gratings", "gabors"]
DENS_CONDS = ["spontaneous", "drifting_gratings", "natural_movie_three"]
N_TOP = 256
SIZES = [64, 128, 256, 512]
SEEDS = range(2)
RULES = ["inference", "inference_out", "lagged_out", "correlation"]
COLS = "directedness, adv inference, adv output, adv lagged, adv23 inference, adv23 output, adv23 lagged, raw directedness, rate (Hz)"


def load_all(url):
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    q = read_strings(u["quality"])
    fr = u["firing_rate"][()]
    ok = np.flatnonzero((q == "good") & (u["presence_ratio"][()] > 0.9) & (fr > 1.0))
    idx = u["spike_times_index"][()]
    starts = np.r_[0, idx[:-1]]
    spikes = [u["spike_times"][starts[i]:idx[i]] for i in ok]
    intervals = {c: np.c_[f[f"intervals/{c}_presentations"]["start_time"][()], f[f"intervals/{c}_presentations"]["stop_time"][()]]
                 for c in CONDS if f"intervals/{c}_presentations" in f}
    inv = np.c_[f["intervals/invalid_times/start_time"][()], f["intervals/invalid_times/stop_time"][()]] \
        if "intervals/invalid_times" in f else np.zeros((0, 2))
    f.close()
    return spikes, fr[ok], intervals, inv


def evaluate(B):
    B = B[:, B.std(0) > 0]
    B = B[:, :B.shape[1] // 2 * 2]
    L = int(np.log2(B.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    adv = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:3]]
    adv23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:3]]
    _, D = lagged_cov(B.astype(np.float32))
    return np.r_[directedness(B)[0], adv, adv23, 1 - symmetry(D), B.mean() / DT]


def run_session(url):
    spikes, fr, intervals, inv = load_all(url)
    t_max = max(s[-1] for s in spikes if len(s))
    bins = {c: condition_bins(iv, inv, t_max) for c, iv in intervals.items()}
    bins = {c: b for c, b in bins.items() if len(b) * DT / 60 >= MIN_MIN}
    if "spontaneous" not in bins or len(bins) < 3:
        return None
    T_eq = min(len(b) for b in bins.values())
    nb = int(t_max / DT) + 1
    full = np.zeros((nb, len(spikes)), np.uint8)
    for j, s in enumerate(spikes):
        k = (s / DT).astype(int)
        full[k[k < nb], j] = 1
    out = {"n_units": len(spikes), "T_eq_min": T_eq * DT / 60, "columns": COLS}
    top = np.sort(np.argsort(-fr)[:N_TOP])
    for c, b in bins.items():                                   # A. rest vs stimuli, 256 units
        out[f"top/{c}"] = evaluate(full[b[:T_eq]][:, top])
        v = out[f"top/{c}"]
        print(f"   {c:20s} dir {v[0]:.3f} (raw {v[7]:.2f})  adv inf {v[1]:+.3f} out {v[2]:+.3f} lag {v[3]:+.3f}", flush=True)
    T_d = min(len(bins[c]) for c in DENS_CONDS if c in bins)
    for c in [c for c in DENS_CONDS if c in bins]:              # B. number of observed units
        B = full[bins[c][:T_d]]
        active = np.flatnonzero(B.std(0) > 0)
        for n in [n for n in SIZES if n <= min(len(active), T_d // 100)]:
            vals = [evaluate(B[:, np.sort(np.random.default_rng(100 * n + s).choice(active, n, replace=False))]) for s in SEEDS]
            out[f"dens/{c}/{n}"] = np.mean(vals, 0)
    return out


def main(n_max=None):
    os.makedirs("results/allen_final", exist_ok=True)
    done = 0
    for path, url in session_urls():
        ses = os.path.basename(path).replace(".nwb", "")
        f = f"results/allen_final/{ses}.npz"
        if not os.path.exists(f):
            t0 = time.time()
            try:
                out = run_session(url)
            except Exception as e:
                print(f"{ses}: failed {e!r}", flush=True)
                continue
            np.savez(f, **({"skipped": True} if out is None else {k: np.asarray(v) for k, v in out.items()}))
            if out is not None:
                line = "  ".join(f"{k.split('/')[1][:8]}/{k.split('/')[2]} {v[0]:.2f}/{v[2]:+.3f}" for k, v in out.items() if k.startswith("dens/"))
                print(f"{ses}: {out['n_units']} units, {out['T_eq_min']:.1f} min | density dir/adv(out): {line}  [{time.time() - t0:.0f} s]", flush=True)
        done += 1
        if n_max and done >= n_max:
            break
    summarize()


def summarize():
    from scipy.stats import wilcoxon, spearmanr
    rows = [dict(np.load(f)) for f in sorted(glob.glob("results/allen_final/*.npz"))]
    rows = [r for r in rows if "skipped" not in r]
    print(f"\n=== A. rest vs stimuli, 256 units, {len(rows)} sessions (stimulus minus spontaneous)")
    for c in CONDS[1:]:
        rr = [r for r in rows if f"top/{c}" in r]
        if len(rr) < 6:
            continue
        d = np.array([r[f"top/{c}"] - r["top/spontaneous"] for r in rr])
        print(f"  {c:20s} (n={len(rr):2d}): directedness {d[:, 0].mean():+.3f} ({(d[:, 0] > 0).sum()}/{len(rr)}, p={wilcoxon(d[:, 0]).pvalue:.2g})"
              f"  adv output {d[:, 2].mean():+.4f} ({(d[:, 2] > 0).sum()}/{len(rr)}, p={wilcoxon(d[:, 2]).pvalue:.2g})"
              f"  adv in+out {d[:, 1].mean():+.4f} (p={wilcoxon(d[:, 1]).pvalue:.2g})  adv lagged {d[:, 3].mean():+.4f}")
    x = np.array([r[k][0] for r in rows for k in r if k.startswith("top/")])
    y = np.array([r[k][2] for r in rows for k in r if k.startswith("top/")])
    print(f"  pooled condition-session pairs (n={len(x)}): directedness {x.min():.3f}-{x.max():.3f}, Spearman(dir, adv output) rho={spearmanr(x, y)[0]:+.2f} p={spearmanr(x, y)[1]:.3g}")
    print(f"\n=== B. number of observed units")
    for c in DENS_CONDS:
        for n in SIZES:
            v = np.array([r[f"dens/{c}/{n}"] for r in rows if f"dens/{c}/{n}" in r])
            if len(v):
                print(f"  {c:20s} n={n:4d} (sessions {len(v):2d}): directedness {v[:, 0].mean():.3f}  adv output {v[:, 2].mean():+.4f}  in+out {v[:, 1].mean():+.4f}  lagged {v[:, 3].mean():+.4f}")
        pairs = [(r[f"dens/{c}/64"][2], r[f"dens/{c}/512"][2]) for r in rows if f"dens/{c}/64" in r and f"dens/{c}/512" in r]
        if len(pairs) >= 6:
            d = np.array([b - a for a, b in pairs])
            print(f"  {c:20s} adv output (512) - (64): {d.mean():+.4f}, {np.sum(d > 0)}/{len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.3g}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
