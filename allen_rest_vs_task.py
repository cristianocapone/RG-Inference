"""
Prediction: spontaneous ("resting") activity is close to time-reversible, stimulus-driven
activity is more directional (as found at the whole-brain scale, e.g. Deco et al.,
Lynn et al.), so the advantage of inference-based over correlation-based coarse-graining
should be larger during stimulation than at rest, in the same animal and on the same
units.

Data: Allen Brain Observatory, Visual Coding - Neuropixels (DANDI 000021, Brain
Observatory 1.1 stimulus set), streamed from DANDI (spike times of the selected units
and the stimulus tables only).  Per session:
  * units with quality == "good", presence_ratio > 0.9, firing rate > 1 Hz; the N = 256
    most active ones, the same units for all conditions;
  * conditions: spontaneous (grey screen), natural_movie_three, natural_movie_one,
    natural_scenes, drifting_gratings, static_gratings, gabors; for each, the time bins
    (10 ms) inside its presentations, invalid times excluded, truncated to the same
    duration T_eq for all conditions (the shortest available, >= 8 min);
  * per condition: directedness (1 - symmetry of D(1)) and closure along the RG for the
    inference, inference_out and correlation rules -> advantage = mean over steps
    2..L-1 of (rule - correlation).
Per-session results cached in results/allen/<session>.npz.

Run from the folder holding results/:
    python allen_rest_vs_task.py [n_sessions]
"""

import os
import sys
import time
import numpy as np
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient

from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

DT = 0.010
N_UNITS = 256
MIN_MIN = 8.0
CONDS = ["spontaneous", "natural_movie_three", "natural_movie_one", "natural_scenes",
         "drifting_gratings", "static_gratings", "gabors"]
RULES = ["inference", "inference_out", "correlation"]


def session_urls(dandiset="000021"):
    with DandiAPIClient() as c:
        for a in c.get_dandiset(dandiset).get_assets():
            if "probe" in a.path or not a.path.endswith(".nwb"):
                continue
            yield a.path, a.get_content_url(follow_redirects=1, strip_query=True)


def read_strings(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]])


def load(url):
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    q = read_strings(u["quality"])
    pr = u["presence_ratio"][()]
    fr = u["firing_rate"][()]
    ok = np.flatnonzero((q == "good") & (pr > 0.9) & (fr > 1.0))
    sel = ok[np.argsort(-fr[ok])][:N_UNITS]
    sel = np.sort(sel)
    idx = u["spike_times_index"][()]
    starts = np.r_[0, idx[:-1]]
    spikes = [u["spike_times"][starts[i]:idx[i]] for i in sel]
    intervals = {}
    for c in CONDS:
        key = f"intervals/{c}_presentations"
        if key in f:
            intervals[c] = np.c_[f[key]["start_time"][()], f[key]["stop_time"][()]]
    inv = np.c_[f["intervals/invalid_times/start_time"][()], f["intervals/invalid_times/stop_time"][()]] \
        if "intervals/invalid_times" in f else np.zeros((0, 2))
    f.close()
    return spikes, intervals, inv


def condition_bins(iv, inv, t_max, dt=DT):
    """Indices of the global 10-ms bins fully inside presentations of a condition."""
    nb = int(t_max / dt) + 1
    m = np.zeros(nb, bool)
    for s, e in iv:
        m[int(np.ceil(s / dt)):int(np.floor(e / dt))] = True
    for s, e in inv:
        m[max(0, int(s / dt)):int(e / dt) + 1] = False
    return np.flatnonzero(m)


def analyse(B):
    _, D = lagged_cov(B.astype(np.float32))
    direct = 1 - symmetry(D)
    L = int(np.log2(B.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    adv = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    return np.r_[direct, adv, B.mean() / DT]


def run_session(path, url):
    t0 = time.time()
    spikes, intervals, inv = load(url)
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
    out = {"T_eq_min": T_eq * DT / 60, "N": len(spikes)}
    for c, b in bins.items():
        B = full[b[:T_eq]]
        B = B[:, B.std(0) > 0]
        B = B[:, :B.shape[1] // 2 * 2]
        out[c] = analyse(B)
        print(f"   {c:20s} directedness {out[c][0]:.3f}  adv inference {out[c][1]:+.3f}  output {out[c][2]:+.3f}  "
              f"rate {out[c][3]:.1f} Hz", flush=True)
    print(f"  {path}: N={len(spikes)}, T_eq={T_eq * DT / 60:.1f} min  [{time.time() - t0:.0f} s]", flush=True)
    return out


def main(n_max=None):
    os.makedirs("results/allen", exist_ok=True)
    done = 0
    for path, url in session_urls():
        ses = os.path.basename(path).replace(".nwb", "")
        out_f = f"results/allen/{ses}.npz"
        if os.path.exists(out_f):
            done += 1
        else:
            print(f"== {ses}", flush=True)
            try:
                out = run_session(path, url)
            except Exception as e:
                print(f"  failed: {e!r}", flush=True)
                continue
            if out is None:
                print("  skipped", flush=True)
                np.savez(out_f, skipped=True)
                continue
            np.savez(out_f, **{k: np.asarray(v) for k, v in out.items()})
            done += 1
        if n_max and done >= n_max:
            break
    summarize()


def summarize():
    import glob
    from scipy.stats import wilcoxon
    rows = []
    for f in sorted(glob.glob("results/allen/*.npz")):
        d = np.load(f)
        if "skipped" in d.files:
            continue
        rows.append({k: d[k] for k in d.files})
    if not rows:
        return
    print(f"\n{len(rows)} sessions; condition minus spontaneous (mean, n>0/n, Wilcoxon p):")
    for c in CONDS[1:]:
        dd = np.array([r[c][0] - r["spontaneous"][0] for r in rows if c in r])
        da = np.array([r[c][1] - r["spontaneous"][1] for r in rows if c in r])
        if len(dd) < 3:
            continue
        p1 = wilcoxon(dd).pvalue if len(dd) >= 6 else np.nan
        p2 = wilcoxon(da).pvalue if len(da) >= 6 else np.nan
        print(f"  {c:20s} d(directedness) {dd.mean():+.3f} ({np.sum(dd > 0)}/{len(dd)}, p={p1:.3g})   "
              f"d(advantage) {da.mean():+.4f} ({np.sum(da > 0)}/{len(da)}, p={p2:.3g})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
