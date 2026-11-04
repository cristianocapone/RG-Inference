"""
In vivo: closure advantage vs bin size and vs coupling estimator (referee control).

Two alternatives to the partial-observation reading of the in-vivo result are tested:
  (i)  the 10-ms bin is longer than monosynaptic latencies, so the one-step model misses
       the directed interactions -> repeat at 2 and 5 ms;
  (ii) the mean-field inversion J = D C^-1 / f' is ill-conditioned when C is dominated
       by a global mode -> repeat with a ridge-regularised inversion and with exact
       maximum likelihood (ml_kinetic_ising.py).
Data: Stringer et al. 2019 (3 mice, spontaneous, 256 most active units) and the Allen
Neuropixels sessions (DANDI 000021; spike times cached once in data/allen_cache/, then
binned at any dt), spontaneous and natural movie three, 256 most active units, same
duration as in allen_final.py.  Rules: inference (in+out), inference (output),
lagged-correlation (output), correlation; groupings on the first 80% of the recording,
closure on the last 20% (closure_test.py).
Results: results/invivo_binsize.npz with keys
  <grp>/<name>[/<cond>]/dt<ms>/dir                      [directedness lag 1, raw, T, N, rate Hz]
  <grp>/<name>[/<cond>]/dt<ms>/<est>/<rule>/{closure,r2_coarse,r2_micro}   est in none, nmf, ridge, ml

Run from the folder holding data/ and results/:
    python invivo_binsize.py stringer            # 3 mice, 2/5/10 ms, all estimators
    python invivo_binsize.py cache 10            # stream and cache 10 Allen sessions
    python invivo_binsize.py allen               # cached sessions, 2/5/10 ms (ML at 5 and 10 ms)
    python invivo_binsize.py summary
"""

import glob
import os
import sys
import time
import numpy as np

from closure_test import rg_closure
from directedness import directedness

DTS = [0.002, 0.005, 0.010]
ESTS = ["nmf", "ridge", "ml"]
RULES_EST = ["inference", "inference_out"]
RULES_FREE = ["lagged_out", "correlation"]
N_UNITS = 256
N_CACHE = 512
OUT = "results/invivo_binsize.npz"
CACHE = "data/allen_cache"
ALLEN_CONDS = ["spontaneous", "natural_movie_three"]
ML_DTS_ALLEN = (0.005, 0.010)


def ms(dt):
    return f"dt{int(round(dt * 1000))}"


def evaluate(B, dt, prefix, store, ests):
    """All rules and estimators for one binned recording; results added to store under prefix/dt<ms>/."""
    B = B[:, B.std(0) > 0]
    B = B[:, :B.shape[1] // 2 * 2]
    L = int(np.log2(B.shape[1])) - 1
    key = f"{prefix}/{ms(dt)}"
    if f"{key}/dir" not in store:
        d, raw = directedness(B)
        store[f"{key}/dir"] = np.r_[d, raw, B.shape[0], B.shape[1], B.mean() / dt]
    for est, rules in [("none", RULES_FREE)] + [(e, RULES_EST) for e in ests]:
        for rule in rules:
            k = f"{key}/{est}/{rule}"
            if f"{k}/closure" in store:
                continue
            t0 = time.time()
            r = rg_closure(B, rule, L, estimator=est if est != "none" else "nmf")
            for q, v in r.items():
                store[f"{k}/{q}"] = v
            np.savez(OUT, **store)
            print(f"    {ms(dt)} {est:5s} {rule:14s} closure {np.round(r['closure'][:L - 1], 3)}  [{time.time() - t0:.0f} s]", flush=True)
    c = lambda e, r: store[f"{key}/{e}/{r}/closure"]
    adv = {f"{e}/{r}": float(np.nanmean((c(e, r) - c("none", "correlation"))[1:L - 1]))
           for e, rules in [("none", ["lagged_out"])] + [(e, RULES_EST) for e in ests] for r in rules}
    print(f"  {key}: T {B.shape[0]} N {B.shape[1]} dir {store[f'{key}/dir'][0]:.3f} | adv " +
          "  ".join(f"{k} {v:+.3f}" for k, v in adv.items()), flush=True)


def load_store():
    return dict(np.load(OUT)) if os.path.exists(OUT) else {}


def run_stringer():
    from invivo_stringer import bin_mouse, MICE
    store = load_store()
    for mouse, tstart in MICE.items():
        for dt in DTS:
            B, *_ = bin_mouse(f"data/stringer2019_neuropixels/spks{mouse}_Feb18.mat", tstart, dt=dt, n_keep=N_UNITS)
            evaluate(B, dt, f"mice/{mouse}", store, ESTS)


# ----------------------------------------------------------------------------- Allen
def cache_allen(n_max):
    """Stream the spike times of the sessions already analysed in allen_final.py and cache them."""
    from allen_rest_vs_task import session_urls
    from allen_final import load_all
    os.makedirs(CACHE, exist_ok=True)
    done = 0
    for path, url in session_urls():
        ses = os.path.basename(path).replace(".nwb", "")
        f_res, f_cache = f"results/allen_final/{ses}.npz", f"{CACHE}/{ses}.npz"
        if not os.path.exists(f_res) or "skipped" in np.load(f_res):
            continue
        if not os.path.exists(f_cache):
            t0 = time.time()
            spikes, fr, intervals, inv = load_all(url)
            keep = np.sort(np.argsort(-fr)[:N_CACHE])             # the most active units only (disk)
            spikes, fr = [spikes[i] for i in keep], fr[keep]
            idx = np.cumsum([len(s) for s in spikes])
            np.savez(f_cache, spikes=np.concatenate(spikes).astype(np.float64), index=idx, fr=fr, inv=inv,
                     T_eq_min=float(np.load(f_res)["T_eq_min"]), **{f"iv/{c}": v for c, v in intervals.items()})
            print(f"cached {ses}: {len(spikes)} units, {idx[-1] / 1e6:.1f} M spikes  [{time.time() - t0:.0f} s]", flush=True)
        done += 1
        if done >= n_max:
            break


def bin_condition(spikes, bins, dt):
    """T x N binary matrix for the global bins `bins` (sorted indices) at resolution dt."""
    B = np.zeros((len(bins), len(spikes)), np.uint8)
    for j, s in enumerate(spikes):
        k = (s / dt).astype(np.int64)
        pos = np.searchsorted(bins, k)
        ok = (pos < len(bins)) & (bins[np.minimum(pos, len(bins) - 1)] == k)
        B[pos[ok], j] = 1
    return B


def run_allen():
    from allen_rest_vs_task import condition_bins
    store = load_store()
    for f in sorted(glob.glob(f"{CACHE}/*.npz")):
        ses = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        spikes = np.split(d["spikes"], d["index"][:-1])
        top = np.sort(np.argsort(-d["fr"])[:N_UNITS])
        spikes = [spikes[i] for i in top]
        t_max = max(s[-1] for s in spikes if len(s))
        print(f"=== {ses}: {len(d['fr'])} units, T_eq {float(d['T_eq_min']):.1f} min", flush=True)
        for cond in ALLEN_CONDS:
            if f"iv/{cond}" not in d:
                continue
            for dt in DTS:
                bins = condition_bins(d[f"iv/{cond}"], d["inv"], t_max, dt)
                T_eq = int(float(d["T_eq_min"]) * 60 / dt)
                B = bin_condition(spikes, bins[:T_eq], dt)
                evaluate(B, dt, f"allen/{ses}/{cond}", store, [e for e in ESTS if e != "ml" or dt in ML_DTS_ALLEN])


def summary():
    from scipy.stats import wilcoxon
    store = load_store()
    recs = sorted({k.rsplit("/dt", 1)[0] for k in store if "/dir" in k})
    for grp in ["mice", "allen"]:
        rr = [r for r in recs if r.startswith(grp)]
        if not rr:
            continue
        print(f"\n=== {grp}: {len(rr)} recordings")
        for dt in DTS:
            for est, rule in [("none", "lagged_out")] + [(e, r) for e in ESTS for r in RULES_EST]:
                vals, dirs = [], []
                for r in rr:
                    k = f"{r}/{ms(dt)}"
                    if f"{k}/{est}/{rule}/closure" in store and f"{k}/none/correlation/closure" in store:
                        c = store[f"{k}/{est}/{rule}/closure"] - store[f"{k}/none/correlation/closure"]
                        L = int(np.sum(~np.isnan(c)))
                        vals.append(float(np.nanmean(c[1:max(2, L - 1)])))
                        dirs.append(store[f"{k}/dir"][0])
                if vals:
                    v = np.array(vals)
                    p = wilcoxon(v).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan
                    print(f"  {ms(dt):5s} {est:5s} {rule:14s} n={len(v):2d} dir {np.mean(dirs):.3f}  adv {v.mean():+.4f} ({np.sum(v > 0)}/{len(v)}, p={p:.2g})")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if what == "stringer":
        run_stringer()
    elif what == "cache":
        cache_allen(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif what == "allen":
        run_allen()
    elif what == "test":
        from invivo_stringer import bin_mouse, MICE
        B, *_ = bin_mouse("data/stringer2019_neuropixels/spksKrebs_Feb18.mat", MICE["Krebs"], dt=0.010, n_keep=N_UNITS)
        evaluate(B, 0.010, "mice/Krebs", load_store(), ESTS)
    summary()
