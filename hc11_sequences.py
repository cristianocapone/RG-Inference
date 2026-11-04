"""
In-vivo test of the prediction where the dynamics is directional at the model's timescale:
hippocampal CA1 during spatial behaviour (theta sequences) and sleep (replay).
Grosmark & Buzsaki 2016, DANDI 000044 (CC BY 4.0; CRCNS hc-11): 8 sessions, 4 rats,
bilateral silicon probes, 50-150 units; epochs PRE sleep, Maze (1.6 m linear track), POST
sleep.  Only the spike times, epochs and behavioural states are streamed (~0.1 GB/session).
Per session and epoch, for bins of 10 and 20 ms: units with rate > 0.5 Hz in the epoch
(n <= T/100), noise-corrected lag-1 directedness, closure advantage over the
correlation-based rule of the inference-based (full, output) and lagged-correlation rules
(mean over RG steps 2..L-1 and over steps 2-3).
Cache results/hc11/<session>.npz.  Run from the folder holding results/:
    python hc11_sequences.py
"""

import glob
import os
import time
import numpy as np
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient

from allen_rest_vs_task import read_strings
from closure_test import rg_closure
from directedness import directedness
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

RULES = ["inference", "inference_out", "lagged_out", "correlation"]
DTS = [0.010, 0.020]
COLS = "directedness, adv inference, adv output, adv lagged, adv23 inference, adv23 output, adv23 lagged, raw directedness, n units, T bins"


def load(url):
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    idx = u["spike_times_index"][()]
    starts = np.r_[0, idx[:-1]]
    spikes = [u["spike_times"][a:b] for a, b in zip(starts, idx)]
    ep = f["intervals/epochs"]
    epochs = {lab.decode() if isinstance(lab, bytes) else str(lab): (float(s), float(e))
              for lab, s, e in zip(ep["label"][()], ep["start_time"][()], ep["stop_time"][()])}
    states = None
    if "processing/behavior/states" in f:
        g = f["processing/behavior/states"]
        try:
            states = {"label": read_strings(g["label"]), "start": g["start_time"][()], "stop": g["stop_time"][()]}
        except Exception as e:
            print("   states table not read:", e, list(g.keys())[:10], flush=True)
    f.close()
    return spikes, epochs, states


def evaluate(B):
    B = B[:, B.std(0) > 0]
    B = B[:, :B.shape[1] // 2 * 2]
    L = int(np.log2(B.shape[1])) - 1
    cl = {r: rg_closure(B, r, L)["closure"] for r in RULES}
    adv = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:3]]
    adv23 = [float(np.nanmean((cl[r] - cl["correlation"])[1:3])) for r in RULES[:3]]
    _, D = lagged_cov(B.astype(np.float32))
    return np.r_[directedness(B)[0], adv, adv23, 1 - symmetry(D), B.shape[1], len(B)]


def binned(spikes, t0, t1, dt):
    T = int((t1 - t0) / dt)
    B = np.zeros((T, len(spikes)), np.uint8)
    for j, s in enumerate(spikes):
        k = ((s[(s >= t0) & (s < t1)] - t0) / dt).astype(int)
        B[k[k < T], j] = 1
    return B


def run_session(url):
    spikes, epochs, states = load(url)
    out = {"columns": COLS, "n_units_total": len(spikes)}
    if states is not None:
        labs, cnt = np.unique(states["label"], return_counts=True)
        print("   states:", dict(zip(labs.tolist(), cnt.tolist())), flush=True)
    for name, (t0, t1) in epochs.items():
        for dt in DTS:
            B = binned(spikes, t0, t1, dt)
            rate = B.mean(0) / dt
            ok = np.flatnonzero(rate > 0.5)
            nmax = min(len(ok), len(B) // 100)
            if nmax < 16:
                continue
            sel = ok[np.argsort(-rate[ok])][:nmax]
            out[f"{name}/{int(dt * 1000)}"] = evaluate(B[:, np.sort(sel)])
            v = out[f"{name}/{int(dt * 1000)}"]
            print(f"   {name:10s} {int(dt * 1000):2d} ms: {int(v[8]):3d} units, {(t1 - t0) / 60:5.1f} min  dir {v[0]:.3f} (raw {v[7]:.2f})  "
                  f"adv inf {v[1]:+.3f} out {v[2]:+.3f} lag {v[3]:+.3f} | steps 2-3: {v[4]:+.3f} {v[5]:+.3f} {v[6]:+.3f}", flush=True)
    # sleep states inside PRE/POST, if available (NREM: replay; REM / WAKE separately)
    if states is not None:
        for lab in np.unique(states["label"]):
            m = states["label"] == lab
            segs = [(a, b) for a, b in zip(states["start"][m], states["stop"][m]) if b - a > 0]
            if not segs:
                continue
            for dt in DTS:
                B = np.concatenate([binned(spikes, a, b, dt) for a, b in segs]) if sum(b - a for a, b in segs) / dt >= 3000 else None
                if B is None:
                    continue
                rate = B.mean(0) / dt
                ok = np.flatnonzero(rate > 0.5)
                nmax = min(len(ok), len(B) // 100)
                if nmax < 16:
                    continue
                sel = ok[np.argsort(-rate[ok])][:nmax]
                key = f"state_{lab}/{int(dt * 1000)}"
                out[key] = evaluate(B[:, np.sort(sel)])
                v = out[key]
                print(f"   {key:16s}: {int(v[8]):3d} units, {len(B) * dt / 60:5.1f} min  dir {v[0]:.3f}  adv inf {v[1]:+.3f} out {v[2]:+.3f} lag {v[3]:+.3f}", flush=True)
    return out


def main():
    os.makedirs("results/hc11", exist_ok=True)
    with DandiAPIClient() as c:
        assets = [(a.path, a.get_content_url(follow_redirects=1, strip_query=True)) for a in c.get_dandiset("000044").get_assets()]
    for path, url in sorted(assets):
        ses = os.path.basename(path).split("_behavior")[0]
        f = f"results/hc11/{ses}.npz"
        if os.path.exists(f):
            continue
        t0 = time.time()
        print(f"=== {ses}", flush=True)
        try:
            out = run_session(url)
        except Exception as e:
            print(f"{ses}: failed {e!r}", flush=True)
            continue
        np.savez(f, **{k: np.asarray(v) for k, v in out.items()})
        print(f"   done [{time.time() - t0:.0f} s]", flush=True)
    summarize()


def summarize():
    from scipy.stats import wilcoxon
    rows = [dict(np.load(f)) for f in sorted(glob.glob("results/hc11/*.npz"))]
    print(f"\n{len(rows)} sessions")
    keys = sorted({k for r in rows for k in r if "/" in k})
    for k in keys:
        v = np.array([r[k] for r in rows if k in r])
        print(f"  {k:22s} (n={len(v)}): dir {v[:, 0].mean():.3f}  adv inference {v[:, 1].mean():+.4f} ({(v[:, 1] > 0).sum()}/{len(v)})"
              f"  output {v[:, 2].mean():+.4f} ({(v[:, 2] > 0).sum()}/{len(v)})  lagged {v[:, 3].mean():+.4f}  units {v[:, 8].mean():.0f}")
    for dt in (10, 20):
        pairs = [(r[f"MazeEpoch/{dt}"], r[f"PREEpoch/{dt}"]) for r in rows if f"MazeEpoch/{dt}" in r and f"PREEpoch/{dt}" in r]
        if len(pairs) >= 5:
            d = np.array([a - b for a, b in pairs])
            print(f"  Maze - PRE ({dt} ms, n={len(d)}): directedness {d[:, 0].mean():+.3f} ({(d[:, 0] > 0).sum()}/{len(d)}, p={wilcoxon(d[:, 0]).pvalue:.2g})"
                  f"  adv output {d[:, 2].mean():+.4f} ({(d[:, 2] > 0).sum()}/{len(d)}, p={wilcoxon(d[:, 2]).pvalue:.2g})")


if __name__ == "__main__":
    main()
