"""
Directedness vs closure advantage on DANDI 002020 (Nakamuta et al.; MaxOne HD-MEA,
dissociated rat cortical cultures, 7 cultures recorded on consecutive days, 30 min of
spontaneous activity per session, channel-level threshold detections; CC-BY-4.0).

Per session:
  * spikes binned at 10 ms (binary per channel);
  * spatial de-duplication: channels are accepted in order of decreasing rate and a
    channel is rejected if an accepted channel lies within MIN_DIST um (one neuron is
    seen by neighbouring electrodes at 17.5 um pitch);
  * N = min(512, T/100) channels with rate >= 0.2 Hz; sessions with N < 64 skipped;
  * directedness = 1 - symmetry index of the lagged covariance D(1) at level 0;
  * closure along the RG for the inference (in+out), inference_out and correlation
    rules; advantage = mean over steps 2..L-1 of (rule - correlation).
Per-session results are cached in results/dandi/<session>.npz (resumable) and
collected in results/dandi_closure.npz: key = session, value = [directedness,
advantage inference, advantage output, N, mean rate, day index within culture].

Run from the folder holding data/dandi002020 and results/:
    python dandi_closure.py
"""

import os
import glob
import time
import numpy as np
import h5py

from closure_test import rg_closure
from rg_metrics import lagged_cov
from spiking_ei_rg import symmetry

ROOT = "data/dandi002020"
DT = 0.010
MIN_DIST = 40.0
MIN_RATE = 0.2
RULES = ["inference", "inference_out", "correlation"]


def load_session(path, dt=DT):
    with h5py.File(path, "r") as f:
        g = f["processing/ecephys"]
        ch = g["detected_spikes/source_channel"][()]
        t = g["detected_spikes/time_from_first_spike_seconds"][()]
        dur = float(g["conversion_provenance/duration_s"][0])
        mch = g["source_mapping/source_channel"][()]
        xy = np.c_[g["source_mapping/source_x"][()], g["source_mapping/source_y"][()]]
    pos = {c: p for c, p in zip(mch, xy)}
    chans = np.array(sorted(set(ch) & set(mch)))
    T = int(dur / dt)
    col = {c: i for i, c in enumerate(chans)}
    keep = (t >= 0) & (t < T * dt) & np.isin(ch, chans)
    B = np.zeros((T, len(chans)), np.uint8)
    B[(t[keep] / dt).astype(int), np.array([col[c] for c in ch[keep]])] = 1
    return B, np.array([pos[c] for c in chans])


def select(B, xy, n_max):
    rates = B.mean(0) / DT
    order = np.argsort(-rates)
    acc = []
    for i in order:
        if rates[i] < MIN_RATE or len(acc) >= n_max:
            break
        if acc and np.min(np.linalg.norm(xy[acc] - xy[i], axis=1)) < MIN_DIST:
            continue
        acc.append(i)
    acc = np.sort(np.array(acc[:len(acc) // 2 * 2], int))
    return B[:, acc], rates[acc]


def analyse(path):
    B, xy = load_session(path)
    T = B.shape[0]
    S, rates = select(B, xy, min(512, T // 100))
    N = S.shape[1]
    if N < 64:
        return None
    _, D = lagged_cov(S.astype(np.float32))
    direct = 1 - symmetry(D)
    L = min(8, int(np.log2(N)) - 1)
    cl = {r: rg_closure(S, r, L)["closure"] for r in RULES}
    adv = [float(np.nanmean((cl[r] - cl["correlation"])[1:L - 1])) for r in RULES[:2]]
    return np.array([direct, adv[0], adv[1], N, rates.mean()])


def main():
    os.makedirs("results/dandi", exist_ok=True)
    files = sorted(glob.glob(f"{ROOT}/sub-*/*.nwb"))
    print(f"{len(files)} sessions", flush=True)
    for path in files:
        ses = os.path.basename(path).split("_")[1].replace("ses-", "")
        out = f"results/dandi/{ses}.npz"
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            r = analyse(path)
        except Exception as e:                                  # corrupt / empty file
            print(f"{ses}: failed {e!r}", flush=True)
            continue
        np.savez(out, r=np.array([]) if r is None else r)
        if r is None:
            print(f"{ses}: skipped (fewer than 64 channels)", flush=True)
        else:
            print(f"{ses}: N={int(r[3])} rate {r[4]:.2f} Hz directedness {r[0]:.3f} adv inference {r[1]:+.3f} "
                  f"output {r[2]:+.3f}  [{time.time() - t0:.0f} s]", flush=True)
    collect()


def collect():
    res = {}
    by_culture = {}
    for f in sorted(glob.glob("results/dandi/*.npz")):
        r = np.load(f)["r"]
        if r.size == 0:
            continue
        ses = os.path.basename(f)[:-4]
        culture, date = ses.split("-")
        by_culture.setdefault(culture, []).append((date, ses, r))
    for culture, lst in by_culture.items():
        lst.sort()
        for day, (date, ses, r) in enumerate(lst):
            res[ses] = np.r_[r, day]
    np.savez("results/dandi_closure.npz", **res)
    X = np.array(list(res.values()))
    if len(X) > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(X[:, 0], X[:, 1])
        print(f"{len(X)} sessions from {len(by_culture)} cultures: Spearman(directedness, advantage) = {rho:.2f}, p = {p:.2g}")
        rho, p = spearmanr(X[:, 0], X[:, 2])
        print(f"   output-only advantage: rho = {rho:.2f}, p = {p:.2g}")


if __name__ == "__main__":
    main()
