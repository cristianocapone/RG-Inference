"""
In vivo check: Stringer, Pachitariu et al. 2019 (Science), eight-probe Neuropixels
recordings in 3 mice (Krebs, Waksman, Robbins), spontaneous period starting at
tstart = 3811 / 3633 / 3323 s (from the authors' ephysLoad.m).  Data: figshare
10.25378/janelia.7739750, spks.zip (CC BY-NC 4.0); spks<Mouse>_Feb18.mat holds, per
probe k, spks(k).st (spike times, s), spks(k).clu (cluster id), spks(k).Wheights.

Sorted units are binned at dt = 10 ms over the spontaneous period, the N = 512 most
active units (all probes pooled) are kept, and the same metrics as for the cultures
are computed (rg_metrics.py): symmetry of inferred J and of D(1) vs shifted surrogate,
held-out cross-unit predictability, variance exponent; mean +/- sem over mice.

Run:  env -i /opt/homebrew/bin/python3 invivo_stringer.py <dir with spks*.mat> [--inspect]
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rg_metrics import data_vs_surrogate, pad, RULES, COLORS

MICE = {"Krebs": 3811.0, "Waksman": 3633.0, "Robbins": 3323.0}
DT = 0.010
L = 9


def load_probes(path):
    """Returns list of (st [s], clu [int]) per probe, for v7 or v7.3 .mat files."""
    try:
        import scipy.io as sio
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        spks = d["spks"]
        spks = np.atleast_1d(spks)
        return [(np.asarray(p.st, float).ravel(), np.asarray(p.clu).ravel().astype(int)) for p in spks]
    except NotImplementedError:
        import h5py
        out = []
        with h5py.File(path, "r") as f:
            g = f["spks"]
            n = g["st"].shape[0] if g["st"].ndim == 2 else 1
            for k in range(g["st"].shape[-1] if g["st"].ndim == 2 else 1):
                st = f[g["st"][0, k]][()].ravel() if g["st"].ndim == 2 else g["st"][()].ravel()
                clu = f[g["clu"][0, k]][()].ravel() if g["clu"].ndim == 2 else g["clu"][()].ravel()
                out.append((np.asarray(st, float), np.asarray(clu).astype(int)))
        return out


def bin_mouse(path, tstart, dt=DT, n_keep=512, max_T_s=None):
    probes = load_probes(path)
    tmax = max(st.max() for st, _ in probes)
    if max_T_s is not None:
        tmax = min(tmax, tstart + max_T_s)
    T = int((tmax - tstart) / dt)
    cols, rates, ids = [], [], []
    for k, (st, clu) in enumerate(probes):
        keep = (st >= tstart) & (st < tstart + T * dt)
        st, clu = st[keep], clu[keep]
        b = ((st - tstart) / dt).astype(np.int64)
        for c in np.unique(clu):
            m = clu == c
            col = np.zeros(T, np.uint8)
            col[b[m]] = 1
            cols.append(col)
            rates.append(m.sum() / (T * dt))
            ids.append((k, int(c)))
    B = np.stack(cols, 1)
    rates = np.array(rates)
    order = np.argsort(-rates)[:n_keep]
    return B[:, order], rates[order], [ids[i] for i in order], len(cols)


def main(root, inspect=False):
    store = {}
    mice = []
    for mouse, tstart in MICE.items():
        path = next((os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
                     if f.startswith(f"spks{mouse}") and f.endswith(".mat")), None)
        if path is None:
            print(f"  {mouse}: file not found under {root}")
            continue
        t0 = time.time()
        B, rates, ids, n_all = bin_mouse(path, tstart)
        T, N = B.shape
        print(f"=== {mouse}: {n_all} units total, kept N={N}, T={T} bins ({T * DT / 60:.0f} min), "
              f"rate median {np.median(rates):.2f} Hz (min {rates.min():.2f}), "
              f"probes used {len(set(i[0] for i in ids))}  [{time.time() - t0:.0f} s]", flush=True)
        if inspect:
            continue
        mice.append(mouse)
        res = data_vs_surrogate(B, L)
        for rule in RULES:
            r, s = res[rule]["real"], res[rule]["surr"]
            print(f"  {rule:12s} sym_J real {np.round(r['sym_J'], 2)}  surr {np.round(s['sym_J'], 2)}")
            print(f"  {'':12s} sym_D real {np.round(r['sym_D'], 2)}  surr {np.round(s['sym_D'], 2)}")
            print(f"  {'':12s} r2-r2auto {np.round(r['r2'] - r['r2_auto'], 3)}  r2-r2ag {np.round(r['r2'] - r['r2_ag'], 3)}  exps {np.round(r['exp'], 2)}", flush=True)
            for m in ["sym_J", "sym_D", "r2", "r2_auto", "r2_ag", "exp"]:
                store[f"{mouse}/{rule}/real/{m}"] = pad(r[m], L)
                store[f"{mouse}/{rule}/surr/{m}"] = pad(s[m], L)
    if inspect or not mice:
        return
    os.makedirs("results", exist_ok=True)
    np.savez("results/stringer_metrics.npz", **store)
    plot(store, mice)


def plot(store, mice):
    steps = np.arange(1, L + 1)

    def agg(A):
        n = np.sum(~np.isnan(A), 0)
        return np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1))

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, key, title in [
        (axs[0, 0], "symJ", "symmetry of inferred J (solid: data, dashed: surrogate)"),
        (axs[0, 1], "xJ", "excess symmetry of J (data - surrogate)"),
        (axs[0, 2], "xD", "excess symmetry of lagged covariance D(1)"),
        (axs[1, 0], "r2", "held-out R2 (dotted: autoregressive only)"),
        (axs[1, 1], "r2x", "cross-unit predictability R2 - R2_auto (dashed: beyond global mode, R2 - R2_ag)"),
        (axs[1, 2], "exp", "variance scaling exponent (dashed: surrogate)"),
    ]:
        for rule in RULES:
            g = lambda kind, m: np.stack([store[f"{mo}/{rule}/{kind}/{m}"] for mo in mice])
            if key == "symJ":
                y, e = agg(g("real", "sym_J")); ys, _ = agg(g("surr", "sym_J"))
                ax.plot(steps, ys, "--", color=COLORS[rule], lw=1)
            elif key == "xJ":
                y, e = agg(g("real", "sym_J") - g("surr", "sym_J"))
            elif key == "xD":
                y, e = agg(g("real", "sym_D") - g("surr", "sym_D"))
            elif key == "r2":
                y, e = agg(g("real", "r2")); ya, _ = agg(g("real", "r2_auto"))
                ax.plot(steps, ya, ":", color=COLORS[rule], lw=1)
            elif key == "r2x":
                y, e = agg(g("real", "r2") - g("real", "r2_auto"))
                ys, _ = agg(g("real", "r2") - g("real", "r2_ag"))
                ax.plot(steps, ys, "--", color=COLORS[rule], lw=1)
            else:
                y, e = agg(g("real", "exp")); ys, _ = agg(g("surr", "exp"))
                ax.plot(steps[1:], ys[:L - 1], "--", color=COLORS[rule], lw=1)
            x = steps[1:] if key == "exp" else steps
            y, e = (y[:L - 1], e[:L - 1]) if key == "exp" else (y, e)
            ax.errorbar(x, y, e, fmt="o-", color=COLORS[rule], ms=3, capsize=2, label=rule)
        ax.axhline(0, color="0.6", ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("RG step")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Stringer 2019 Neuropixels, spontaneous, n = {len(mice)} mice, N = 512 units, mean +/- sem", fontsize=10)
    fig.tight_layout()
    fig.savefig("Figures/Stringer_excess.png", dpi=170)
    print("saved Figures/Stringer_excess.png")


if __name__ == "__main__":
    main(sys.argv[1], inspect="--inspect" in sys.argv)
