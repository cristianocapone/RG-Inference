"""
Do inference-RG blocks respect brain areas, and is the inter-area coarse-grained
coupling directed?  Stringer 2019 eight-probe Neuropixels, 3 mice; unit -> area from
the probe depth borders (probeBorders.mat, as in the authors' ephysLoad.m).

Per mouse, N = 512 units (random among labelled units with rate > 0.2 Hz; "top" =
most active as alternative), dt = 10 ms, three pairing rules.  Per RG level:
  area_purity   mean over blocks of the largest area fraction (membership-weighted)
  probe_purity  same with probe identity
  dir_pres      preservation of the inter-area directed structure: Pearson correlation
                between the antisymmetric part of the area x area coupling matrix at
                level k (A_k[a,b] = membership-weighted mean of J^(k) from area b to
                area a) and the one at level 0
  sym_area      symmetry index of the area x area matrix at level k
Also printed: level-0 inter-area directionality (A - A^T) for every mouse and the sign
consistency across mice for area pairs present in >= 2 mice.
Figure: Figures/Stringer_areas.png; results: results/stringer_areas.npz.

Run:  env -i /opt/homebrew/bin/python3 invivo_areas.py
"""

import os
import time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import infer_J
from rg_metrics import pairs_for, RULES, COLORS
from spiking_ei_rg import symmetry

D = "data/stringer2019_neuropixels"
MICE = {"Krebs": 3811.0, "Waksman": 3633.0, "Robbins": 3323.0}
AREAS = ["FrCtx", "FrMoCtx", "SomMoCtx", "SSCtx", "V1", "V2", "RSP", "CP", "LS", "LH", "HPF", "TH", "SC", "MB"]
DT = 0.010
L = 9
N_KEEP = 512


def load_units(mouse, tstart, dt=DT):
    """Binned activity (T x units), per-unit rate, area index (-1 = unlabelled), probe."""
    spks = np.atleast_1d(sio.loadmat(f"{D}/spks{mouse}_Feb18.mat", squeeze_me=True,
                                     struct_as_record=False)["spks"])
    pb = sio.loadmat(f"{D}/probeBorders.mat", squeeze_me=True, struct_as_record=False)["probeBorders"]
    borders = next(np.atleast_1d(m.borders) for m in pb if m.mouseName == mouse)
    tmax = max(np.asarray(p.st).max() for p in spks)
    T = int((tmax - tstart) / dt)
    cols, rates, area, probe = [], [], [], []
    for k, p in enumerate(spks):
        st, clu = np.asarray(p.st, float).ravel(), np.asarray(p.clu).ravel().astype(int)
        wh = np.asarray(p.Wheights, float).ravel()
        segs = [(s.acronym, float(s.lowerBorder), float(s.upperBorder)) for s in np.atleast_1d(borders[k])]
        keep = (st >= tstart) & (st < tstart + T * dt)
        st, clu = st[keep], clu[keep]
        b = ((st - tstart) / dt).astype(np.int64)
        for c in np.unique(clu):
            m = clu == c
            col = np.zeros(T, np.uint8)
            col[b[m]] = 1
            cols.append(col)
            rates.append(m.sum() / (T * dt))
            h = wh[c - 1] if c - 1 < len(wh) else np.nan
            a = next((AREAS.index(ac) for ac, lo, up in segs if lo <= h < up), -1)
            area.append(a)
            probe.append(k)
    return np.stack(cols, 1), np.array(rates), np.array(area), np.array(probe)


def select_units(rates, area, mode, rng, min_rate=0.2):
    ok = np.flatnonzero((area >= 0) & (rates > min_rate))
    if mode == "top":
        idx = ok[np.argsort(-rates[ok])][:N_KEEP]
    else:
        idx = rng.choice(ok, size=min(N_KEEP, len(ok)), replace=False)
    return np.sort(idx[:len(idx) // 2 * 2])


def area_matrix(J, W):
    """W: blocks x areas membership fractions. A[a, b] = weighted mean of J[i, j]
    (j -> i) over block pairs with i in a, j in b (diagonal of J excluded)."""
    off = ~np.eye(len(J), dtype=bool)
    num = W.T @ (J * off) @ W
    den = W.T @ off.astype(float) @ W
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(den > 0, num / den, np.nan)
    return A


def antisym_vec(A, present):
    """Upper-triangle entries of (A - A^T) restricted to present areas."""
    S = (A - A.T)[np.ix_(present, present)]
    return S[np.triu_indices(len(present), 1)]


def rg_areas(B, area, probe, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    N = S.shape[1]
    Wa = np.eye(len(AREAS))[area]                     # units x areas
    Wp = np.eye(probe.max() + 1)[probe]
    M = np.eye(N, dtype=np.float32)                   # units x blocks
    present = np.flatnonzero(Wa.sum(0) > 0)
    out = {k: [] for k in ["area_purity", "probe_purity", "dir_pres", "sym_area", "N"]}
    A0 = None
    for k in range(n_levels):
        J = infer_J(S, k)
        Wab = (M.T @ Wa) / M.sum(0)[:, None]          # blocks x areas
        Wpb = (M.T @ Wp) / M.sum(0)[:, None]
        A = area_matrix(J, Wab)
        if A0 is None:
            A0, v0 = A, antisym_vec(A, present)
        v = antisym_vec(A, present)
        ok = np.isfinite(v) & np.isfinite(v0)
        out["dir_pres"].append(np.corrcoef(v[ok], v0[ok])[0, 1] if ok.sum() > 2 else np.nan)
        Ap = np.nan_to_num(A[np.ix_(present, present)])
        out["sym_area"].append(symmetry(Ap))
        out["area_purity"].append(Wab.max(1).mean())
        out["probe_purity"].append(Wpb.max(1).mean())
        out["N"].append(S.shape[1])
        if k == n_levels - 1 or S.shape[1] < 4:
            break
        p = pairs_for(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return {k: np.array(v, float) for k, v in out.items()}, A0, present


def main(modes=("random", "top")):
    os.makedirs("results", exist_ok=True)
    store, A0s = {}, {}
    for mouse, tstart in MICE.items():
        t0 = time.time()
        B_all, rates, area, probe = load_units(mouse, tstart)
        lab = area >= 0
        print(f"=== {mouse}: {len(rates)} units, {lab.sum()} labelled; per area: "
              + ", ".join(f"{AREAS[a]} {np.sum(area == a)}" for a in np.unique(area[lab]))
              + f"  [{time.time() - t0:.0f} s]", flush=True)
        for mode in modes:
            idx = select_units(rates, area, mode, np.random.default_rng(5))
            B, ar, pr = B_all[:, idx], area[idx], probe[idx]
            chance_area = np.sum((np.bincount(ar, minlength=len(AREAS)) / len(ar)) ** 2)
            chance_probe = np.sum((np.bincount(pr) / len(pr)) ** 2)
            print(f"  [{mode}] N={len(idx)}, areas present {[AREAS[a] for a in np.unique(ar)]}, "
                  f"chance same-area {chance_area:.2f}, same-probe {chance_probe:.2f}")
            for rule in RULES:
                t0 = time.time()
                o, A0, present = rg_areas(B, ar, pr, rule, L)
                for m in o:
                    store[f"{mouse}/{mode}/{rule}/{m}"] = np.pad(o[m], (0, L - len(o[m])), constant_values=np.nan)
                print(f"    {rule:12s} area purity {np.round(o['area_purity'], 2)}")
                print(f"    {'':12s} probe purity {np.round(o['probe_purity'], 2)}")
                print(f"    {'':12s} dir. preservation {np.round(o['dir_pres'], 2)}  sym_area {np.round(o['sym_area'], 2)}"
                      f"  [{time.time() - t0:.0f} s]", flush=True)
                if rule == "inference" and mode == "random":
                    A0s[mouse] = (A0, present)
    np.savez("results/stringer_areas.npz", **store)
    directionality_report(A0s)
    plot(store, list(A0s))


def directionality_report(A0s):
    print("\n=== level-0 inter-area directionality  d[a,b] = mean J(b->a) - mean J(a->b)  (positive: b drives a)")
    pairs = {}
    for mouse, (A, present) in A0s.items():
        Dm = A - A.T
        print(f"  {mouse}: areas {[AREAS[a] for a in present]}")
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                d = Dm[a, b]
                scale = np.nanmean(np.abs(A[np.ix_(present, present)]))
                pairs.setdefault((AREAS[a], AREAS[b]), []).append(d / scale)
    print("  pairs present in >= 2 mice (d / mean|A|, per mouse):")
    for (a, b), v in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 2:
            print(f"    {a:9s} <- {b:9s}: {np.round(v, 2)}   consistent sign: {len(set(np.sign(v))) == 1}")


def plot(store, mice):
    steps = np.arange(1, L + 1)
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for row, mode in enumerate(["random", "top"]):
        for col, (m, title) in enumerate([("area_purity", "area purity of blocks"),
                                          ("probe_purity", "probe purity of blocks"),
                                          ("dir_pres", "preservation of inter-area directionality (corr with level 0)"),
                                          ("sym_area", "symmetry of area x area coupling")]):
            ax = axs[row, col]
            for rule in RULES:
                A = np.stack([store[f"{mo}/{mode}/{rule}/{m}"] for mo in mice])
                y = np.nanmean(A, 0)
                e = np.nanstd(A, 0) / np.sqrt(np.sum(~np.isnan(A), 0))
                ax.errorbar(steps, y, e, fmt="o-", color=COLORS[rule], ms=3, capsize=2, label=rule)
            ax.set_title(f"[{mode} 512 units] {title}", fontsize=8)
            ax.set_xlabel("RG step")
            ax.axhline(0, color="0.6", ls=":")
        axs[row, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Stringer 2019, 3 mice, mean +/- sem", fontsize=10)
    fig.tight_layout()
    fig.savefig("Figures/Stringer_areas.png", dpi=170)
    print("saved Figures/Stringer_areas.png")


if __name__ == "__main__":
    main()
