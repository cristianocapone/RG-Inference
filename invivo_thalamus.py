"""
Thalamo-cortical directionality along the RG (Stringer 2019, 3 mice).

invivo_areas.py found that the level-0 inferred couplings are consistently directed
from thalamus (TH) to its cortical / striatal targets (FrMoCtx 3/3 mice, SomMoCtx,
V1, CP 2/2).  Here we quantify it with a null and follow it along the RG:
  d_TC(k) = [ mean J^(k)(TH -> Ctx) - mean J^(k)(Ctx -> TH) ] / mean |A^(k)|
with A^(k) the membership-weighted area x area coupling at level k, Ctx = all cortical
areas, computed for the three pairing rules on the data and on shifted surrogates
(level 0: 5 surrogates for a null distribution; along the RG: 1 surrogate per rule).
Same for HPF -> Ctx and TH -> HPF.
Figure: Figures/Stringer_thalamus.png; results: results/stringer_thalamus.npz.

Run:  env -i /opt/homebrew/bin/python3 invivo_thalamus.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import infer_J
from rg_metrics import pairs_for, shift_surrogate, RULES, COLORS
from invivo_areas import load_units, select_units, area_matrix, AREAS, MICE, L

CTX = [AREAS.index(a) for a in ["FrCtx", "FrMoCtx", "SomMoCtx", "SSCtx", "V1", "V2", "RSP"]]
PAIRS = {"TH -> Ctx": (AREAS.index("TH"), CTX), "HPF -> Ctx": (AREAS.index("HPF"), CTX),
         "TH -> HPF": (AREAS.index("TH"), [AREAS.index("HPF")])}


def directed_index(A, src, dst):
    """(mean J(src -> dst) - mean J(dst -> src)) / mean |A| over present areas."""
    fwd = np.nanmean(A[dst, src])          # A[post, pre]
    bwd = np.nanmean(A[src, dst])
    scale = np.nanmean(np.abs(A))
    return (fwd - bwd) / scale if np.isfinite(fwd) and np.isfinite(bwd) and scale > 0 else np.nan


def rg_directed(B, area, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    N = S.shape[1]
    Wa = np.eye(len(AREAS))[area]
    M = np.eye(N, dtype=np.float32)
    out = {name: [] for name in PAIRS}
    for k in range(n_levels):
        J = infer_J(S, k)
        Wab = (M.T @ Wa) / M.sum(0)[:, None]
        A = area_matrix(J, Wab)
        for name, (src, dst) in PAIRS.items():
            out[name].append(directed_index(A, src, dst))
        if k == n_levels - 1 or S.shape[1] < 4:
            break
        p = pairs_for(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return {k: np.pad(np.array(v, float), (0, n_levels - len(v)), constant_values=np.nan) for k, v in out.items()}


def main(modes=("top", "random"), n_null=5):
    os.makedirs("results", exist_ok=True)
    store = {}
    for mouse, tstart in MICE.items():
        t0 = time.time()
        B_all, rates, area, probe = load_units(mouse, tstart)
        print(f"=== {mouse}  [{time.time() - t0:.0f} s]", flush=True)
        for mode in modes:
            idx = select_units(rates, area, mode, np.random.default_rng(5))
            B, ar = B_all[:, idx], area[idx]
            nTH, nCtx, nHPF = np.sum(ar == AREAS.index("TH")), np.sum(np.isin(ar, CTX)), np.sum(ar == AREAS.index("HPF"))
            # level-0 null distribution from shifted surrogates
            A0 = area_matrix(infer_J(B, 0), np.eye(len(AREAS))[ar])
            null = {name: [] for name in PAIRS}
            for s in range(n_null):
                Bs = shift_surrogate(B, np.random.default_rng(100 + s))
                As = area_matrix(infer_J(Bs, 0), np.eye(len(AREAS))[ar])
                for name, (src, dst) in PAIRS.items():
                    null[name].append(directed_index(As, src, dst))
            print(f"  [{mode}] N={len(idx)}: TH {nTH}, Ctx {nCtx}, HPF {nHPF} units")
            for name, (src, dst) in PAIRS.items():
                d0 = directed_index(A0, src, dst)
                nl = np.array(null[name])
                z = (d0 - nl.mean()) / (nl.std() + 1e-12)
                store[f"{mouse}/{mode}/level0/{name}"] = d0
                store[f"{mouse}/{mode}/null/{name}"] = nl
                print(f"    level 0 {name:10s}: d = {d0:+.3f}   null {nl.mean():+.3f} +/- {nl.std():.3f}   z = {z:+.1f}")
            Bs = shift_surrogate(B, np.random.default_rng(7))
            for rule in RULES:
                r = rg_directed(B, ar, rule, L)
                s = rg_directed(Bs, ar, rule, L)
                for name in PAIRS:
                    store[f"{mouse}/{mode}/{rule}/real/{name}"] = r[name]
                    store[f"{mouse}/{mode}/{rule}/surr/{name}"] = s[name]
                print(f"    {rule:12s} TH->Ctx along RG: {np.round(r['TH -> Ctx'], 2)}   surr {np.round(s['TH -> Ctx'], 2)}", flush=True)
    np.savez("results/stringer_thalamus.npz", **store)
    plot(store)


def plot(store):
    mice = list(MICE)
    steps = np.arange(1, L + 1)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for row, mode in enumerate(["top", "random"]):
        for col, name in enumerate(PAIRS):
            ax = axs[row, col]
            for rule in RULES:
                A = np.stack([store[f"{mo}/{mode}/{rule}/real/{name}"] for mo in mice])
                S = np.stack([store[f"{mo}/{mode}/{rule}/surr/{name}"] for mo in mice])
                y, e = np.nanmean(A, 0), np.nanstd(A, 0) / np.sqrt(np.sum(~np.isnan(A), 0))
                ax.errorbar(steps, y, e, fmt="o-", color=COLORS[rule], ms=3, capsize=2, label=rule)
                ax.plot(steps, np.nanmean(S, 0), "--", color=COLORS[rule], lw=1)
            ax.axhline(0, color="0.6", ls=":")
            ax.set_title(f"[{mode} 512 units] directed index {name} (dashed: surrogate)", fontsize=8)
            ax.set_xlabel("RG step")
        axs[row, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Stringer 2019, 3 mice, mean +/- sem; index = (J src->dst - J dst->src) / mean|A|", fontsize=9)
    fig.tight_layout()
    fig.savefig("Figures/Stringer_thalamus.png", dpi=170)
    print("saved Figures/Stringer_thalamus.png")


if __name__ == "__main__":
    main()
