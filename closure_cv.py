"""
Closure with 5-fold contiguous cross-validation (referee control for the single 80/20 split).

The recording is cut into 5 contiguous segments; for each fold f the segment f is held
out, the groupings (and J, and the ridge) are formed on the other four, and the closure
is evaluated on segment f.  Cultures (9) and mice (3), rules inference (in+out),
inference (output), lagged-correlation (output), correlation.
Results: results/closure_cv.npz, keys <grp>/<name>/fold<f>/<rule>/{closure,r2_coarse,r2_micro}.
Run from the folder holding data/ and results/:  python closure_cv.py
"""

import os
import time
import numpy as np

from closure_test import ridge_r2, pairs5, infer_J_est, L
from real_data_rg import load_culture
from invivo_stringer import bin_mouse, MICE

RULES = ["inference", "inference_out", "lagged_out", "correlation"]
NF = 5
OUT = "results/closure_cv.npz"


def fold_mask(T, f, nf=NF):
    edges = np.linspace(0, T, nf + 1).astype(int)
    m = np.zeros(T, bool)
    m[edges[f]:edges[f + 1]] = True
    return m


def closure_fold(S0, M, f, nf=NF):
    """Closure with segment f held out; the training set is the concatenation of the other segments."""
    te = fold_mask(len(S0), f, nf)
    Y = S0 @ (M / M.sum(0))
    mx, my = S0[~te].mean(0), Y[~te].mean(0)
    dX, dY = S0 - mx, Y - my
    tr1 = ~te[:-1] & ~te[1:]              # transitions entirely inside the training segments
    te1 = te[:-1] & te[1:]
    r2c = ridge_r2(dY[:-1][tr1], dY[1:][tr1], dY[:-1][te1], dY[1:][te1])
    r2m = ridge_r2(dX[:-1][tr1], dY[1:][tr1], dX[:-1][te1], dY[1:][te1])
    return r2c, r2m


def rg_closure_fold(B, rule, n_levels, f, seed=0):
    rng = np.random.default_rng(seed)
    S0 = B.astype(np.float32)
    S = S0
    tr = ~fold_mask(len(S0), f)
    M = np.eye(S.shape[1], dtype=np.float32)
    out = {"r2_coarse": [], "r2_micro": [], "closure": []}
    for k in range(n_levels):
        r2c, r2m = closure_fold(S0, M, f)
        out["r2_coarse"].append(r2c)
        out["r2_micro"].append(r2m)
        out["closure"].append(r2c / r2m if r2m > 0 else np.nan)
        if k == n_levels - 1 or S.shape[1] < 4:
            break
        Str = S[tr]
        J = infer_J_est(Str, k) if rule.startswith("inference") else None
        p = pairs5(rule, Str, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
    return {k: np.pad(np.array(v, float), (0, n_levels - len(v)), constant_values=np.nan) for k, v in out.items()}


def datasets():
    for c in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]:
        B_all, _, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        yield "cultures", c, B_all[:, :N]
    for mouse, tstart in MICE.items():
        B, *_ = bin_mouse(f"data/stringer2019_neuropixels/spks{mouse}_Feb18.mat", tstart)
        yield "mice", mouse, B


def main():
    os.makedirs("results", exist_ok=True)
    store = dict(np.load(OUT)) if os.path.exists(OUT) else {}
    for grp, name, B in datasets():
        t0 = time.time()
        for f in range(NF):
            for rule in RULES:
                key = f"{grp}/{name}/fold{f}/{rule}"
                if f"{key}/closure" in store:
                    continue
                for q, v in rg_closure_fold(B, rule, L, f).items():
                    store[f"{key}/{q}"] = v
        np.savez(OUT, **store)
        adv = {r: np.array([np.nanmean((store[f"{grp}/{name}/fold{f}/{r}/closure"] - store[f"{grp}/{name}/fold{f}/correlation/closure"])[1:7])
                            for f in range(NF)]) for r in RULES[:3]}
        print(f"{grp}/{name}: N {B.shape[1]} T {B.shape[0]} | adv over folds (steps 2-7): " +
              "  ".join(f"{r} {a.mean():+.3f}+/-{a.std(ddof=1):.3f} ({np.sum(a > 0)}/{NF})" for r, a in adv.items()) +
              f"  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
