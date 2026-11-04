"""
Statistics requested by the referees on the stored results (no new computation):
  - advantage of the inference-based rules over the correlation-based one per RG step,
    for every culture, and for three step windows (2-7, 2-3, 5-8)
  - Spearman correlation with the lag-1 directedness: leave-one-out range and
    permutation p-value (20000 permutations), for the output and the in+out rule
  - absolute held-out R^2 (coarse and micro) along the RG, cultures / mice / spiking
  - 5-fold contiguous cross-validation of the advantage (closure_cv.npz, if present)
  - Allen: advantage vs number of observed units for all three rules (allen_final/)
Output: results/a2_stats.txt and Figures/Supp_perstep.png.
Run from the folder holding results/:  python a2_stats.py
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
MICE = ["Krebs", "Waksman", "Robbins"]
WIN = {"2-7": slice(1, 7), "2-3": slice(1, 3), "5-8": slice(4, 8)}
RULES = ["inference_out", "inference", "lagged_out"]
LAB = {"inference_out": "output", "inference": "in+out", "lagged_out": "lagged"}
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def pad(a, n=8):
    a = np.asarray(a, float)
    return np.pad(a, (0, max(0, n - len(a))), constant_values=np.nan)[:n]


def perm_p(x, y, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = spearmanr(x, y)[0]
    r = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
    return float(np.mean(np.abs(r) >= abs(r0)))


def main():
    cl0, dr = np.load("results/closure.npz"), np.load("results/directedness.npz")
    cl = {k: pad(cl0[k]) for k in cl0}
    x = np.array([float(dr[f"cultures/{c}/lag1"]) for c in CULT])
    adv = {r: np.array([cl[f"cultures/{c}/{r}/closure"] - cl[f"cultures/{c}/correlation/closure"] for c in CULT]) for r in RULES}

    say("=== Per-step advantage over correlation, cultures (steps 1..8; step 1 = no reduction)")
    for r in RULES:
        say(f"--- {LAB[r]} rule")
        for c, a, d in zip(CULT, adv[r], x):
            say(f"  {c:3s} dir {d:.3f}  " + " ".join(f"{v:+.3f}" if not np.isnan(v) else "   nan" for v in a))
        for s in range(1, 8):
            v = adv[r][:, s]
            v = v[~np.isnan(v)]
            if len(v) >= 5:
                say(f"  step {s + 1}: mean {v.mean():+.4f}, {np.sum(v > 0)}/{len(v)} > 0, Wilcoxon p = {wilcoxon(v).pvalue:.3f}")

    say("\n=== Spearman(directedness lag 1, advantage) by step window; leave-one-out; permutation")
    for r in RULES:
        for w, sl in WIN.items():
            y = np.nanmean(adv[r][:, sl], 1)
            rho, p = spearmanr(x, y)
            loo = [spearmanr(np.delete(x, i), np.delete(y, i))[1] for i in range(len(x))]
            say(f"  {LAB[r]:7s} steps {w}: rho {rho:+.2f} p {p:.3f} | LOO p {min(loo):.3f}-{max(loo):.3f} | perm p {perm_p(x, y):.3f}"
                f" | mean adv {y.mean():+.4f}, {np.sum(y > 0)}/{len(y)} > 0")

    say("\n=== Absolute held-out R^2 along the RG (mean over recordings): micro / coarse(correlation) / coarse(output)")
    for grp, lst in [("cultures", CULT), ("mice", MICE), ("spiking", ["clustered", "chain"])]:
        rm = np.nanmean([cl[f"{grp}/{n}/correlation/r2_micro"] for n in lst], 0)
        rc = np.nanmean([cl[f"{grp}/{n}/correlation/r2_coarse"] for n in lst], 0)
        ro = np.nanmean([cl[f"{grp}/{n}/inference_out/r2_coarse"] for n in lst], 0)
        say(f"  {grp:9s} micro  " + " ".join(f"{v:.3f}" for v in rm))
        say(f"  {'':9s} corr   " + " ".join(f"{v:.3f}" for v in rc))
        say(f"  {'':9s} output " + " ".join(f"{v:.3f}" for v in ro))

    cv = None
    if os.path.exists("results/closure_cv.npz"):
        cv0 = np.load("results/closure_cv.npz")
        cv = {k: pad(cv0[k]) for k in cv0}
        say("\n=== 5-fold contiguous cross-validation (advantage over steps 2-7; mean +/- sd over folds)")
        for r in RULES:
            ys = []
            for c in CULT:
                a = np.array([np.nanmean((cv[f"cultures/{c}/fold{f}/{r}/closure"] - cv[f"cultures/{c}/fold{f}/correlation/closure"])[1:7])
                              for f in range(5) if f"cultures/{c}/fold{f}/{r}/closure" in cv])
                if len(a):
                    ys.append(a.mean())
                    say(f"  {LAB[r]:7s} {c:3s}: {a.mean():+.4f} +/- {a.std(ddof=1) if len(a) > 1 else np.nan:.4f}  ({np.sum(a > 0)}/{len(a)} folds > 0)")
            if len(ys) == len(CULT):
                rho, p = spearmanr(x, ys)
                say(f"  {LAB[r]:7s} fold-averaged: rho {rho:+.2f} p {p:.3f}, {np.sum(np.array(ys) > 0)}/9 > 0")
        for m in MICE:
            if f"mice/{m}/fold0/inference_out/closure" in cv:
                a = np.array([np.nanmean((cv[f"mice/{m}/fold{f}/inference_out/closure"] - cv[f"mice/{m}/fold{f}/correlation/closure"])[1:7]) for f in range(5)])
                say(f"  output  {m}: {a.mean():+.4f} +/- {a.std(ddof=1):.4f}")

    say("\n=== Allen: advantage vs number of observed units (random units, mean over sessions)")
    rows = [dict(np.load(f)) for f in sorted(glob.glob("results/allen_final/*.npz"))]
    rows = [r for r in rows if "skipped" not in r]
    dens = {}
    for cond in ["spontaneous", "drifting_gratings", "natural_movie_three"]:
        for n in [64, 128, 256, 512]:
            v = np.array([r[f"dens/{cond}/{n}"] for r in rows if f"dens/{cond}/{n}" in r])
            if len(v):
                dens[(cond, n)] = v
                say(f"  {cond:20s} n={n:3d} (sessions {len(v):2d}): dir {v[:, 0].mean():.3f}  adv in+out {v[:, 1].mean():+.4f}  output {v[:, 2].mean():+.4f}  lagged {v[:, 3].mean():+.4f}")
        for j, r in [(1, "in+out"), (2, "output"), (3, "lagged")]:
            pairs = [(x_[f"dens/{cond}/64"][j], x_[f"dens/{cond}/512"][j]) for x_ in rows if f"dens/{cond}/64" in x_ and f"dens/{cond}/512" in x_]
            if len(pairs) >= 6:
                d = np.array([b - a for a, b in pairs])
                say(f"  {cond:20s} {r:7s} (512) - (64): {d.mean():+.4f}, {np.sum(d > 0)}/{len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.3g}")

    with open("results/a2_stats.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ figure
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    steps = np.arange(1, 9)
    cmap = plt.get_cmap("Reds")
    ax = axs[0, 0]
    for c, d in zip(CULT, x):
        col = cmap(0.3 + 0.7 * (d - x.min()) / (x.max() - x.min()))
        ax.plot(steps, cl[f"cultures/{c}/correlation/closure"], "-", color="#8c8c00", lw=1, alpha=0.6)
        ax.plot(steps, cl[f"cultures/{c}/inference_out/closure"], "-", color=col, lw=1.2)
    ax.axhline(1, color="0.6", ls=":")
    ax.set_xlabel("RG step"); ax.set_ylabel("closure"); ax.set_title("(a) cultures: correlation (olive), output rule (red, darker = more directional)", fontsize=9, loc="left")
    ax = axs[0, 1]
    for c, d in zip(CULT, x):
        col = cmap(0.3 + 0.7 * (d - x.min()) / (x.max() - x.min()))
        ax.plot(steps, adv["inference_out"][CULT.index(c)], "o-", color=col, ms=3, lw=1.2, label=f"{c} ({d:.2f})")
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("RG step"); ax.set_ylabel("advantage of the output rule over correlation"); ax.set_title("(b) advantage per step, every culture (lag-1 directedness)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axs[1, 0]
    for grp, lst, ls in [("cultures", CULT, "-"), ("mice", MICE, "--")]:
        ax.plot(steps, np.nanmean([cl[f"{grp}/{n}/correlation/r2_micro"] for n in lst], 0), "k" + ls, label=f"R2(y|x) micro, {grp}")
        ax.plot(steps, np.nanmean([cl[f"{grp}/{n}/correlation/r2_coarse"] for n in lst], 0), ls, color="#8c8c00", label=f"R2(y|y) correlation, {grp}")
        ax.plot(steps, np.nanmean([cl[f"{grp}/{n}/inference_out/r2_coarse"] for n in lst], 0), ls, color="#8b0000", label=f"R2(y|y) output rule, {grp}")
    ax.set_xlabel("RG step"); ax.set_ylabel("held-out R^2 (absolute)"); ax.set_title("(c) absolute R^2 behind the closure ratio", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1, 1]
    for cond, ls in [("spontaneous", "-"), ("drifting_gratings", "--"), ("natural_movie_three", ":")]:
        ns = [n for n in [64, 128, 256, 512] if (cond, n) in dens]
        for j, r, col in [(1, "in+out", "#d62728"), (2, "output", "#8b0000"), (3, "lagged", "#5e3c99")]:
            ax.plot(ns, [dens[(cond, n)][:, j].mean() for n in ns], "o" + ls, color=col, ms=3, label=f"{r}, {cond}" if cond == "spontaneous" else None)
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xscale("log", base=2); ax.set_xlabel("observed units (random)"); ax.set_ylabel("advantage over correlation")
    ax.set_title("(d) Allen, 30 sessions: spontaneous (solid), gratings (dashed), movie (dotted)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_perstep.png", dpi=170)
    fig.savefig("Figures/Supp_perstep.pdf")
    print("saved results/a2_stats.txt, Figures/Supp_perstep.png")


if __name__ == "__main__":
    main()
