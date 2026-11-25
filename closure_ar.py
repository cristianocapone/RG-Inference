"""
Closure with multi-lag and nonlinear predictors: is the linear one-step closure hiding anything?

The paper's closure is R^2[y(t+1) | y(t)] / R^2[y(t+1) | S(t)] with linear ridge regressions
on one bin of history. Here the same partitions (the pairing rules of the paper, formed on the
first 80% of every recording) are evaluated with richer predictors, used for both the coarse
and the micro regression:
  lin<p>   linear VAR(p): y(t+1) from [y(t), ..., y(t-p+1)]; p = 1 is the paper's metric,
           a long autoregression approximates an ARMA model
  arma<p>  linear ARMA(p, p), Hannan-Rissanen: innovations estimated as the residuals of a long
           VAR(10) fitted on the training part, then y(t+1) regressed on p lags of y and p lags
           of the innovations
  nl<p>    nonlinear ARX: the same lagged inputs, standardised, plus 1024 random tanh features
           (random-feature network with the linear part kept, ridge on top)
  nn<p>    as nl<p>, with the lagged block variables added to the micro inputs. Only the LINEAR
           part is nested: the random tanh features of the blocks are not in the micro design.
Caveat: with random features the micro predictor (thousands of inputs) gains much less from the
nonlinearity than the coarse one (few inputs); at level 3 the coarse R^2 can exceed the micro R^2,
so closure > 1 and the nonlinear results are partly an artefact of unequal capacity. The
multi-lag linear results (VAR, ARMA) do not have this problem in the cultures; in the Allen
sessions the coarse ARMA R^2 also exceeds the micro one.
Ridge chosen on the last quarter of the training part, R^2 on the last 20% (as closure_test).
Rules: inference (in+out), inference output, model-free lagged output, correlation, random.
Advantage = mean over levels 1..6 (blocks of 2..64 units) of closure(rule) - closure(correlation).
Results: results/closure_ar_cultures.npz, results/closure_ar_allen.npz (closure, R^2 coarse and
micro per rule, metric and level).
Robustness of the random features: python closure_ar.py robust <tag> <metric> <culture...> with the
environment variables CLOSURE_AR_NRF (number of features) and CLOSURE_AR_GAIN (gain); results in
results/closure_ar_robust_<tag>.npz.  Figure: Figures/Supp_closure_ar.png (`plot`).
Run from the folder holding data/ and results/:  python closure_ar.py cultures|allen|summary|plot
"""

import glob
import os
import sys
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from closure_test import pairs5, infer_J_est, FRAC, L
from real_data_rg import load_culture

RULES = ["inference", "inference_out", "lagged_out", "correlation", "random"]
METRICS = ["lin1", "lin5", "lin10", "arma5", "nl1", "nl5"]
P_LONG = 10
N_RF = int(os.environ.get("CLOSURE_AR_NRF", 1024))        # random features (robustness runs override)
GAIN = float(os.environ.get("CLOSURE_AR_GAIN", 1.5))      # gain of the random tanh features
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CULT = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]
KEEP = slice(1, 7)
OUT = {"cultures": "results/closure_ar_cultures.npz", "allen": "results/closure_ar_allen.npz"}


def memberships(B, rule, seed=0):
    """Membership matrices (N x N/2^k), k = 0..L-1, pairing on the training part only."""
    rng = np.random.default_rng(seed)
    S = B[:int(FRAC * len(B))].astype(np.float32)
    M = np.eye(S.shape[1], dtype=np.float32)
    Ms = [M]
    for k in range(L - 1):
        if S.shape[1] < 4:
            break
        J = infer_J_est(S, k) if rule.startswith("inference") else None
        p = pairs5(rule, S, J, rng)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        M = M[:, p[:, 0]] + M[:, p[:, 1]]
        Ms.append(M)
    return Ms


def lagged_design(Z, p):
    """Row r = [Z(t), Z(t-1), ..., Z(t-p+1)] with t = p-1+r; it predicts Z[p+r]."""
    T = len(Z)
    return np.concatenate([Z[p - 1 - k:T - 1 - k] for k in range(p)], 1)


def features(X, kind, n_tr, seed):
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    if kind == "lin":
        return Xs
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((X.shape[1], N_RF)) * GAIN / np.sqrt(X.shape[1])).astype(np.float32)
    b = rng.uniform(-1, 1, N_RF).astype(np.float32)
    return np.concatenate([Xs, np.tanh(Xs @ W + b)], 1)


def _r2(Y, Yhat):
    return float(np.mean(1 - ((Y - Yhat) ** 2).sum(0) / ((Y ** 2).sum(0) + 1e-12)))


def xtx(A, B=None, chunk=8192):
    """A^T B accumulated in float64 by chunks of rows (float32 products lose the small eigenvalues)."""
    B = A if B is None else B
    G = np.zeros((A.shape[1], B.shape[1]))
    for i in range(0, len(A), chunk):
        G += A[i:i + chunk].astype(np.float64).T @ B[i:i + chunk].astype(np.float64)
    return G


class Ridge:
    """Ridge regressions from a fixed design Phi (rows x features): fit on rows [0, n_fit),
    ridge chosen on rows [n_fit, n_tr), refit on [0, n_tr), R^2 on rows [n_tr, end)."""

    def __init__(self, Phi, n_fit, n_tr):
        self.Phi, self.n_fit, self.n_tr = Phi, n_fit, n_tr
        self.cho = {}
        for part, n in [("fit", n_fit), ("tr", n_tr)]:
            G = xtx(Phi[:n]) / n
            tr = np.trace(G) / len(G)
            self.cho[part] = {lam: cho_factor(G + lam * tr * np.eye(len(G))) for lam in RIDGES}

    def cross(self, Y):
        return xtx(self.Phi[:self.n_fit], Y[:self.n_fit]) / self.n_fit, xtx(self.Phi[:self.n_tr], Y[:self.n_tr]) / self.n_tr

    def coef(self, Y, cross=None):
        """Y: rows x targets, centred on the training rows; cross: precomputed cross-moments.
        Ridge chosen on the validation rows, coefficients refitted on all training rows."""
        cf, ct = self.cross(Y) if cross is None else cross
        Pv, Yv = self.Phi[self.n_fit:self.n_tr], Y[self.n_fit:self.n_tr]
        best = max(RIDGES, key=lambda lam: _r2(Yv, Pv @ cho_solve(self.cho["fit"][lam], cf).astype(np.float32)))
        return cho_solve(self.cho["tr"][best], ct).astype(np.float32)

    def r2(self, Y, cross=None):
        Bt = self.coef(Y, cross)
        return _r2(Y[self.n_tr:], self.Phi[self.n_tr:] @ Bt)


def design(Z, met, Tt, seed):
    """Feature matrix for predicting Z(t+1) under metric `met`; row r predicts Z[p + r]."""
    kind, p = parse(met)
    n_tr = Tt - p
    if kind == "arma":
        n1 = Tt - P_LONG
        rid = Ridge(features(lagged_design(Z, P_LONG), "lin", n1, seed), int(0.75 * n1), n1)
        Zc = Z[P_LONG:] - Z[P_LONG:Tt].mean(0)
        E = np.zeros_like(Z)
        E[P_LONG:] = Zc - rid.Phi @ rid.coef(Zc)              # innovations, model fitted on the training part
        return features(np.concatenate([lagged_design(Z, p), lagged_design(E, p)], 1), "lin", n_tr, seed), n_tr
    return features(lagged_design(Z, p), kind, n_tr, seed), n_tr


def parse(met):
    kind = met.rstrip("0123456789")
    return kind, int(met[len(kind):])


def evaluate(B, name, store, metrics=METRICS):
    S0 = B.astype(np.float32)
    T, N = S0.shape
    Tt = int(FRAC * T)
    t0 = time.time()
    Ms = {rule: memberships(B, rule) for rule in RULES}
    print(f"{name}: N {N}, T {T}, partitions [{time.time() - t0:.0f} s]", flush=True)
    for met in metrics:
        kind, p = parse(met)
        t0 = time.time()
        n_tr = Tt - p                                   # training targets: times p .. Tt-1
        n_fit = int(0.75 * n_tr)
        Yc = S0[p:] - S0[p:Tt].mean(0)
        if kind != "nn":                                # micro design shared by all partitions
            micro = Ridge(design(S0, met, Tt, seed=0)[0], n_fit, n_tr)
            cf, ct = micro.cross(Yc)
        for rule in RULES:
            cl, r2c, r2m = (np.full(L, np.nan) for _ in range(3))
            cl[0] = 1.0
            for k, M in enumerate(Ms[rule]):
                if k == 0:
                    continue
                Mn = (M / M.sum(0)).astype(np.float32)
                Y = S0 @ Mn
                if kind == "nn":                        # nested nonlinear: the micro features also see the block variables
                    r2m[k] = Ridge(design(np.concatenate([S0, Y], 1), f"nl{p}", Tt, seed=0)[0], n_fit, n_tr).r2(Yc @ Mn)
                else:
                    r2m[k] = micro.r2(Yc @ Mn, (cf @ Mn, ct @ Mn))
                coarse = Ridge(design(Y, met.replace("nn", "nl"), Tt, seed=1)[0], n_fit, n_tr)
                r2c[k] = coarse.r2(Yc @ Mn)
                cl[k] = r2c[k] / r2m[k] if r2m[k] > 0 else np.nan
            store[f"{name}/{rule}/{met}/closure"], store[f"{name}/{rule}/{met}/r2c"], store[f"{name}/{rule}/{met}/r2m"] = cl, r2c, r2m
        adv = {r: np.nanmean((store[f"{name}/{r}/{met}/closure"] - store[f"{name}/correlation/{met}/closure"])[KEEP]) for r in RULES if r != "correlation"}
        print(f"  {met:5s} R2 micro level 3 {store[f'{name}/correlation/{met}/r2m'][3]:.3f}, closure corr "
              + " ".join(f"{v:.2f}" for v in store[f"{name}/correlation/{met}/closure"][KEEP])
              + " | adv " + " ".join(f"{r[:9]} {v:+.3f}" for r, v in adv.items()) + f"  [{time.time() - t0:.0f} s]", flush=True)


def load(f):
    return dict(np.load(f)) if os.path.exists(f) else {}


def run_cultures():
    store = load(OUT["cultures"])
    for c in CULT:
        if f"{c}/random/{METRICS[-1]}/closure" in store:
            continue
        B_all, _, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        evaluate(B_all[:, :N], c, store)
        np.savez(OUT["cultures"], **store)


def run_allen():
    from invivo_binsize import bin_condition, CACHE
    from allen_rest_vs_task import condition_bins
    store = load(OUT["allen"])
    for f in sorted(glob.glob(f"{CACHE}/*.npz")):
        ses = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        spikes = np.split(d["spikes"], d["index"][:-1])
        t_max = max(s[-1] for s in spikes if len(s))
        T_eq = int(float(d["T_eq_min"]) * 60 / 0.01)
        for cond in ["spontaneous", "natural_movie_three"]:
            name = f"{ses}/{cond}"
            if f"iv/{cond}" not in d or f"{name}/random/{METRICS[-1]}/closure" in store:
                continue
            bins = condition_bins(d[f"iv/{cond}"], d["inv"], t_max, 0.01)[:T_eq]
            B = bin_condition(spikes, bins, 0.01)
            B = B[:, B.std(0) > 0]
            evaluate(B[:, :B.shape[1] // 2 * 2], name, store)
            np.savez(OUT["allen"], **store)


def summary():
    from scipy.stats import wilcoxon, spearmanr
    dr = np.load("results/directedness.npz") if os.path.exists("results/directedness.npz") else None
    for grp, f in OUT.items():
        store = load(f)
        names = sorted({k.split("/" + RULES[0] + "/")[0] for k in store if f"/{RULES[0]}/" in k})
        if not names:
            continue
        x = np.array([float(dr[f"cultures/{n}/lag1"]) for n in names]) if grp == "cultures" and dr is not None else None
        print(f"\n=== {grp}, n = {len(names)}: advantage over correlation (levels 1-6), mean (count > 0, Wilcoxon p)"
              + ("; Spearman with lag-1 directedness" if x is not None else ""))
        for q, lab in [("closure", "closure"), ("r2c", "absolute R^2 of the coarse regression")]:
            print(f"  -- {lab}")
            for met in METRICS:
                if f"{names[0]}/correlation/{met}/closure" not in store:
                    continue
                r2m = np.mean([store[f"{n}/correlation/{met}/r2m"][3] for n in names])
                r2c = np.mean([store[f"{n}/correlation/{met}/r2c"][3] for n in names])
                line = f"  {met:5s} R2 micro {r2m:.3f} coarse(corr) {r2c:.3f} at level 3 |"
                for r in RULES:
                    if r == "correlation":
                        continue
                    a = np.array([np.nanmean((store[f"{n}/{r}/{met}/{q}"] - store[f"{n}/correlation/{met}/{q}"])[KEEP]) for n in names])
                    line += f" {r[:9]} {a.mean():+.4f} ({np.sum(a > 0)}/{len(a)}, p={wilcoxon(a).pvalue:.3f})"
                    if x is not None:
                        line += f" rho {spearmanr(x, a)[0]:+.2f}"
                    line += " |"
                print(line)
        if grp == "allen":
            print("  output rule minus lagged rule per metric:")
            for met in METRICS:
                if f"{names[0]}/correlation/{met}/closure" not in store:
                    continue
                d = np.array([np.nanmean((store[f"{n}/inference_out/{met}/closure"] - store[f"{n}/lagged_out/{met}/closure"])[KEEP]) for n in names])
                print(f"    {met:5s} {d.mean():+.4f} ({np.sum(d > 0)}/{len(d)}, p={wilcoxon(d).pvalue:.3f})")


MCOL = {"lin1": "k", "lin5": "0.55", "lin10": "0.55", "arma5": "#1f77b4", "nl1": "#ff7f0e", "nl5": "#d62728"}
MLS = {"lin1": "-", "lin5": "-", "lin10": "--", "arma5": "-", "nl1": "-", "nl5": "-"}
MLAB = {"lin1": "linear, 1 step (paper)", "lin5": "VAR(5)", "lin10": "VAR(10)", "arma5": "ARMA(5,5)", "nl1": "nonlinear, 1 step", "nl5": "nonlinear, 5 steps"}
DIRECTIONAL = ["d1", "d2", "d15", "d4", "d3"]


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = load(OUT["cultures"])
    dr = np.load("results/directedness.npz")
    x = np.array([float(dr[f"cultures/{c}/lag1"]) for c in CULT])
    lv = np.arange(1, L)
    fig, axs = plt.subplots(1, 4, figsize=(16.5, 3.9))
    ax = axs[0]
    for m in METRICS:
        v = np.mean([st[f"{c}/correlation/{m}/closure"] for c in DIRECTIONAL], 0)
        ax.plot(lv, v[1:], MLS[m], color=MCOL[m], marker="o", ms=3, label=MLAB[m])
    ax.axhline(1, color="0.6", ls=":")
    ax.set_xlabel("RG level (blocks of $2^k$ units)"); ax.set_ylabel("closure, correlation rule")
    ax.set_title("(a) closure of the correlation blocks, 5 directional cultures", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    for m in METRICS:
        v = np.mean([st[f"{c}/inference_out/{m}/closure"] - st[f"{c}/correlation/{m}/closure"] for c in DIRECTIONAL], 0)
        ax.plot(lv, v[1:], MLS[m], color=MCOL[m], marker="o", ms=3, label=MLAB[m])
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("RG level"); ax.set_ylabel("closure, output rule $-$ correlation")
    ax.set_title("(b) advantage of the output rule by level, 5 directional cultures", fontsize=9, loc="left")
    ax = axs[2]
    for m, mk in [("lin1", "o"), ("arma5", "s"), ("nl5", "^")]:
        a = [np.nanmean((st[f"{c}/inference_out/{m}/closure"] - st[f"{c}/correlation/{m}/closure"])[KEEP]) for c in CULT]
        ax.scatter(x, a, marker=mk, s=32, color=MCOL[m], label=MLAB[m])
    ax.axhline(0, color="0.6", ls=":")
    ax.set_xlabel("lag-1 directedness"); ax.set_ylabel("advantage of the output rule (levels 1-6)")
    ax.set_title("(c) per culture", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    ax = axs[3]
    sa = load(OUT["allen"])
    names = sorted({k.split("/correlation/")[0] for k in sa if "/correlation/" in k})
    if names:
        xs = np.arange(len(METRICS))
        for r, lab, col, off in [("inference_out", "output rule", "#8b0000", -0.15), ("lagged_out", "lagged rule", "#2a9d8f", 0.15)]:
            a = np.array([[np.nanmean((sa[f"{n}/{r}/{m}/closure"] - sa[f"{n}/correlation/{m}/closure"])[KEEP]) for m in METRICS] for n in names])
            ax.errorbar(xs + off, a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a)), fmt="o", color=col, capsize=3, label=lab)
        ax.axhline(0, color="0.6", ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([MLAB[m].replace(", ", "\n") for m in METRICS], fontsize=7)
        ax.set_ylabel("advantage over correlation (levels 1-6)")
        ax.set_title(f"(d) Allen, {len(names)} recordings (mean $\\pm$ s.e.m.)", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig("Figures/Supp_closure_ar.png", dpi=170)
    fig.savefig("Figures/Supp_closure_ar.pdf")
    print("saved Figures/Supp_closure_ar.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if what == "cultures":
        run_cultures()
    elif what == "allen":
        run_allen()
    elif what == "plot":
        plot()
        sys.exit()
    elif what == "robust":                              # python closure_ar.py robust <tag> <metric> <culture...>
        tag, met, cults = sys.argv[2], sys.argv[3], sys.argv[4:]
        f = f"results/closure_ar_robust_{tag}.npz"
        store = load(f)
        for c in cults:
            if f"{c}/random/{met}/closure" in store:
                continue
            B_all, _, _ = load_culture(c, 10000)
            N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
            evaluate(B_all[:, :N], c, store, metrics=[met])
            np.savez(f, **store)
        sys.exit()
    elif what == "test":
        B_all, _, _ = load_culture("d1", 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        evaluate(B_all[:, :N], "d1", {}, metrics=["lin1", "arma5", "nl5"])
        sys.exit()
    summary()
