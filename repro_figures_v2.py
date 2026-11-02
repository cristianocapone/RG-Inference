"""
Reproduction (v2) of Figures 1 and 2 of
"Inference drives Renormalization Group in Neural Networks with temporal dynamics".

v1 (repro_figures.py) assumed the original analysis code was missing and used a
sharp-attractor regime in which the network stays locked in one pattern; the RG
then sees no transitions and the inferred matrices are noise.  The original MATLAB
code is in ../Inference (RG_2Attractors_BlockRule_release_PART_I.m, ../RHT_pm.m,
../nMF.m) and v2 follows its conventions:

  * states S in {0,1}, P(S_i(t+1)=1) = sigmoid(beta * sum_j J0_ij S_j(t) + h), h = -0.5,
    synchronous update, start from S = 0;
  * Hebb J0 = (xi - a)^T (xi - a) / N with a = eta, J0_ii = 0;
  * beta just below the retrieval bifurcation, where the network keeps switching
    between attractors (beta ~ 17 for N=2048, 3 patterns, eta=1/4, the value used
    in the release code);
  * neuron indices randomly permuted before the RG (structure must be recovered);
  * mean-field kinetic-Ising inversion J = diag(1/f'(Hbar)) D(1) C^-1, sigmoid f for
    the binary level and f_inf for block neurons (Appendix of the paper);
  * pairing by similarity of pre/post synaptic vectors V = [J, J^T] (V V^T, cosine),
    greedy from the most similar pair; block neuron = average of the pair.

RG step numbering follows the MATLAB code: step 1 = original neurons (K = 1).

Run:  env -i /opt/homebrew/bin/python3 repro_figures_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H_EXT = -0.5


# --------------------------------------------------------------------------- #
#  Network and dynamics
# --------------------------------------------------------------------------- #
def make_patterns(N, P, eta):
    size = int(round(N * eta))
    xi = np.zeros((P, N))
    for mu in range(P):
        xi[mu, mu * size:(mu + 1) * size] = 1.0
    return xi


def simulate(xi, eta, beta, T, rng, h=H_EXT, burn=500):
    """Glauber dynamics with the low-rank Hebb matrix (J0 never built explicitly)."""
    P, N = xi.shape
    X = xi - eta
    diag = (X ** 2).sum(0) / N
    S = np.zeros(N)
    out = np.empty((T, N), np.uint8)
    for t in range(T + burn):
        H = beta * (X.T @ (X @ S) / N - diag * S) + h
        S = (rng.random(N) < 1.0 / (1.0 + np.exp(-H))).astype(float)
        if t >= burn:
            out[t - burn] = S
    return out


def tune_beta(xi, eta, T=8000, n=41, seed=0):
    """beta just below the retrieval bifurcation that maximises the number of
    transitions between attractors (the paper's 'retrieval phase with transitions');
    exactly at the susceptibility peak the network is critically slowed down."""
    betas = np.linspace(3 / eta, 7 / eta, n)
    switches = []
    for b in betas:
        S = simulate(xi, eta, b, T, np.random.default_rng(seed), burn=300)
        ov = np.stack([np.convolve(S[:, xi[m] == 1].mean(1), np.ones(10) / 10, "same")
                       for m in range(len(xi))], 1)
        dom = np.where(ov.max(1) > 0.55, ov.argmax(1), -1)
        dom = dom[dom >= 0]
        switches.append(np.sum(np.diff(dom) != 0))
    return betas[int(np.argmax(switches))]


# --------------------------------------------------------------------------- #
#  Inference (nMF inversion of the kinetic Ising model)
# --------------------------------------------------------------------------- #
def _f_infty_table():
    H = np.linspace(-80, 80, 400001)
    with np.errstate(all="ignore"):
        f = 1 - 1 / H + 1 / np.expm1(H)
        fp = 1 / H ** 2 + 1 / (2 - 2 * np.cosh(H))
    small = np.abs(H) < 1e-2
    f[small] = 0.5 + H[small] / 12
    fp[small] = 1 / 12
    return H, f, fp


_HG, _FG, _FPG = _f_infty_table()


def infer_J(S, level):
    S = np.asarray(S, float)
    T, N = S.shape
    m = S.mean(0)
    d = S - m
    C = d.T @ d / T
    D = d[1:].T @ d[:-1] / (T - 1)
    mc = np.clip(m, 1e-4, 1 - 1e-4)
    if level == 0:
        fp = mc * (1 - mc)
    else:
        fp = np.interp(np.interp(mc, _FG, _HG), _HG, _FPG)
    J = np.linalg.solve(C.T, D.T).T / fp[:, None]
    np.fill_diagonal(J, 0)
    return J


# --------------------------------------------------------------------------- #
#  RG
# --------------------------------------------------------------------------- #
def greedy_pairs(G):
    """Pair units greedily from the largest to the smallest entry of G."""
    N = G.shape[0]
    iu = np.triu_indices(N, 1)
    order = np.argsort(-G[iu], kind="stable")
    taken = np.zeros(N, bool)
    pairs = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        if not (taken[i] or taken[j]):
            taken[i] = taken[j] = True
            pairs.append((i, j))
            if len(pairs) == N // 2:
                break
    return np.array(pairs)


def synaptic_similarity(J):
    V = np.hstack([J, J.T])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V @ V.T


def rg(S0, n_levels, rng, labels=None):
    """Returns per-level activities, inferred J, mean original position, and
    block purity (fraction of the dominant pattern label inside each block)."""
    perm = rng.permutation(S0.shape[1])
    S = S0[:, perm].astype(np.float32)
    pos = perm.astype(float)
    comp = None
    if labels is not None:
        L = labels.max() + 1
        comp = np.eye(L)[labels[perm]]
    levels, Js, poss, purity = [S], [], [pos], [1.0]
    for k in range(n_levels - 1):
        J = infer_J(S, k)
        Js.append(J)
        pairs = greedy_pairs(synaptic_similarity(J))
        S = 0.5 * (S[:, pairs[:, 0]] + S[:, pairs[:, 1]])
        pos = 0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])
        levels.append(S)
        poss.append(pos)
        if comp is not None:
            comp = 0.5 * (comp[pairs[:, 0]] + comp[pairs[:, 1]])
            purity.append(comp.max(1).mean())
    if S.shape[1] > 1:
        Js.append(infer_J(S, n_levels - 1))
    return levels, Js, poss, purity


def variance_scaling(levels):
    K = 2.0 ** np.arange(len(levels))
    M2 = np.array([np.var(L * k, axis=0).mean() for L, k in zip(levels, K)])
    return K, M2


def local_exponent(K, M2):
    return K[1:], np.diff(np.log(M2)) / np.diff(np.log(K))


def pattern_labels(xi):
    lab = np.full(xi.shape[1], xi.shape[0])
    for mu in range(xi.shape[0]):
        lab[xi[mu] == 1] = mu
    return lab


def sorted_J(J, pos):
    o = np.argsort(pos)
    return J[np.ix_(o, o)]


# =========================================================================== #
#  FIGURE 1
# =========================================================================== #
def figure1(T=100_000, n_runs=5):
    print("=== Figure 1 ===")
    N, P, eta = 2048, 3, 0.25
    xi = make_patterns(N, P, eta)
    beta = tune_beta(xi, eta)
    print(f"  beta* = {beta:.2f}")
    labels = pattern_labels(xi)
    n_levels = 11                                  # K = 1 ... 1024
    slopes = []
    for r in range(n_runs):
        S = simulate(xi, eta, beta, T, np.random.default_rng(100 + r))
        lev, _, poss, pur = rg(S, n_levels, np.random.default_rng(200 + r), labels)
        K, M2 = variance_scaling(lev)
        slopes.append(local_exponent(K, M2)[1])
        if r == 0:
            S_raw, lev0, poss0, K0, M20 = S, lev, poss, K, M2
            print("  block purity per level:", np.round(pur, 3))
        print(f"  run {r}: local exponents {np.round(slopes[-1], 2)}")
    slopes = np.array(slopes)

    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.0, 1.0], hspace=0.6,
                          wspace=0.32, left=0.10, right=0.97, top=0.95, bottom=0.08)
    axa = fig.add_subplot(gs[0, :])
    axa.axis("off")
    _schematic(axa)
    fig.text(0.02, 0.96, "(a)", fontweight="bold")

    win = slice(0, 3000)
    axb1 = fig.add_subplot(gs[1, 0])
    axb1.imshow(S_raw[win].T, aspect="auto", cmap="gray_r", interpolation="nearest")
    axb1.set_title("original dynamics", fontsize=9)
    axb1.set_xlabel("time")
    axb1.set_ylabel("neuron")
    axb2 = fig.add_subplot(gs[1, 1])
    lv = 3                                         # 256 block neurons, as in the paper
    o = np.argsort(poss0[lv])
    axb2.imshow(lev0[lv][win][:, o].T, aspect="auto", cmap="gray_r",
                interpolation="nearest")
    axb2.set_title("coarse grained dynamics", fontsize=9)
    axb2.set_xlabel("time")
    axb2.set_ylabel("block neuron")
    fig.text(0.02, 0.64, "(b)", fontweight="bold")

    axc1 = fig.add_subplot(gs[2, 0])
    a = np.polyfit(np.log(K0), np.log(M20), 1)[0]
    axc1.loglog(K0, M20, "ko--", ms=3, lw=1)
    axc1.loglog(K0, M20[0] * K0, "b-", lw=1)
    axc1.loglog(K0, M20[0] * K0 ** 2, "r-", lw=1)
    axc1.text(0.05, 0.90, r"$\alpha\sim2$", color="r", transform=axc1.transAxes)
    axc1.text(0.05, 0.78, rf"data, $\alpha\sim{a:.1f}$", transform=axc1.transAxes)
    axc1.text(0.05, 0.66, r"$\alpha\sim1$", color="b", transform=axc1.transAxes)
    axc1.set_xlabel("block neuron size")
    axc1.set_ylabel(r"variance $M_2(K)$")
    axc2 = fig.add_subplot(gs[2, 1])
    steps = np.arange(1, slopes.shape[1] + 1)
    m, s = slopes.mean(0), slopes.std(0)
    axc2.fill_between(steps, m - s, m + s, color="0.75")
    axc2.plot(steps, m, "k-", lw=1.2)
    axc2.set_xlabel("RG step")
    axc2.set_ylabel("variance exponent")
    fig.text(0.02, 0.32, "(c)", fontweight="bold")
    fig.savefig("Figures/Figure1_repro_v2.png", dpi=200)
    plt.close(fig)
    print(f"  global alpha = {a:.2f}; saved Figures/Figure1_repro_v2.png")


def _schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    rs = np.random.default_rng(3)

    def net(cx, cy, cols, r=0.5):
        n = len(cols)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = cx + r * np.cos(ang), cy + r * np.sin(ang)
        for i in range(n):
            for j in range(i + 1, n):
                same = cols[i] == cols[j] and cols[i] != "k"
                if same or rs.random() < 0.15:
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-",
                            color=cols[i] if same else "0.6", lw=0.8, zorder=1)
        ax.scatter(xs, ys, s=60, c=cols, zorder=2)

    net(1.3, 1.2, ["r", "r", "r", "r", "k", "b", "b", "b", "b", "k", "k", "k"])
    ax.text(1.3, 2.2, "original network", ha="center", fontsize=9)
    for k, y in enumerate(np.linspace(0.3, 2.0, 6)[::-1]):
        c = "r" if k == 0 else ("b" if k == 1 else "k")
        ax.scatter([3.9, 4.3], [y, y], s=40, c=c)
    ax.text(4.1, 2.25, "synaptic\nsimilarity", ha="center", fontsize=8)
    for k, y in enumerate(np.linspace(0.3, 2.0, 6)[::-1]):
        c = "r" if k == 0 else ("b" if k == 1 else "k")
        ax.scatter(5.6, y, s=60, c=c)
    ax.text(5.6, 2.25, "block\nneuron", ha="center", fontsize=8)
    for x0 in (2.2, 4.6, 6.2):
        ax.annotate("", xy=(x0 + 0.6, 1.2), xytext=(x0, 1.2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="0.4"))
    net(8.3, 1.2, ["r", "r", "k", "b", "b", "k"])
    ax.text(8.3, 2.2, "coarse grained network", ha="center", fontsize=9)


# =========================================================================== #
#  FIGURE 2
# =========================================================================== #
def figure2(T=100_000):
    print("=== Figure 2 ===")
    N = 1024
    curves = []
    for eta in [1 / 2, 1 / 4, 1 / 8, 1 / 16]:
        P = 2
        xi = make_patterns(N, P, eta)
        beta = tune_beta(xi, eta)
        S = simulate(xi, eta, beta, T, np.random.default_rng(1))
        lev, _, _, pur = rg(S, 10, np.random.default_rng(2), pattern_labels(xi))
        K, M2 = variance_scaling(lev)
        Kc, sl = local_exponent(K, M2)
        curves.append((int(N * eta), Kc, sl))
        print(f"  pattern {int(N * eta):4d}: beta*={beta:.1f}  exps {np.round(sl, 2)}"
              f"  purity {np.round(pur, 2)}")

    eta, P = 1 / 4, 3
    xi = make_patterns(N, P, eta)
    beta = tune_beta(xi, eta)
    S = simulate(xi, eta, beta, T, np.random.default_rng(3))
    lev, Js, poss, pur = rg(S, 6, np.random.default_rng(4), pattern_labels(xi))
    print(f"  panel B/C: beta*={beta:.1f}, purity {np.round(pur, 3)}")

    fig = plt.figure(figsize=(6.4, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.0, 0.8], hspace=0.55,
                          wspace=0.5, left=0.12, right=0.93, top=0.96, bottom=0.07)
    axA = fig.add_subplot(gs[0, :])
    cols = ["#0000cc", "#4d4de6", "#9999f0", "#ccccf7"]
    for (size, Kc, sl), c in zip(curves, cols):
        axA.plot(Kc, sl, "o-", color=c, ms=4, lw=2)
        axA.text(0.04, 0.5 - 0.09 * cols.index(c), str(size), color=c,
                 transform=axA.transAxes, fontsize=10)
    axA.set_xscale("log", base=2)
    axA.set_xticks(curves[0][1])
    axA.set_xticklabels([str(int(k)) for k in curves[0][1]])
    axA.set_xlabel("Cluster Size")
    axA.set_ylabel("Variance Scaling Exponent")
    axA.annotate("", xy=(48, 0.3), xytext=(8, 0.3),
                 arrowprops=dict(arrowstyle="-|>", lw=1.5))
    axA.text(10, 0.45, "RG", fontsize=12)
    fig.text(0.02, 0.96, "(a)", fontweight="bold")

    vmax = 0.2
    for c, lv in enumerate([0, 2, 4]):
        ax = fig.add_subplot(gs[1, c])
        im = ax.imshow(sorted_J(Js[lv], poss[lv]), cmap="viridis", vmin=-vmax,
                       vmax=vmax, interpolation="nearest")
        ax.set_title(f"RG step {lv + 1} " + r"$J_{ij}$", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("neuron i")
        if c == 0:
            ax.set_ylabel("neuron j")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-vmax, 0, vmax])

        axh = fig.add_subplot(gs[2, c])
        w = Js[lv][~np.eye(len(Js[lv]), dtype=bool)]
        n, e = np.histogram(w, bins=np.linspace(-0.12, 0.25, 75), density=True)
        ctr = 0.5 * (e[1:] + e[:-1])
        axh.bar(ctr, n, width=e[1] - e[0],
                color=plt.cm.viridis((ctr + vmax) / (2 * vmax)))
        axh.set_xlabel(r"$J_{ij}$")
        if c == 0:
            axh.set_ylabel("density")
    fig.text(0.02, 0.60, "(b)", fontweight="bold")
    fig.text(0.02, 0.30, "(c)", fontweight="bold")
    fig.savefig("Figures/Figure2_repro_v2.png", dpi=200)
    plt.close(fig)
    print("  saved Figures/Figure2_repro_v2.png")


if __name__ == "__main__":
    figure1()
    figure2()
    print("done.")
