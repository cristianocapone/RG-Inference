"""
Reproduction of Figure 3: asymmetric (sequence) Hopfield network, correlation-based
vs inference-based RG, and stimulation of the coarse-grained networks.

Network as in ../RG_inference/Dynamics_generator_sequences.m: N = 512, T = 3e5,
5 orthogonal patterns of 64 units, J = J0 + alpha*J_T with alpha = 0.2, h = -2, beta = 55, states {0,1}.
Pattern 2 is split into 2a -> 3 and 2b -> 4, so the network cycles 1 -> 2 -> (3|4) -> 5 -> 1.

(The paper text says 1024 neurons; at N = 1024, T = 1e5 the inferred 2a->3 / 2b->4
couplings are below the inference noise and no pairing rule can keep 2a and 2b apart.)

Coarse-grained dynamics (step 5 = 32 block neurons) are sampled from the level-k model
used for the inference, P(S_i = n/2^k) ~ exp(b * H_i * n/2^k), H = J S + h, with the
inferred J and h and the gain b chosen to minimise the CV of the pattern-1
inter-activation intervals.
Stimulation: +I on the blocks of sub-population 2a, -I on those of 2b.

Run:  env -i /opt/homebrew/bin/python3 repro_figure3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity, _FG, _HG

N, ETA, ALPHA, BETA, H_EXT = 512, 1 / 8, 0.2, 55.0, -2.0
LEVEL = 4                     # RG step 5 in the MATLAB numbering -> 32 block neurons
THR = 0.6
# composition labels: 0..4 = patterns 1..5 except pattern 2 -> 1 = 2a, 5 = 2b; 6 = background
NAMES = ["1", "2a", "3", "4", "5", "2b", "bg"]


def seq_network():
    s = int(N * ETA)
    xi = np.zeros((5, N))
    for m in range(5):
        xi[m, m * s:(m + 1) * s] = 1
    sa, sb = np.zeros(N), np.zeros(N)
    sa[s:s + s // 2] = 1
    sb[s + s // 2:2 * s] = 1
    X = xi - ETA
    J0 = X.T @ X / N
    JT = (np.outer(xi[1], xi[0]) + np.outer(xi[4], xi[3]) + np.outer(xi[4], xi[2])
          + np.outer(xi[0], xi[4])) / N
    JT += 2 * (np.outer(xi[2], sa) + np.outer(xi[3], sb)) / N
    J = J0 + ALPHA * JT
    np.fill_diagonal(J, 0)
    lab = np.full(N, 6)
    for m in range(5):
        lab[xi[m] == 1] = m
    lab[sb == 1] = 5
    return J, xi, lab


def glauber(J, h, gain, T, rng, S0=None, I=None):
    n = len(J)
    S = np.zeros(n) if S0 is None else S0.astype(float).copy()
    Jg, hg = gain * J, gain * h
    out = np.empty((T, n), np.uint8)
    for t in range(T):
        H = Jg @ S + hg
        if I is not None:
            H = H + I
        S = (rng.random(n) < 1 / (1 + np.exp(-H))).astype(float)
        out[t] = S
    return out


def block_dynamics(J, h, gain, T, rng, I=None, nlev=2 ** LEVEL):
    n = len(J)
    v = np.arange(nlev + 1) / nlev
    S = np.zeros(n)
    out = np.empty((T, n), np.float32)
    for t in range(T):
        H = gain * (J @ S + h)
        if I is not None:
            H = H + I
        L = np.outer(H, v)
        L -= L.max(1, keepdims=True)
        p = np.exp(L)
        c = np.cumsum(p / p.sum(1, keepdims=True), 1)
        S = v[np.minimum((rng.random(n)[:, None] > c).sum(1), nlev)]
        out[t] = S
    return out


def rg_rule(S0, lab, rule, rng):
    perm = rng.permutation(S0.shape[1])
    S = S0[:, perm].astype(np.float32)
    comp = np.eye(7)[lab[perm]]
    for k in range(LEVEL):
        if rule == "inference":
            G = synaptic_similarity(infer_J(S, k))
        else:
            G = np.nan_to_num(np.corrcoef(S.T))
        np.fill_diagonal(G, -np.inf)
        p = greedy_pairs(G)
        S = 0.5 * (S[:, p[:, 0]] + S[:, p[:, 1]])
        comp = 0.5 * (comp[p[:, 0]] + comp[p[:, 1]])
    J = infer_J(S, LEVEL)
    m = np.clip(S.mean(0), 1e-4, 1 - 1e-4)
    Hbar = np.interp(m, _FG, _HG)
    h = Hbar - J @ m
    return S, J, h, comp


def block_order(comp):
    """Sort blocks by dominant label (1, 2a, 2b, 3, 4, 5, bg), then by purity."""
    rank = np.array([0, 1, 3, 4, 5, 2, 6])
    dom = comp.argmax(1)
    return np.lexsort((-comp.max(1), rank[dom]))


def overlaps(S, comp):
    """Mean activity of the blocks dominated by patterns 1, 3, 4 (and 2a/2b)."""
    dom = comp.argmax(1)
    return {nm: S[:, dom == i].mean(1) if np.any(dom == i) else np.zeros(len(S))
            for i, nm in enumerate(NAMES)}


def activation_wave(x, w=5):
    return uniform_filter1d(x.astype(float), w) > THR


def cycles(ov):
    """Intervals between consecutive pattern-1 activations and, for each, whether
    pattern 3 / pattern 4 crossed threshold inside it."""
    a1 = activation_wave(ov["1"]).astype(int)
    on = np.flatnonzero(np.diff(a1) == 1) + 1
    off = np.flatnonzero(np.diff(a1) == -1) + 1
    res = []
    for e in off:
        nxt = on[on > e]
        if len(nxt) == 0:
            break
        s, t = e, nxt[0]
        res.append((t - s, activation_wave(ov["3"][s:t]).any(),
                    activation_wave(ov["4"][s:t]).any()))
    return np.array(res, float).reshape(-1, 3)


def tune_gain(J, h, comp, rng, gains):
    best, best_cv = gains[0], np.inf
    for g in gains:
        S = block_dynamics(J, h, g, 6000, rng)
        c = cycles(overlaps(S, comp))
        if len(c) < 20:
            continue
        cv = c[:, 0].std() / c[:, 0].mean()
        if cv < best_cv:
            best, best_cv = g, cv
    return best, best_cv


def stim_vector(comp, I):
    return I * (comp[:, 1] - comp[:, 5])        # + on 2a content, - on 2b content


def main(T=300_000):
    rng = np.random.default_rng(0)
    J, xi, lab = seq_network()
    # glauber() multiplies h by the gain, so pass h / beta to get beta*J S + h
    S0 = glauber(J, np.full(N, H_EXT / BETA), BETA, T + 500, rng, S0=xi[1])[500:]

    res = {}
    for rule in ["correlation", "inference"]:
        S, Jk, hk, comp = rg_rule(S0, lab, rule, np.random.default_rng(1))
        o = block_order(comp)
        Jk, hk, comp = Jk[np.ix_(o, o)], hk[o], comp[o]
        t = comp[:, 1] + comp[:, 5]
        sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
        print(f"{rule}: purity {comp.max(1).mean():.3f}, 2a/2b separation per block {np.round(sep, 2)}")
        g, cv = tune_gain(Jk, hk, comp, np.random.default_rng(2), np.linspace(0.6, 1.2, 7))
        print(f"  gain* = {g:.2f}, CV = {cv:.2f}")
        Is = np.linspace(0, 8.0, 21)
        p3, p4, rasters = [], [], {}
        for I in Is:
            Sg = block_dynamics(Jk, hk, g, 20000, np.random.default_rng(3),
                                I=stim_vector(comp, I))
            c = cycles(overlaps(Sg, comp))
            p3.append(c[:, 1].mean() if len(c) else np.nan)
            p4.append(c[:, 2].mean() if len(c) else np.nan)
            rasters[round(I, 2)] = Sg[:800]
        print("  P(3):", np.round(p3, 2))
        print("  P(4):", np.round(p4, 2))
        res[rule] = dict(J=Jk, comp=comp, Is=Is, p3=np.array(p3), p4=np.array(p4),
                         rasters=rasters)

    plot(J, S0, res)


def plot(J, S0, res, lo=0.0, hi=6.4):
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 1.2], hspace=0.5, wspace=0.3,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(J * 1e4, cmap="viridis", vmin=-2, vmax=16)
    ax.set_title("couplings", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"$J_{ij}\times10^{-4}$")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(S0[:800].T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_title("original activity", fontsize=10)
    ax.set_xticks([]); ax.set_ylabel("neuron")
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.5, 0.5, "stim: +I on sub A (2a)\n      -I on sub B (2b)\n\n2a -> pattern 3\n2b -> pattern 4",
            ha="center", va="center", fontsize=10, family="monospace")

    for r, rule in enumerate(["correlation", "inference"]):
        d = res[rule]
        ax = fig.add_subplot(gs[r + 1, 0])
        im = ax.imshow(d["J"], cmap="viridis", vmin=-1, vmax=4)
        ax.set_title(f"{rule[:4]} RG step 5", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

        sub = gs[r + 1, 1].subgridspec(2, 1, hspace=0.1)
        for q, I in enumerate([lo, hi]):
            ax = fig.add_subplot(sub[q])
            R = d["rasters"][round(I, 2)]
            ov = overlaps(R, d["comp"])
            ax.imshow(R.T, aspect="auto", cmap="gray_r", interpolation="nearest")
            for nm, col in [("3", "#f08080"), ("4", "#e6d36a")]:
                act = activation_wave(ov[nm])
                ax.fill_between(np.arange(len(act)), -0.5, len(d["J"]) - 0.5, where=act,
                                color=col, alpha=0.45, step="mid", lw=0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(f"SI={I}", fontsize=8)
            if q == 0:
                ax.set_title(f"coarse-grained network activity from {rule} RG", fontsize=10)

        ax = fig.add_subplot(gs[r + 1, 2])
        ax.plot(d["Is"], d["p3"], "o-", color="#e06070", ms=3, label="Pattern 3")
        ax.plot(d["Is"], d["p4"], "o-", color="#a09020", ms=3, label="Pattern 4")
        for I, c in [(lo, "purple"), (hi, "navy")]:
            ax.axvline(I, color=c, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("transition probability")
        ax.set_xlabel("stimulus intensity")
        ax.legend(fontsize=8, frameon=False)
    fig.savefig("Figures/Figure3_repro.png", dpi=170)
    plt.close(fig)
    print("saved Figures/Figure3_repro.png")


if __name__ == "__main__":
    main()
