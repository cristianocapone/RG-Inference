"""
Inference-based RG on a simulated spiking E/I network (new test, not in the draft).

Network: LIF neurons with exponential synapses and clustered excitation, as in
Litwin-Kumar & Doiron, Nat. Neurosci. 2012 (p_EE = 0.2, p_EI = p_IE = p_II = 0.5,
J_EE = 0.024, J_EI = -0.045, J_IE = 0.014, J_II = -0.057, rescaled by sqrt(5000/N);
16 E clusters with R_EE = 2.5 and in-cluster weights x1.7), N = 1600 E + 448 I = 2048.
Two structures:
  * "clustered": symmetric cluster structure -> metastable switching between clusters;
  * "chain":     same + extra E->E connections from cluster c to cluster c+1
                 (asymmetric, sequence-like transitions).
Dale's law makes the true W strongly non-symmetric (E columns > 0, I columns < 0).

Synaptic transmission has a delay `delay_ms` (same for all synapses).  With the bin
size equal to the delay (bins aligned to multiples of it) a spike emitted in bin k
reaches its targets in bin k+1, as the one-step kinetic Ising model assumes; without a
delay (delay_ms = 0) most of the effect lands in the same bin and the lagged
covariance D(1) misses it, so the inferred couplings come out small or wrong.

Spikes are cached at 0.5 ms resolution and binned at several bin sizes.  Step 1
("Dale check"): how well does the level-0 inferred J recover the sign/structure of W
as a function of the bin size?  Also tested: inference with an exponentially filtered
presynaptic activity (temporal kernel instead of a single bin, `infer_J_kernel`).  Step 2: RG with three pairing rules (inference,
correlation, random) at the chosen bin sizes, tracking
  * symmetry of the inferred J, (|J_s| - |J_a|) / (|J_s| + |J_a|), Frobenius, no diag
    (0 for iid random couplings, 1 symmetric, -1 antisymmetric), as in Fig. 4;
  * the same quantity for the TRUE W coarse-grained along the same pairing
    (block sums of W), i.e. what the coarse-grained model should look like;
  * E/I purity and cluster purity of the block neurons;
  * variance-scaling exponent.

Run:  env -i /opt/homebrew/bin/python3 spiking_ei_rg.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_figures_v2 import (infer_J, greedy_pairs, synaptic_similarity,
                              variance_scaling, local_exponent)

NE, NI, NCLU = 1600, 448, 16
N = NE + NI
CACHE = "sim_cache"
DT = 0.5                      # integration step (ms)
SKIP_MS = 1000.0              # transient discarded before binning


# --------------------------------------------------------------------------- #
#  Spiking network
# --------------------------------------------------------------------------- #
def build(chain=0.0, REE=2.5, wclu=1.7, seed=0, Nref=5000, pEE=0.2, p=0.5):
    """W[i, j] = weight from j to i."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(Nref / N)
    JEE, JEI, JIE, JII = 0.024 * s, -0.045 * s, 0.014 * s, -0.057 * s
    clu = np.repeat(np.arange(NCLU), NE // NCLU)
    f = 1 / NCLU
    pout = pEE / (f * REE + (1 - f))
    pin = REE * pout
    same = clu[:, None] == clu[None, :]
    P_EE = np.where(same, pin, pout)
    if chain > 0:
        nxt = (clu[None, :] + 1) % NCLU == clu[:, None]     # j in c -> i in c+1
        P_EE = P_EE + chain * pin * nxt
    W = np.zeros((N, N), np.float32)
    W[:NE, :NE] = (rng.random((NE, NE)) < P_EE) * JEE * np.where(same, wclu, 1.0)
    W[:NE, NE:] = (rng.random((NE, NI)) < p) * JEI
    W[NE:, :NE] = (rng.random((NI, NE)) < p) * JIE
    W[NE:, NE:] = (rng.random((NI, NI)) < p) * JII
    np.fill_diagonal(W, 0)
    return W, clu


def simulate(W, T_ms, rng, dt=DT, delay_ms=0.0, delay_jitter=0.0, sigma=0.0,
             muE=(1.1, 1.2), muI=(1.0, 1.05), stim=None):
    """Euler LIF, tau dV/dt = mu - V, synaptic currents with unit-area exp kernels
    (a spike of j shifts V_i by ~W_ij) arriving after an axonal delay that depends on
    the presynaptic neuron, uniform in delay_ms * [1 - jitter, 1 + jitter]; V_th = 1,
    V_reset = 0, 5 ms refractory; `sigma` is the std of an independent white-noise
    current per neuron (per sqrt(ms)); `stim(step)` may return an extra drive vector.
    Returns spike times (in integration steps, int32) and neuron ids (int16)."""
    tau = np.r_[np.full(NE, 15.0), np.full(NI, 10.0)]
    mu = np.r_[rng.uniform(*muE, NE), rng.uniform(*muI, NI)]
    tsE, tsI = 3.0, 2.0
    V = rng.random(N)
    IE, II, ref = np.zeros(N), np.zeros(N), np.zeros(N)
    WE, WI = W[:, :NE], W[:, NE:]
    dly = np.round(delay_ms * (1 + delay_jitter * rng.uniform(-1, 1, N)) / dt).astype(int)
    L = int(dly.max())
    classes = np.unique(dly)
    bufE, bufI = np.zeros((L + 1, N)), np.zeros((L + 1, N))     # delay lines
    ts, ids = [], []
    for step in range(int(T_ms / dt)):
        k = step % (L + 1)
        IE += bufE[k]
        II += bufI[k]
        bufE[k] = 0
        bufI[k] = 0
        ext = None if stim is None else stim(step)
        drive = mu if ext is None else mu + ext
        V += dt * ((drive - V) / tau + IE + II)
        if sigma > 0:
            V += sigma * np.sqrt(dt) * rng.standard_normal(N)
        IE -= dt * IE / tsE
        II -= dt * II / tsI
        V[ref > 0] = 0
        ref -= dt
        sp = np.flatnonzero(V >= 1)
        if sp.size:
            V[sp] = 0
            ref[sp] = 5.0
            for dc in (classes if len(classes) > 1 else [L]):
                spc = sp[dly[sp] == dc] if len(classes) > 1 else sp
                if spc.size == 0:
                    continue
                e, i = spc[spc < NE], spc[spc >= NE] - NE
                ka = (step + dc) % (L + 1)
                if e.size:
                    bufE[ka] += WE[:, e].sum(1) / tsE
                if i.size:
                    bufI[ka] += WI[:, i].sum(1) / tsI
            ts.append(np.full(sp.size, step, np.int32))
            ids.append(sp.astype(np.int16))
    return np.concatenate(ts), np.concatenate(ids)


def bin_spikes(ts, ids, bin_ms, dt=DT, skip_ms=SKIP_MS):
    t_ms = ts * dt - skip_ms
    keep = t_ms >= 0
    b = (t_ms[keep] / bin_ms).astype(np.int64)
    B = np.zeros((b.max() + 1, N), np.uint8)
    B[b, ids[keep]] = 1
    return B


def run_tag(delay_ms=0.0, delay_jitter=0.0, sigma=0.0, net_kw=None):
    tag = ""
    if delay_ms:
        tag += f"_d{delay_ms:g}" + (f"j{delay_jitter:g}" if delay_jitter else "")
    if sigma:
        tag += f"_s{sigma:g}"
    for k, v in sorted((net_kw or {}).items()):
        tag += f"_{k}{v:g}"
    return tag


def get_spikes(name, chain, T_s, delay_ms=0.0, delay_jitter=0.0, sigma=0.0, net_kw=None):
    os.makedirs(CACHE, exist_ok=True)
    tag = run_tag(delay_ms, delay_jitter, sigma, net_kw)
    path = f"{CACHE}/{name}{tag}_spikes_T{T_s}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["ts"], d["ids"], d["W"], d["clu"]
    W, clu = build(chain=chain, **(net_kw or {}))
    t0 = time.time()
    ts, ids = simulate(W, T_s * 1000 + SKIP_MS, np.random.default_rng(1), delay_ms=delay_ms,
                       delay_jitter=delay_jitter, sigma=sigma)
    print(f"  simulated {name} for {T_s} s in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(path, ts=ts, ids=ids, W=W, clu=clu)
    return ts, ids, W, clu


# --------------------------------------------------------------------------- #
#  Analyses
# --------------------------------------------------------------------------- #
def symmetry(J):
    off = ~np.eye(len(J), dtype=bool)
    Js, Ja = 0.5 * (J + J.T), 0.5 * (J - J.T)
    a, b = np.linalg.norm(Js[off]), np.linalg.norm(Ja[off])
    return (a - b) / (a + b)


def infer_J_kernel(B, tau_bins):
    """nMF inversion with an exponential temporal kernel on the presynaptic side:
    S_i(t+1) is predicted from G_j(t) = sum_{tau>=0} exp(-tau/tau_bins) S_j(t-tau)
    (tau_bins = 0 recovers the single-bin model)."""
    from scipy.signal import lfilter
    S = np.asarray(B, np.float32)
    T, n = S.shape
    a = np.exp(-1.0 / tau_bins) if tau_bins > 0 else 0.0
    G = lfilter([1.0], [1.0, -a], S, axis=0).astype(np.float32)
    m = S.mean(0)
    dS = S[1:] - m
    dG = G[:-1] - G.mean(0)
    C = (dG.T @ dG) / (T - 1)
    D = (dS.T @ dG) / (T - 1)
    fp = np.maximum(m * (1 - m), 1e-6)
    J = np.linalg.solve(C.T, D.T).T / fp[:, None]
    np.fill_diagonal(J, 0)
    return J


def dale_check(J, W):
    """How much of W is recovered by the inferred J at level 0."""
    off = ~np.eye(N, dtype=bool)
    out_mean = J.sum(0) / (N - 1)                           # mean outgoing weight
    sE, sI = out_mean[:NE], out_mean[NE:]
    dprime = (sE.mean() - sI.mean()) / np.sqrt(0.5 * (sE.var() + sI.var()))
    cc = np.corrcoef(J[off], W[off])[0, 1]
    syn = W[off] != 0
    sign_acc = np.mean(np.sign(J[off][syn]) == np.sign(W[off][syn]))
    return dict(outE=sE.mean(), outI=sI.mean(), frac_I_neg=np.mean(sI < 0),
                dprime=dprime, corr=cc, sign_acc=sign_acc)


def rg_rules(B, W, clu, rule, n_levels, seed=0):
    rng = np.random.default_rng(seed)
    S = B.astype(np.float32)
    lab_type = np.r_[np.zeros(NE), np.ones(NI)]
    lab_clu = np.r_[clu, np.full(NI, NCLU)]
    M = np.eye(N, dtype=np.float32)               # membership: original x block
    out = dict(sym_inf=[], sym_true=[], ei_purity=[], clu_purity=[], levels=[S])
    for k in range(n_levels):
        J = infer_J(S, k)
        out["sym_inf"].append(symmetry(J))
        out["sym_true"].append(symmetry(M.T @ W @ M))      # block sums of true W
        frac_I = (M.T @ lab_type) / M.sum(0)
        out["ei_purity"].append(np.mean(np.maximum(frac_I, 1 - frac_I)))
        cl = np.stack([(M.T @ (lab_clu == c)) for c in range(NCLU + 1)], 1) / M.sum(0)[:, None]
        out["clu_purity"].append(cl.max(1).mean())
        if k == n_levels - 1:
            break
        if rule == "inference":
            pairs = greedy_pairs(synaptic_similarity(J))
        elif rule == "correlation":
            C = np.nan_to_num(np.corrcoef(S.T))
            np.fill_diagonal(C, -np.inf)
            pairs = greedy_pairs(C)
        else:
            pairs = rng.permutation(S.shape[1]).reshape(-1, 2)
        S = 0.5 * (S[:, pairs[:, 0]] + S[:, pairs[:, 1]])
        M = M[:, pairs[:, 0]] + M[:, pairs[:, 1]]
        out["levels"].append(S)
    K, M2 = variance_scaling(out["levels"])
    out["exp"] = local_exponent(K, M2)[1]
    del out["levels"]
    return out


# --------------------------------------------------------------------------- #
def main(T_s=600, n_levels=10, delay_ms=0.0, delay_jitter=0.0, sigma=0.0, net_kw=None,
         dale_bins=(1, 2, 5, 10), rg_bins=(2, 10), kernel_taus_ms=(2, 5, 10), kernel_bin_ms=1,
         kernel_T_s=300):
    tag = run_tag(delay_ms, delay_jitter, sigma, net_kw)
    nets = [("clustered", 0.0), ("chain", 0.1)]
    rules = ["inference", "correlation", "random"]
    colors = {"inference": "#d62728", "correlation": "#8c8c00", "random": "#4a7fd6"}
    res, dale = {}, {}
    for name, chain in nets:
        print(f"=== {name}", flush=True)
        ts, ids, W, clu = get_spikes(name, chain, T_s, delay_ms, delay_jitter, sigma, net_kw)
        r = np.bincount(ids, minlength=N) / (T_s + SKIP_MS / 1000)
        print(f"  E {r[:NE].mean():.1f} Hz, I {r[NE:].mean():.1f} Hz, symmetry(W) = {symmetry(W):.3f}")
        dale[name] = []
        for bm in dale_bins:
            B = bin_spikes(ts, ids, bm)
            t0 = time.time()
            J = infer_J(B, 0)
            dc = dale_check(J, W)
            dale[name].append(dc)
            print(f"  bin {bm:g} ms (T={B.shape[0]}): out E {dc['outE']:+.4f} I {dc['outI']:+.4f}"
                  f"  frac I<0 {dc['frac_I_neg']:.2f}  d' {dc['dprime']:.2f}"
                  f"  corr(J,W) {dc['corr']:.3f}  sign acc {dc['sign_acc']:.2f}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
            del B, J
        for tk in kernel_taus_ms:
            B = bin_spikes(ts, ids, kernel_bin_ms)[:int(kernel_T_s * 1000 / kernel_bin_ms)]
            t0 = time.time()
            dc = dale_check(infer_J_kernel(B, tk / kernel_bin_ms), W)
            print(f"  kernel tau {tk:g} ms @ {kernel_bin_ms} ms bins (T={B.shape[0]}): out E {dc['outE']:+.4f}"
                  f" I {dc['outI']:+.4f}  frac I<0 {dc['frac_I_neg']:.2f}  d' {dc['dprime']:.2f}"
                  f"  corr(J,W) {dc['corr']:.3f}  sign acc {dc['sign_acc']:.2f}  [{time.time() - t0:.0f} s]",
                  flush=True)
            del B
        res[name] = {"clu": clu}
        for bm in rg_bins:
            B = bin_spikes(ts, ids, bm)
            res[name][bm] = {"B": B[:int(10000 / bm)]}
            for rule in rules:
                t0 = time.time()
                o = rg_rules(B, W, clu, rule, n_levels)
                res[name][bm][rule] = o
                print(f"  bin {bm} ms {rule:12s} [{time.time() - t0:.0f} s]")
                print(f"  {'':15s} sym_inf {np.round(o['sym_inf'], 2)}")
                print(f"  {'':15s} sym_true {np.round(o['sym_true'], 2)}")
                print(f"  {'':15s} E/I purity {np.round(o['ei_purity'], 2)}  cluster purity {np.round(o['clu_purity'], 2)}")
                print(f"  {'':15s} variance exps {np.round(o['exp'], 2)}", flush=True)
            del B

    # ---- Dale check figure ----
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for name, _ in nets:
        d = dale[name]
        axs[0].plot(dale_bins, [x["corr"] for x in d], "o-", label=name)
        axs[1].plot(dale_bins, [x["sign_acc"] for x in d], "o-", label=name)
        axs[2].plot(dale_bins, [x["frac_I_neg"] for x in d], "o-", label=name)
    axs[0].set_ylabel("corr(J inferred, W true)")
    axs[1].set_ylabel("sign accuracy on existing synapses")
    axs[1].axhline(0.5, color="0.6", ls=":")
    axs[2].set_ylabel("fraction of I neurons with J_out < 0")
    for ax in axs:
        ax.set_xlabel("bin size (ms)")
        ax.set_xscale("log")
        ax.set_xticks(dale_bins)
        ax.set_xticklabels([str(b) for b in dale_bins])
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.suptitle(f"run {tag or 'default'}", fontsize=9)
    fig.savefig(f"Figures/Spiking_EI_Dale{tag}.png", dpi=170)

    # ---- RG figure ----
    rows = [(name, bm) for name, _ in nets for bm in rg_bins]
    fig, axs = plt.subplots(len(rows), 4, figsize=(15, 3.2 * len(rows)),
                            gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1]))
    steps = np.arange(1, n_levels + 1)
    for row, (name, bm) in enumerate(rows):
        d = res[name][bm]
        ax = axs[row, 0]
        ax.imshow(d["B"].T, aspect="auto", cmap="gray_r", interpolation="nearest")
        ax.axhline(NE - 0.5, color="r", lw=0.6)
        ax.set_title(f"{name}, {bm} ms bins: raster, 10 s (E above red line)", fontsize=9)
        ax.set_xlabel("time bin")
        ax.set_ylabel("neuron")
        for rule in rules:
            o = d[rule]
            axs[row, 1].plot(steps, o["sym_inf"], "o-", color=colors[rule], ms=3, label=rule)
            axs[row, 1].plot(steps, o["sym_true"], "--", color=colors[rule], lw=1)
            axs[row, 2].plot(steps, o["ei_purity"], "o-", color=colors[rule], ms=3)
            axs[row, 2].plot(steps, o["clu_purity"], "s:", color=colors[rule], ms=3)
            axs[row, 3].plot(steps[1:], o["exp"][:n_levels - 1], "o-", color=colors[rule], ms=3)
        axs[row, 1].axhline(0, color="0.6", ls=":")
        axs[row, 1].set_ylim(-1, 1)
        axs[row, 1].set_title("symmetry (solid: inferred J, dashed: true W)", fontsize=9)
        axs[row, 2].set_title("purity (o: E/I type, s: E cluster)", fontsize=9)
        axs[row, 2].set_ylim(0, 1.02)
        axs[row, 3].set_title("variance scaling exponent", fontsize=9)
        for ax in axs[row, 1:]:
            ax.set_xlabel("RG step")
        axs[row, 1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"Figures/Spiking_EI_RG{tag}.png", dpi=170)
    print(f"saved Figures/Spiking_EI_Dale{tag}.png, Figures/Spiking_EI_RG{tag}.png")


if __name__ == "__main__":
    # usage: spiking_ei_rg.py [delay_ms delay_jitter sigma Nref wclu]
    a = sys.argv[1:]
    if a:
        delay, jitter, sigma = float(a[0]), float(a[1]), float(a[2])
        net_kw = dict(Nref=float(a[3]), wclu=float(a[4])) if len(a) > 4 else None
        main(delay_ms=delay, delay_jitter=jitter, sigma=sigma, net_kw=net_kw,
             dale_bins=(2, delay, 2 * delay), rg_bins=(delay,))
    else:
        main()
