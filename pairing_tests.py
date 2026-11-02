"""
Can a less noise-sensitive pairing (or a better inference) rescue the inference-based
RG where the greedy cosine similarity on the raw nMF J fails?

Test beds
  A. Sequence network of Fig. 3 at N = 1024, T = 1e5 (the paper's stated size), where
     the greedy pairing on raw J mixes the 2a / 2b sub-populations.
  B. Clustered spiking E/I network (10 ms bins), where correlation pairing beat
     inference pairing on E/I and cluster purity.

Pairing rules compared (all greedy from the most similar pair, except kmeans):
  raw      cosine similarity of [J, J^T]                      (current)
  lowrank  same, on the rank-r SVD truncation of J (denoised)
  kmeans   k-means on [J, J^T] (as in the original MATLAB), then greedy within cluster
  corr     Pearson correlation of the activities
For test A we also refit J by maximising the exact kinetic-Ising log-likelihood
(gradient ascent from the nMF solution, 'ml') and repeat the pairings on it.

Metrics: fraction of mixed pairs (different labels), 2a/2b mixing, and after 4 RG
levels the per-block purity.

Run:  env -i /opt/homebrew/bin/python3 pairing_tests.py <path to S0seq.npy>
"""

import sys
import time
import numpy as np
from sklearn.cluster import KMeans

from repro_figures_v2 import infer_J, greedy_pairs, synaptic_similarity, _FG, _HG


# --------------------------------------------------------------------------- #
#  Pairings
# --------------------------------------------------------------------------- #
def lowrank(J, r):
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def lowrank_mp(J, margin=1.1):
    """Keep only the singular values above the Marchenko-Pastur (quarter-circle) noise
    edge: for an N x N iid-noise matrix the median singular value is ~0.81 sigma sqrt(N)
    and the edge is 2 sigma sqrt(N), i.e. edge ~ 2.47 * median."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    r = max(1, int(np.sum(s > margin * 2.47 * np.median(s))))
    return (U[:, :r] * s[:r]) @ Vt[:r], r


def pairs_kmeans(J, k, seed=0):
    F = np.hstack([J, J.T])
    idx = KMeans(k, n_init=10, random_state=seed).fit_predict(F)
    G = synaptic_similarity(J)
    same = idx[:, None] == idx[None, :]
    G = np.where(same, G, -np.inf)            # pair only within a cluster ...
    np.fill_diagonal(G, -np.inf)
    N = len(J)
    taken = np.zeros(N, bool)
    pairs = []
    iu = np.triu_indices(N, 1)
    order = np.argsort(-G[iu], kind="stable")
    for o in order:
        i, j = iu[0][o], iu[1][o]
        if G[i, j] == -np.inf:
            break
        if not (taken[i] or taken[j]):
            taken[i] = taken[j] = True
            pairs.append((i, j))
    left = np.flatnonzero(~taken)              # ... leftovers (odd clusters) paired across
    if len(left) > 1:
        Gl = synaptic_similarity(J)[np.ix_(left, left)]
        np.fill_diagonal(Gl, -np.inf)
        for a, b in greedy_pairs(Gl):
            pairs.append((left[a], left[b]))
    return np.array(pairs)


def pairs_corr(S):
    C = np.nan_to_num(np.corrcoef(S.T))
    np.fill_diagonal(C, -np.inf)
    return greedy_pairs(C)


def make_pairs(rule, J, S, r=8, k=8):
    if rule == "raw":
        return greedy_pairs(synaptic_similarity(J))
    if rule == "lowrank":
        return greedy_pairs(synaptic_similarity(lowrank(J, r)))
    if rule == "lowrank_mp":
        return greedy_pairs(synaptic_similarity(lowrank_mp(J)[0]))
    if rule == "kmeans":
        return pairs_kmeans(J, k)
    if rule == "corr":
        return pairs_corr(S)
    raise ValueError(rule)


# --------------------------------------------------------------------------- #
#  Exact-likelihood refinement of the level-0 kinetic Ising model
# --------------------------------------------------------------------------- #
def ml_refine(S, J0, n_iter=40, verbose=True):
    """Maximise the exact kinetic-Ising log-likelihood
        L_i = (1/T) sum_t [S_i(t+1) H_i(t) - log(1 + e^{H_i(t)})],  H = J S(t) + h,
    independently for every neuron i (the problem is separable), starting from the nMF
    solution.  Step: gradient preconditioned with C^{-1} / <f'>_i (the nMF Hessian
    approximation), with a per-neuron backtracking line search.  Returns J, h."""
    S = np.asarray(S, np.float32)
    T, N = S.shape
    Sp, Sn = S[:-1], S[1:]
    m = S.mean(0)
    d = Sp - m
    C = (d.T @ d) / (T - 1) + 1e-6 * np.eye(N, dtype=np.float32)
    Cinv = np.linalg.inv(C).astype(np.float32)
    J = J0.astype(np.float32).copy()
    h = (np.log(m / (1 - m)) - J @ m).astype(np.float32)

    def loglik(J, h):
        H = Sp @ J.T + h
        return np.mean(Sn * H - np.logaddexp(0, H), 0), H

    L, H = loglik(J, h)
    eta = np.ones(N, np.float32)
    for it in range(n_iter):
        P = 1 / (1 + np.exp(-H))
        R = Sn - P
        fp = np.maximum((P * (1 - P)).mean(0), 1e-3)
        dJ = ((R.T @ Sp) / (T - 1)) @ Cinv / fp[:, None]
        dh = R.mean(0) / fp
        for _ in range(6):                                   # backtracking
            Jc = J + eta[:, None] * dJ
            np.fill_diagonal(Jc, 0)
            hc = h + eta * dh
            Lc, Hc = loglik(Jc, hc)
            ok = Lc >= L
            if ok.all():
                break
            eta[~ok] *= 0.5
        acc = Lc >= L
        J[acc], h[acc], L[acc], H[:, acc] = Jc[acc], hc[acc], Lc[acc], Hc[:, acc]
        eta[acc] = np.minimum(eta[acc] * 1.3, 1.0)
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"    ml iter {it:2d}  <logL> = {L.mean():.5f}  accepted {acc.mean():.2f}  <eta> {eta.mean():.3f}", flush=True)
    return J, h


# --------------------------------------------------------------------------- #
#  Test A: sequence network
# --------------------------------------------------------------------------- #
def labels_seq(N=1024, eta=1 / 8):
    s = int(N * eta)
    lab = np.full(N, 6)
    for m in range(5):
        lab[m * s:(m + 1) * s] = m
    lab[s + s // 2:2 * s] = 5                   # 2b
    return lab, ["1", "2a", "3", "4", "5", "2b", "bg"]


def pair_stats(pairs, lab):
    a, b = lab[pairs[:, 0]], lab[pairs[:, 1]]
    mixed = np.mean(a != b)
    ab = np.mean(((a == 1) & (b == 5)) | ((a == 5) & (b == 1)))
    n2 = np.sum(((a == 1) | (a == 5)) & ((b == 1) | (b == 5)))
    return mixed, ab * len(pairs) / max(n2, 1)


def rg_levels(S0, lab, rule, n_levels, r=8, k=8, J_first=None):
    S = S0.astype(np.float32)
    comp = np.eye(lab.max() + 1)[lab]
    for lev in range(n_levels):
        J = J_first if (lev == 0 and J_first is not None) else infer_J(S, lev)
        pairs = make_pairs(rule, J, S, r, k)
        S = 0.5 * (S[:, pairs[:, 0]] + S[:, pairs[:, 1]])
        comp = 0.5 * (comp[pairs[:, 0]] + comp[pairs[:, 1]])
    return comp


def test_seq(path, n_levels=4):
    print("=== Test A: sequence network N=1024 (Fig. 3 size)")
    S0 = np.load(path)
    T, N = S0.shape
    lab, names = labels_seq(N)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    S0, lab = S0[:, perm], lab[perm]
    t0 = time.time()
    J = infer_J(S0, 0)
    print(f"  nMF J in {time.time() - t0:.0f} s")
    blk = lambda Jm, a, b: Jm[np.ix_(lab == a, lab == b)].mean()
    print(f"  nMF   3<-2a {blk(J, 2, 1):+.4f}  3<-2b {blk(J, 2, 5):+.4f}  4<-2a {blk(J, 3, 1):+.4f}  4<-2b {blk(J, 3, 5):+.4f}")
    Jml, _ = ml_refine(S0, J)
    print(f"  ML    3<-2a {blk(Jml, 2, 1):+.4f}  3<-2b {blk(Jml, 2, 5):+.4f}  4<-2a {blk(Jml, 3, 1):+.4f}  4<-2b {blk(Jml, 3, 5):+.4f}")
    print(f"  MP-adaptive rank of nMF J: {lowrank_mp(J)[1]}, of ML J: {lowrank_mp(Jml)[1]}")
    for jname, Jm in [("nMF", J), ("ML", Jml)]:
        for rule, kw in [("raw", {}), ("lowrank", dict(r=8)), ("lowrank_mp", {}),
                         ("kmeans", dict(k=8)), ("corr", {})]:
            if rule in ("corr", "kmeans") and jname == "ML":
                continue
            t0 = time.time()
            p = make_pairs(rule, Jm, S0, **kw)
            mixed, ab = pair_stats(p, lab)
            comp = rg_levels(S0, lab, rule, n_levels, J_first=Jm, **kw)
            t = comp[:, 1] + comp[:, 5]
            sep = np.abs(comp[t > 0.5, 1] - comp[t > 0.5, 5]) / t[t > 0.5]
            print(f"  {jname:3s} {rule:8s} {str(kw):12s} level-0 mixed pairs {mixed:.3f}, 2a/2b mixed {ab:.2f};"
                  f"  after {n_levels} levels purity {comp.max(1).mean():.3f}, 2a/2b sep {np.round(sep, 2)}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)


# --------------------------------------------------------------------------- #
#  Test B: spiking E/I network (10 ms bins, cached by the old spiking_ei_rg.py)
# --------------------------------------------------------------------------- #
def test_spiking(path="sim_cache/clustered_T1000.npz", n_levels=4, NE=1600, NI=448, NCLU=16):
    print("=== Test B: clustered spiking E/I network, 10 ms bins")
    d = np.load(path)
    B, W, clu = d["B"], d["W"], d["clu"]
    lab = np.r_[clu, np.full(NI, NCLU)]
    typ = np.r_[np.zeros(NE, int), np.ones(NI, int)]
    J = infer_J(B, 0)
    for rule, kw in [("raw", {}), ("lowrank", dict(r=16)), ("lowrank", dict(r=32)),
                     ("kmeans", dict(k=17)), ("kmeans", dict(k=24)), ("corr", {})]:
        t0 = time.time()
        comp_c = rg_levels(B, lab, rule, n_levels, J_first=J, **kw)
        comp_t = rg_levels(B, typ, rule, n_levels, J_first=J, **kw)
        print(f"  {rule:8s} {str(kw):12s} after {n_levels} levels: cluster purity {comp_c.max(1).mean():.3f},"
              f" E/I purity {comp_t.max(1).mean():.3f}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    test_seq(sys.argv[1])
    if "--spiking" in sys.argv:
        test_spiking()
