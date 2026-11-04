"""
Replicated directionality sweep (referee control): independent connectivity seeds and
unit subsets per chain strength, so that the ordering of the rules carries an error bar
across networks rather than across Monte-Carlo runs of one network.

Per network (chain c, connectivity seed s): clustered LIF E/I network built with seed s
(spiking_ei_rg.build), 600 s of spontaneous activity, two stimulation runs of the full
network (response matrix R_full, split-half ceiling); on three random 512-unit subsets
(400 E + 112 I, 10 ms bins): lag-1 directedness, closure at RG levels 3 and 4 for the
rules inference (in+out), inference (output), lagged correlation (output), correlation,
random; perturbation transfer at level 3 for every rule and for the micro model.
An optional synaptic delay (ms) gives the variant in which the delay is comparable to
the bin (10 ms), so that directed interactions fall at lag 1.
Results: results/dir_sweep_rep/chain<c>_net<s>_d<delay>.npz (one file per network).

Run from the folder holding sim_cache/ and results/:
    python dir_sweep_replicated.py <net_seed> [delay_ms=5] [chains=0,0.05,0.1,0.2]
    python dir_sweep_replicated.py summary [delay_ms=5]
"""

import glob
import os
import sys
import time
import numpy as np

ARGS = sys.argv[1:]
DELAY = float(ARGS[1]) if len(ARGS) > 1 else 5.0
CHAINS = [float(c) for c in ARGS[2].split(",")] if len(ARGS) > 2 else [0.0, 0.05, 0.1, 0.2]
sys.argv = [sys.argv[0], "40", "0.5", "sweep"]            # perturbation_transfer reads its settings from argv
import spiking_ei_rg as spk
import perturbation_transfer as P
from directionality_sweep_spiking import membership, NET_KW, cache_name
from closure_test import closure
from directedness import directedness

RULES = ["inference", "inference_out", "lagged_out", "correlation", "random"]
SUBSETS = range(3)
SIM = dict(delay_ms=DELAY, delay_jitter=0.5, sigma=0.15)
P.SIM_KW.update(SIM)
OUT = "results/dir_sweep_rep"


def spikes(c, s):
    tag = spk.run_tag(SIM["delay_ms"], SIM["delay_jitter"], SIM["sigma"], NET_KW)
    path = f"sim_cache/{cache_name(c)}_net{s}{tag}_spikes_T600.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["ts"], d["ids"], d["W"], d["clu"]
    W, clu = spk.build(chain=c, seed=s, **NET_KW)
    t0 = time.time()
    ts, ids = spk.simulate(W, 600 * 1000 + spk.SKIP_MS, np.random.default_rng(1000 + s), **SIM)
    print(f"  simulated chain {c} net {s} for 600 s in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(path, ts=ts, ids=ids, W=W, clu=clu)
    return ts, ids, W, clu


def run_network(c, s):
    f = f"{OUT}/chain{c:g}_net{s}_d{DELAY:g}.npz"
    out = dict(np.load(f)) if os.path.exists(f) else {}
    if "done" in out:
        return
    t0 = time.time()
    ts, ids, W, clu = spikes(c, s)
    B = spk.bin_spikes(ts, ids, P.BIN)
    off16 = ~np.eye(spk.NCLU, dtype=bool)
    if "R_full" not in out:
        R1 = P.run_full(W, clu, np.random.default_rng(3 + 100 * s))
        R2 = P.run_full(W, clu, np.random.default_rng(4 + 100 * s))
        R_full = 0.5 * (R1 + R2)
        Ra, Rs = 0.5 * (R_full - R_full.T), 0.5 * (R_full + R_full.T)
        out["R_full"] = R_full
        out["R_asym"] = np.linalg.norm(Ra[off16]) / np.linalg.norm(Rs[off16])
        out["ceiling"] = P.corr(R1, R2, True)
        np.savez(f, **out)
    R_full = out["R_full"]
    self_target = float(np.mean(np.diag(R_full)))
    print(f"=== chain {c} net {s}: response asymmetry {float(out['R_asym']):.2f}, ceiling {float(out['ceiling']):.2f}  [{time.time() - t0:.0f} s]", flush=True)
    for i in SUBSETS:
        if f"sub{i}/random/pert" in out:
            continue
        rng0 = np.random.default_rng(10 * s + i)
        sub = np.sort(np.r_[rng0.choice(spk.NE, 400, replace=False), spk.NE + rng0.choice(spk.NI, 112, replace=False)])
        S0 = B[:, sub].astype(np.float32)
        clu_sub = np.r_[clu, np.full(spk.NI, -1)][sub]
        Wc0 = np.stack([(clu_sub == k).astype(np.float32) for k in range(spk.NCLU)], 1)
        out[f"sub{i}/dir1"] = directedness(S0)[0]
        J0, h0 = P.fit_model(S0, 0)
        out[f"sub{i}/micro/pert"] = P.corr(P.model_response(J0, h0, 1, Wc0, S0, np.random.default_rng(100 * s + i), self_target)[0], R_full, True)
        line = f"  sub {i}: dir1 {float(out[f'sub{i}/dir1']):.3f} micro {float(out[f'sub{i}/micro/pert']):.3f} |"
        for rule in RULES:
            for level in (3, 4):
                M = membership(S0, rule, level, seed=i)
                r2c, r2m = closure(S0, M)
                out[f"sub{i}/{rule}/closure{level}"] = r2c / r2m
                if level == 3:
                    y = S0 @ (M / M.sum(0))
                    J, h = P.fit_model(y, level)
                    R = P.model_response(J, h, 2 ** level, M.T @ Wc0, y, np.random.default_rng(1000 * s + 10 * i), self_target)[0]
                    out[f"sub{i}/{rule}/pert"] = P.corr(R, R_full, True)
            line += f" {rule[:9]} {out[f'sub{i}/{rule}/closure4']:.3f}/{out[f'sub{i}/{rule}/pert']:.3f}"
        np.savez(f, **out)
        print(line + f"  [{time.time() - t0:.0f} s]", flush=True)
    out["done"] = True
    np.savez(f, **out)


def summary(delay):
    print(f"\n=== replicated sweep, delay {delay:g} ms: mean +/- s.d. across networks (subsets averaged within network)")
    for c in [0.0, 0.05, 0.1, 0.2]:
        files = sorted(glob.glob(f"{OUT}/chain{c:g}_net*_d{delay:g}.npz"))
        rows = [dict(np.load(f)) for f in files]
        rows = [r for r in rows if "done" in r]
        if not rows:
            continue
        def agg(key):
            v = [np.mean([r[f"sub{i}/{key}"] for i in SUBSETS]) for r in rows]
            return np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan
        d1, ra = agg("dir1"), (np.mean([r["R_asym"] for r in rows]), np.std([r["R_asym"] for r in rows]))
        print(f"chain {c:4g} (n={len(rows)}): dir1 {d1[0]:.3f}+/-{d1[1]:.3f}  R_asym {ra[0]:.2f}+/-{ra[1]:.2f}  micro pert {agg('micro/pert')[0]:.3f}")
        for rule in RULES:
            c4, pt = agg(f"{rule}/closure4"), agg(f"{rule}/pert")
            dc = [np.mean([r[f"sub{i}/{rule}/closure4"] - r[f"sub{i}/correlation/closure4"] for i in SUBSETS]) for r in rows]
            dp = [np.mean([r[f"sub{i}/{rule}/pert"] - r[f"sub{i}/correlation/pert"] for i in SUBSETS]) for r in rows]
            print(f"   {rule:14s} closure4 {c4[0]:.3f}+/-{c4[1]:.3f} (vs corr {np.mean(dc):+.3f}+/-{np.std(dc, ddof=1) if len(dc) > 1 else np.nan:.3f}, {np.sum(np.array(dc) > 0)}/{len(dc)})"
                  f"  pert {pt[0]:.3f}+/-{pt[1]:.3f} (vs corr {np.mean(dp):+.3f}+/-{np.std(dp, ddof=1) if len(dp) > 1 else np.nan:.3f}, {np.sum(np.array(dp) > 0)}/{len(dp)})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if ARGS and ARGS[0] == "summary":
        summary(DELAY)
    else:
        for c in CHAINS:
            run_network(c, int(ARGS[0]))
        summary(DELAY)
