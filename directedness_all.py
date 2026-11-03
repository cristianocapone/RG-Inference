"""
Noise-corrected directedness (directedness.py) of every recording and simulated network
used in the paper, for lag windows of 1 bin and of 1-30 bins (10-300 ms), together with
the raw index 1 - symmetry(D(1)).  The window is validated on the spiking networks,
where the directedness of the propagation is known (asymmetry of the response to
stimulation, results/dir_sweep_spiking.npz).
Results: results/directedness.npz.  Run from the folder holding data/ and sim_cache/.
"""

import numpy as np

import spiking_ei_rg as spk
from directedness import directedness
from real_data_rg import load_culture
from invivo_stringer import bin_mouse, MICE

TAG = "d5j0.5_s0.15_Nref3000_wclu1.9"
SIMS = {"0.0": "clustered", "0.05": "chain0.05", "0.1": "chain", "0.2": "chain0.2"}
WINDOWS = {"lag1": (1,), "lag1-30": tuple(range(1, 31))}


def datasets():
    for c in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d15"]:
        B_all, rates, _ = load_culture(c, 10000)
        N = min(512, B_all.shape[0] // 100, B_all.shape[1]) // 2 * 2
        yield f"cultures/{c}", B_all[:, :N]
    for mouse, tstart in MICE.items():
        B, *_ = bin_mouse(f"data/stringer2019_neuropixels/spks{mouse}_Feb18.mat", tstart)
        yield f"mice/{mouse}", B
    for chain, f in SIMS.items():
        d = np.load(f"sim_cache/{f}_{TAG}_spikes_T600.npz")
        B = spk.bin_spikes(d["ts"], d["ids"], 10.0)
        rng = np.random.default_rng(0)                       # same 512-unit subset as closure_test.py
        idx = np.sort(np.r_[rng.choice(spk.NE, 400, replace=False), spk.NE + rng.choice(spk.NI, 112, replace=False)])
        yield f"spiking/{chain}", B[:, idx]


def main():
    store = {}
    for key, B in datasets():
        for w, lags in WINDOWS.items():
            store[f"{key}/{w}"], store[f"{key}/raw"] = directedness(B, lags)
        print(key, " ".join(f"{w} {store[f'{key}/{w}']:.3f}" for w in WINDOWS), f"raw {store[f'{key}/raw']:.3f}", flush=True)
    np.savez("results/directedness.npz", **store)


if __name__ == "__main__":
    main()
