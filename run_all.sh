#!/usr/bin/env bash
# Reproduce all figures of the paper, in dependency order.  Total run time on a laptop
# (Apple M-series, 11 cores): roughly 5-6 hours, dominated by the spiking simulations
# and the exact-ML inference.  Every step can also be run on its own.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p Figures results sim_cache

[ -d data/lonardoni2017_spont ] || ./download_data.sh

# --- simulations used as input by several scripts ---
$PY make_sequence_data.py                                   # Fig. 3 network, N=1024 and N=512

# --- Figures 1-2: Hopfield network, variance scaling ---
$PY repro_figures_v2.py                                     # ~10 min

# --- Figure 3: sequence network, correlation vs inference RG, stimulation ---
$PY repro_figure3.py                                        # N=512, mean-field inference
$PY fig3_ml_rg.py sim_cache/seq1024_T100000.npy             # N=1024: grouping with exact ML (~15 min)
$PY repro_figure3_ml.py                                     # N=1024: full figure with exact ML (~45 min)

# --- spiking E/I networks (ground truth) ---
$PY spiking_ei_rg.py                                        # no delay: Dale check vs bin size (~10 min)
$PY spiking_ei_rg.py 5 0.5 0.15 3000 1.9                    # 5 ms delays, matched bins (~15 min)
$PY directionality_sweep_spiking.py                         # directedness sweep: closure + perturbation (~60 min)
$PY perturbation_transfer.py 40 0.5 chain                   # perturbation transfer, chain network
$PY perturbation_transfer.py 60 0.5 clustered               # perturbation transfer, clustered network

# --- experimental data ---
$PY excess_symmetry_cultures.py                             # 9 cultures: symmetry vs surrogate, predictability
$PY invivo_stringer.py data/stringer2019_neuropixels        # 3 mice
$PY closure_test.py sim_cache/seq512_T300000.npy            # closure on cultures, mice, spiking, Hopfield
$PY robustness.py data/stringer2019_neuropixels             # bin size x unit selection
$PY directionality_data.py                                  # closure advantage vs directedness

# --- Figure 4 (uses the results above) ---
$PY fig4_new.py

# --- supplementary / controls ---
$PY supp_symmetry.py                                        # symmetry of inferred J vs surrogate (spiking, cultures, mice)
$PY real_data_rg.py
$PY real_data_surrogate.py
$PY null_autocorr_symmetry.py
$PY excess_symmetry_sim.py sim_cache/seq512_T300000.npy
$PY pairing_tests.py sim_cache/seq1024_T100000.npy
