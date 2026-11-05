# Coarse-graining neural dynamics: when correlations suffice and when inferred couplings are needed

Code for the paper *Coarse-graining neural dynamics: when correlations suffice and when inferred couplings are needed* (Capone, Mancini, Gigante, Del Giudice).

The method coarse-grains a recorded network by inferring a kinetic Ising model, pairing the units whose inferred input and output couplings are most similar, and averaging each pair into a block neuron. It then re-infers the model on the block variables and iterates. Pairing units with the same inputs and outputs is the condition under which the merged variable is an exact reduction of the dynamics (lumpability). The scripts compare this rule with:

- correlation-based real-space coarse-graining
- PCA
- random pairing

They do so on Hopfield networks, spiking E/I networks with known connectivity, cultures recorded on a high-density MEA, and Neuropixels recordings in mice.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh          # ~900 MB of public data into data/
python make_sequence_data.py
```

Tested with Python 3.12, numpy 2.4, scipy 1.17, matplotlib, scikit-learn 1.9.

`run_all.sh` reproduces everything in dependency order. It takes roughly 5–6 hours on a laptop. Outputs go to:

- `Figures/` (figures)
- `results/` (`.npz` with the numbers)
- `sim_cache/` (simulations)

## Data

| folder | content | source | licence |
|---|---|---|---|
| `data/lonardoni2017_spont` | spontaneous activity of rat hippocampal cultures, 3Brain 4096-electrode CMOS-MEA, thresholded multi-unit activity | Lonardoni et al. 2017, *PLoS Comput Biol* 13:e1005672; Zenodo 4977727 | CC0 |
| `data/lonardoni2017_drug` | same cultures, basal vs bicuculline / APV | same | CC0 |
| Allen Brain Observatory, Visual Coding Neuropixels | streamed from DANDI 000021 by `allen_final.py` (no download needed) | Allen Institute; DANDI 000021 | CC BY 4.0 |
| `data/stringer2019_neuropixels` | eight-probe Neuropixels, 3 mice, spontaneous period | Stringer et al. 2019, *Science* 364:eaav7893; figshare 10.25378/janelia.7739750 | CC BY-NC 4.0 |

## Code map

**Core**
- `repro_figures_v2.py`:
  - Hopfield simulation
  - mean-field inversion of the kinetic Ising model (`infer_J`)
  - greedy pairing on synaptic similarity
  - variance scaling
  - Figs. 1–2
- `ml_kinetic_ising.py`: exact maximum-likelihood inference of the kinetic Ising model at any RG level (L-BFGS, analytic gradient, exact partition function of the 2^k+1-valued block variables).
- `rg_metrics.py`: symmetry index, lagged covariance, held-out predictability (full, autoregressive, autoregressive + global mode), shifted surrogates.
- `directedness.py`: noise-corrected directedness of a recording (antisymmetric fraction of the lag-1 cross-correlation power, surrogate-subtracted); `directedness_all.py` computes it for every recording and simulated network (`results/directedness.npz`).
- `closure_test.py`:
  - pairing rules (inference in+out / output / input, correlation, random)
  - closure R²(y|y)/R²(y|x) of the coarse-grained dynamics
- `spiking_ei_rg.py`:
  - LIF E/I network with clustered excitation (Litwin-Kumar & Doiron 2012)
  - optional cluster-to-cluster chain
  - axonal delays, noise
  - binning; Dale check of the inferred couplings vs bin size
  - RG with the three rules

**Figures**

| paper | script |
|---|---|
| Fig. 1 (concept, Nat Comm version) | `fig1_concept.py` |
| Fig. 2 and Suppl. Fig. 1 (Hopfield variance scaling) | `repro_figures_v2.py` |
| Fig. 3 (N = 512, mean field) | `repro_figure3.py` |
| Fig. 3 (N = 1024, exact ML) | `repro_figure3_ml.py` (simulation, inference, stimulation; ~45 min), then `fig3_publication.py` (figure); `fig3_ml_rg.py` (grouping only) |
| Fig. 3 controls (model-free lagged grouping; one-shot pairing on level-0 ML couplings) | `fig3_lagged_test.py`, `fig3_oneshot_test.py` |
| Fig. 4 | `fig4_new.py`, using the results of `directionality_data.py`, `closure_test.py`, `directionality_sweep_spiking.py`, `excess_symmetry_cultures.py`, `invivo_stringer.py`; computes `results/iter_deep.npz` itself |
| Suppl.: symmetry vs surrogate | `supp_symmetry.py` (figure); `real_data_rg.py`, `real_data_surrogate.py`, `null_autocorr_symmetry.py`, `excess_symmetry_sim.py` |
| Suppl.: robustness (bin size, unit selection) | `robustness.py` |
| Fig. 5, partial observation (simulation, cultures, mice) | `fig_partial_observation.py`, using `partial_observation.py`, `partial_observation_cultures.py`, `allen_rest_vs_task.py` |
| Fig. 5d, Allen Brain Observatory with the final pipeline (streamed, ~1 GB/session, 32 sessions) | `allen_final.py` (analysis), `fig_allen_final.py` (stand-alone figure) |
| Fig. 5e, rat CA1 epochs and sleep states (DANDI 000044, streamed, ~0.1 GB/session) | `hc11_sequences.py` |
| Suppl.: closure of non-grouping reductions (PCA, reduced-rank regression, NMF) at matched dimension | `dimred_comparison.py` |
| Cover-letter figure | `cover_figure.py` |
| Earlier Allen analyses (first pipeline, kept for reference) | `supp_allen.py`, using `allen_rest_vs_task.py` and `allen_density.py` |
| Suppl.: in vivo single probes | `invivo_local.py` |
| Suppl.: perturbation transfer | `perturbation_transfer.py` (args: trials, amplitude, network) |
| Suppl.: pairing variants, nMF vs ML | `pairing_tests.py` |
| Fig. 4c,d: replicated spiking sweep (5 networks per chain, 3 subsets) | `dir_sweep_replicated.py <seed> [delay_ms] [chains]`; `summary` aggregates `results/dir_sweep_rep/` |
| Suppl.: stimulation of 2a alone, full-network reference | `fig3_fair.py` (`plot` re-draws from `results/fig3_fair.npz`) |
| Suppl.: advantage per RG step, LOO/permutation, 5-fold CV, Allen unit count | `a2_stats.py` (reads `results/closure_cv.npz` from `closure_cv.py`) |
| Suppl.: scaling observables (variance, silence, autocorrelation time, spectrum) under the rules | `rg_observables.py` (`plot` re-draws) |
| Suppl.: block compactness, spatial (Kadanoff) rule, bursts, global mode in the cultures | `culture_mechanism.py` (`plot` re-draws) |
| Suppl.: in vivo at 2/5/10 ms, nMF vs ridge vs ML couplings | `invivo_binsize.py stringer|cache|allen|summary` (Allen spike times cached in `data/allen_cache/`) |

**Exploratory analyses with negative or inconclusive results.** They are kept for completeness and are not used in the paper.
- `invivo_areas.py`, `invivo_thalamus.py`: inter-area grouping and thalamo-cortical directionality in the mice. Not significant against shifted surrogates.
- `drug_prediction.py`: in-silico bicuculline/APV on the culture model vs the observed drug response. Not robust across channel subsets.
- `dandi_closure.py`: DANDI 002020 MaxOne cultures. At 10 ms the one-step predictability is about zero (sparse activity, slow drifts), so closure is undefined; not used.
- `directionality_sweep_hopfield.py`: directedness sweep on the Hopfield network. The closure saturates for every structured rule, so the sweep does not discriminate between them.
- `directional_sequence_spiking.py`, `directional_metrics_spiking.py`: free-running reduced models of the spiking networks (sequence of cluster activations) and lag-1 metrics sensitive to directional structure (specific predictability, multi-step closure) along the chain sweep. The sequence test depends strongly on how the reduced model's gain is set (tuned: inference best only at the strongest chain, not monotone; as fitted: no reduced model reproduces the sequence); the lag-1 metrics do not grow with the chain. Not used in the paper; the directional metric in the simulations is the response to stimulation (Fig. 4d).
- `dir_sweep_replicated.py <seed> 10`: synaptic delay equal to the bin (10 ms). Does not raise the lag-1 directedness of the chain networks (0.04-0.05 at chain 0 and 0.2); the full-network response becomes unreliable at chain 0 (split-half 0.21).
- `kernel_closure_spiking.py`: kinetic Ising model with an exponential presynaptic kernel (30, 100, 300 ms) on the chain networks, couplings by mean-field inversion of the filtered statistics, closure evaluated with the same kernel. The advantage of inference over correlation does not grow with the chain for any kernel, and the closure ratio becomes ill-conditioned at long kernels (values above 1); the model-free lagged rule does as well or better. Negative, not used in the paper.

## Conventions

- Binary states S ∈ {0,1}; P(S_i(t+1)=1) = σ(β Σ_j J_ij S_j(t) + h).
- RG step 1 = original units (as in the original MATLAB code).
- Block variables are averages over 2^k units.
- Directedness of a recording = 1 − symmetry index of the lagged covariance D(1).

## Licence

Code: MIT (see `LICENSE`). The data sets keep their own licences (see the table above): CC0 for the cultures, CC BY-NC 4.0 for the Neuropixels recordings.
