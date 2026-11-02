# Inference-based renormalization group for neural networks with temporal dynamics

Code for the paper *Inference drives Renormalization Group in Neural Networks with temporal dynamics* (Capone, Mancini, Gigante, Del Giudice).

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
| Fig. 1, Fig. 2 | `repro_figures_v2.py` |
| Fig. 3 (N = 512, mean field) | `repro_figure3.py` |
| Fig. 3 (N = 1024, exact ML) | `fig3_ml_rg.py` (grouping), `repro_figure3_ml.py` (full figure) |
| Fig. 4 | `fig4_new.py`, using the results of `directionality_data.py`, `closure_test.py`, `directionality_sweep_spiking.py`, `excess_symmetry_cultures.py`, `invivo_stringer.py`; computes `results/iter_deep.npz` itself |
| Suppl.: symmetry vs surrogate | `real_data_rg.py`, `real_data_surrogate.py`, `null_autocorr_symmetry.py`, `excess_symmetry_sim.py` |
| Suppl.: robustness (bin size, unit selection) | `robustness.py` |
| Suppl.: perturbation transfer | `perturbation_transfer.py` (args: trials, amplitude, network) |
| Suppl.: pairing variants, nMF vs ML | `pairing_tests.py` |

**Exploratory analyses with negative or inconclusive results.** They are kept for completeness and are not used in the paper.
- `invivo_areas.py`, `invivo_thalamus.py`: inter-area grouping and thalamo-cortical directionality in the mice. Not significant against shifted surrogates.
- `drug_prediction.py`: in-silico bicuculline/APV on the culture model vs the observed drug response. Not robust across channel subsets.
- `directionality_sweep_hopfield.py`: directedness sweep on the Hopfield network. The closure saturates for every structured rule, so the sweep does not discriminate between them.

## Conventions

- Binary states S ∈ {0,1}; P(S_i(t+1)=1) = σ(β Σ_j J_ij S_j(t) + h).
- RG step 1 = original units (as in the original MATLAB code).
- Block variables are averages over 2^k units.
- Directedness of a recording = 1 − symmetry index of the lagged covariance D(1).

## Licence

Code: MIT (see `LICENSE`). The data sets keep their own licences (see the table above): CC0 for the cultures, CC BY-NC 4.0 for the Neuropixels recordings.
