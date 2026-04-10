# Brain-Wide RRR Encoding Model — IBL Example
 
A **complete** example of fitting a reduced-rank regression (RRR) model to multiple sessions of the IBL brain-wide-map dataset, as used in the paper:
 
**"Rarely categorical, highly separable representations along the cortical hierarchy"**
L. Posani\*, S. Wang\*, S. Muscinelli, L. Paninski$, and S. Fusi$ (2026)
 
\*Equal contribution  $Co-senior authors

## Overview

This subfolder walks through the full pipeline for training and exporting the RRR model:
 
1. Download sessions using the IBL public API
2. Train the RRR model
3. Export the trained model to a DataFrame
4. Examine the trained model (figure scripts)
 
Steps 1–3 can take a long time (~3 days) to complete end-to-end. To save effort, we provide a pre-trained model in `./trained_RRR_model/` — you can skip directly to step 4 if you only want to reproduce the analyses.
 
The notation and parameter names follow the conventions established in the main text and Methods section of the paper. Please refer to the paper for detailed definitions and further context.
 
Related resources:
- **clustering analysis of RRR-estimated single-neuron selectivity** — [realwsq/clustering-analysis](https://github.com/realwsq/clustering-analysis)
- **Decoding and dimensionality analyses** — [lposani/decodanda](https://github.com/lposani/decodanda)

> **Note:** The pre-trained model is large and thus we used [Git LFS](https://git-lfs.com). Install it (`git lfs install`) before cloning, or run `git lfs pull` after cloning to fetch it.

## Getting Started
 
This repository uses [Git LFS](https://git-lfs.com) to track the pre-trained RRR model, which is large. Install Git LFS before cloning:
 
```bash
git lfs install
git clone <repository-url>
```
 
If you have already cloned the repository without Git LFS, run `git lfs pull` to fetch the model file.
 
### Dependencies

**Python >= 3.10** is required. Install all dependencies before running any script:

```bash
# IBL data access
pip install ONE-api ibllib

# Core scientific stack
pip install torch numpy scipy pandas scikit-learn tqdm

# Plotting
pip install matplotlib seaborn
```

## Pipeline
 
### Step 1 — Download sessions from the IBL public API
 
```bash
python step1_IBL_downloaddata.py
```
 
Downloaded data will be saved in `./data/downloaded/`.
 
### Step 2 — Train the RRR model
 
```bash
python step2_train_RRR.py
```
 
This script first pre-processes the downloaded data (see Methods of the paper for details) and constructs two matrices:
 
- `Xall` — trial-aligned, time-varying input variables, shape `[n_trials, n_timesteps, n_vars]`
- `yall` — trial-aligned, time-varying neural activity, shape `[n_trials, n_timesteps, n_neurons]`
 
It then fits an RRR model to these matrices. The optimal rank and L2 regularization penalty are selected via 3-fold cross-validation across trials. The trained model is saved to `./trained_RRR_model/RRRGD/`.
 
### Step 3 — Export the trained model to a DataFrame
 
```bash
python step3_save_trainedRRR_2_df.py
```
 
This produces a DataFrame version of the trained model, saved as `./trained_RRR_model/RRR_selectivity.json`. Each row corresponds to a neuron, with the following columns:
 
- `uuids` — unique neuron identifier
- `eid` — session identifier
- `acronym` — brain region of the neuron
- `RRR_r2` — average 3-fold cross-validated R² of the RRR model
- `RRR_beta` — time-varying coefficients of the neuron, shape `(n_vars+1, n_timesteps)`. The first `n_vars` rows are the input-variable coefficients; the last row is the bias term.
- `RRR_U` — per-neuron loadings of the temporal basis vectors, shape `(n_vars, n_components)`, where `n_components` is the rank of the RRR model.
- `RRR_V` — shared temporal basis vectors (same across all neurons), shape `(n_components, n_timesteps)`.
- `RRR_b` — per-neuron time-varying biases, shape `(n_timesteps,)`. These should be close to zero if the input variables are normalized.
- `null_r2` — average 3-fold cross-validated R² of a baseline null model that is agnostic to input variables (only time in trial).

### Step 4 — Examine the trained model
 
Figure scripts for further analyses of the model coefficients live in `Fig2_globally_structured_brain/`. Run them from within that directory:
 
```bash
cd Fig2_globally_structured_brain/
python <figure_script>.py
```
 
## Contact
 
For questions, please contact [shuqi.wang@epfl.ch](mailto:shuqi.wang@epfl.ch).