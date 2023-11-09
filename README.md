# clinical-additivity

Author: Haeun (Hannah) Hwangbo

This repository contains source codes for "Additivity predicts the efficacy of most approved combination therapies for advanced cancer". Data is deposited in [figshare](doi:10.6084/m9.figshare.22229677).

## Getting Started

Installing all required packages. You would need conda to access the virtual environment.

```bash
git clone https://github.com/palmerlabunc/clinical-additivity.git
conda env create -f env/environment_short.yml

conda activate surv
```

Dependencies: numpy, scipy, pandas, scikit-learn, lifelines, matplotlib, seaborn, snakemake

## Reconstructing the analysis

Download all data and place them into a directory of your choice. The original data directory was organized as

```bash
data
├── all_phase3_trials
├── approved_trials
├── experimental
├── placebo
├── raw
│   ├── all_phase3_trials
│   ├── approved_trials
│   ├── experimental
│   └── placebo

```

- `raw/` contains files before preprocessing.
- `raw/placebo/` contains digitized KM curves from clinical trials for placebo or best supportive care treatments.
- `raw/approved_trials/` contains digitized KM curves from all FDA-approved combination therapies and constituent monotherapies clinical trials used for the analysis.
- `raw/all_phase3_trials/` contains digitized KM curves from all positive and negative phase III trials (2014-2018) of combination therapies and constituent therapies clinical trials used for the analysis.

All data files should be placed in their approprite directories as described in `config.yaml` file.

You can reconstruct all tables and figures in the article by running the following code. We recommend using at least 4 cores because the code utilizes parallel computing. If you have limitied computation power, reducing the `NRUN` in the `src/all_phase3_predictive_power.py` to 100 or 1000 will significantly reduce the run time. 

```bash
# specify number of cores {N} you want to use
snakemake --cores {N} all
```

## Note on Methods

For each patient the combination's progression-free survival (PFS) time is:
- Highest single agent model (a.k.a. independent drug action model): $max(PFS_A + PFS_B)$
- Additivity model: $PFS_A + PFS_B - first scan time$

Note, to ensure that additivity does not result in shorter PFS than the monotherapy, we took the maximum between $PFS_A + PFS_B - first scan time$ and $PFS_A$.
