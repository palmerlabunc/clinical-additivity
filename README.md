# clinical-additivity

Author: Haeun (Hannah) Hwangbo

This repository contains source codes for "Additivity predicts the efficacy of most approved combination therapies for advanced cancer". Data is deposited in [figshare](10.6084/m9.figshare.21651236).

## Getting Started

Installing all required packages. You would need conda to access the virtual environment.

```bash
git clone https://github.com/palmerlabunc/clinical-additivity.git
conda env create -f env/environment_short.yml

conda activate surv
```

Dependencies: numpy, scipy, pandas, scikit-learn, lifelines, matplotlib, seaborn, snakemake

## Reconstructing the analysis

The data is uploaded into [figshare](10.6084/m9.figshare.21651236). Download all data and place them into a directory of your choice. The original data directory was organized as

```bash
data
├── experimental
├── placebo
├── raw
│   ├── experimental
│   ├── placebo
│   └── trials
├── trials

```

- `raw/` contains files before preprocessing.
- `raw/experimental/` contains drug resposne data from CTRPv2 and Novartis PDXE
- `raw/placebo/` contains digitized KM curves from clinical trials for placebo or best supportive care treatments.
- `raw/trials/` contains digitized KM curves from all combination therapy and constituent therapy clinical trial used for the analysis.

All data files should be placed in their approprite directories as described in `config.yaml` file.

You can reconstruct all tables and figures in the article by running the following code.

```bash
# specify number of cores {N} you want to use
snakemake --cores {N} all
```
