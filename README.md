# clinical-additivity

Author: Haeun (Hannah) Hwangbo

## Getting Started
 
Installing all required packages. You would need conda to access the virtual environment.

```bash
git clone https://github.com/palmerlabunc/clinical-additivity.git
conda env create -f env/environment.yml

conda activate surv
```

Dependencies: numpy, scipy, pandas, scikit-learn,  lifelines, matplotlib, seaborn

## Running the model


## Generating figures

To reproduce figures in the manuscript, run the following scripts.

```bash
# main figures
python plot_all_main_figures.py
# suppl figures
python plot_all_suppl_figures.py
```
