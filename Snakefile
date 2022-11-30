configfile: "config.yaml"

import pandas as pd
import glob
import os

# DIRECTORIES
RAW_COMBO_DIR = config['dir']['raw_combo_data']
RAW_PLACEBO_DIR = config['dir']['raw_placebo_data']
COMBO_DATA_DIR = config['dir']['combo_data']
PLACEBO_DATA_DIR = config['dir']['placebo_data']
FIG_DIR = config['dir']['figures']
PFS_PRED_DIR = config['dir']['PFS_prediction']

trials_df = pd.read_csv(config['metadata_sheet']['combo'], sep='\t', header=0)
trials_set = set()
pred_list = []
for i in range(trials_df.shape[0]):
    for arm in ['Experimental', 'Control', 'Combination']:
        t = trials_df.at[i, arm]
        trials_set.add(t)
    name_a = trials_df.at[i, 'Experimental']
    name_b = trials_df.at[i, 'Control']
    pred_list.append(f'{name_a}-{name_b}')
trials_list = list(trials_set)

placebo_df = pd.read_csv(config['metadata_sheet']['placebo'], sep='\t', header=0)
placebo_list = list(placebo_df['File prefix'])

# FILE LISTS
ALL_TRIAL_FILES = expand(f"{COMBO_DATA_DIR}/{{trial}}.clean.csv", trial=trials_list)
ALL_PLACEBO_FILES = expand(f"{PLACEBO_DATA_DIR}/{{placebo}}.clean.csv", placebo=placebo_list)
ALL_PRED_FILES =  expand(f"{PFS_PRED_DIR}/{{pred}}_combination_predicted_{{model}}.csv", 
                         pred=pred_list, model=['ind', 'add'])

# OTHER RESULTS
COX_RESULT = config['cox_result']

""""
rule all:
    input:
        expand(f"{COMBO_DATA_DIR}/{{trial}}.clean.csv", trial=trials_list)

rule preprocess_main_trials:
    input:
        f"{RAW_COMBO_DIR}/{{trial}}.csv"
    output:
        f"{COMBO_DATA_DIR}/{{trial}}.clean.csv"
    shell:
        "python src/preprocessing.py {input} --output {output}"
"""

rule preprocess:
    input:
        expand(f"{RAW_COMBO_DIR}/{{trial}}.csv", trial=trials_list),
        expand(f"{RAW_PLACEBO_DIR}/{{placebo}}.csv", placebo=placebo_list)
    output:
        f"{FIG_DIR}/preprocess_sanity_check.png",
        ALL_TRIAL_FILES,
        ALL_PLACEBO_FILES
    script:
        "src/preprocessing.py"

rule find_seeds:
    input:
        ALL_TRIAL_FILES,
        config['metadata_sheet']['combo']
    output:
        config['metadata_sheet']['combo_seed']
    script:
        "src/find_median_sim.py"

rule hsa_additivity_prediction:
    input:
        config['metadata_sheet']['combo_seed'],
        ALL_TRIAL_FILES
    output:
        ALL_PRED_FILES
    script:
        "src/hsa_additivity_model.py"

rule cox_ph_test:
    input:
        ALL_PRED_FILES
    output:
        COX_RESULT
    shell:
        "python src/coxhazard_test.py {output}"

rule figure2:
    input:
        COX_RESULT
    output:
        f'{FIG_DIR}/forest_plot.pdf'
    shell:
        "python src/plotting/forest_plot.py {output}"

rule figure3:
    input:
        COX_RESULT,
        ALL_PRED_FILES,
        ALL_TRIAL_FILES
    output:
        f"{FIG_DIR}/additive_survival_plots.pdf",
        f"{FIG_DIR}/between_survival_plots.pdf",
        f"{FIG_DIR}/hsa_survival_plots.pdf"
    script:
        "src/plotting/plot_survival_curves.py"

rule figure4:
    input:
        COX_RESULT,
        ALL_PRED_FILES,
        ALL_TRIAL_FILES
    output:
        f"{FIG_DIR}/all_combo_qqplot.pdf",
        f"{FIG_DIR}/qqplot_legend.pdf",
        f"{FIG_DIR}/r2_histogram.pdf",
        f"{FIG_DIR}/msd_histogram.pdf"
    shell:
        """
        python src/plotting/all_survival_qqplot.py;
        python src/plotting/r2_msd_histogram.py
        """

# placebo curves
rule figureS1:
    input:
        ALL_PLACEBO_FILES
    output:
        f"{FIG_DIR}/placebo_survival_plots.pdf"
    script:
    

