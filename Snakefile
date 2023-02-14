configfile: "config.yaml"

import pandas as pd
import glob
import os

# DIRECTORIES
RAW_COMBO_DIR = config['dir']['raw_combo_data']
RAW_PLACEBO_DIR = config['dir']['raw_placebo_data']
RAW_PHASE3_DIR = config['dir']['raw_all_phase3_data']
COMBO_DATA_DIR = config['dir']['combo_data']
PLACEBO_DATA_DIR = config['dir']['placebo_data']
PHASE3_DATA_DIR = config['dir']['all_phase3_data']
EXPERIMENTAL_DATA_DIR = config['dir']['experimental_data']
FIG_DIR = config['dir']['figures']
TABLE_DIR = config['dir']['tables']
PFS_PRED_DIR = config['dir']['PFS_prediction']

#FIXME make this into a function later
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

"""
all_phase3_df = pd.read_csv(config['metadata_sheet']['all_phase3'], sep='\t', header=0)
all_phase3_set = set()
pred_list = []
for i in range(all_phase3_df.shape[0]):
    for arm in ['Experimental', 'Control', 'Combination']:
        t = all_phase3_df.at[i, arm]
        all_phase3_set.add(t)
    name_a = all_phase3_df.at[i, 'Experimental']
    name_b = all_phase3_df.at[i, 'Control']
    pred_list.append(f'{name_a}-{name_b}')
all_phase3_list = list(all_phase3_set)
"""

placebo_df = pd.read_csv(config['metadata_sheet']['placebo'], sep='\t', header=0)
placebo_list = list(placebo_df['File prefix'])

# FILE LISTS
ALL_TRIAL_FILES = expand(f"{COMBO_DATA_DIR}/{{trial}}.clean.csv", trial=trials_list)
ALL_PLACEBO_FILES = expand(f"{PLACEBO_DATA_DIR}/{{placebo}}.clean.csv", placebo=placebo_list)
#ALL_PHASE3_FILES = expand(f"{PHASE3_DATA_DIR}/{{placebo}}.clean.csv", placebo=all_phase3_list)
ALL_PRED_FILES =  expand(f"{PFS_PRED_DIR}/{{pred}}_combination_predicted_{{model}}.csv", 
                         pred=pred_list, model=['ind', 'add'])

# OTHER RESULTS
COX_RESULT = config['cox_result']


rule all:
    input:
        f'{FIG_DIR}/forest_plot.pdf',
        f"{FIG_DIR}/additive_survival_plots.pdf",
        f"{FIG_DIR}/between_survival_plots.pdf",
        f"{FIG_DIR}/hsa_survival_plots.pdf",
        f"{FIG_DIR}/all_combo_qqplot.pdf",
        f"{FIG_DIR}/r2_histogram.pdf",
        f"{FIG_DIR}/msd_histogram.pdf",
        f"{FIG_DIR}/ici_boxplot.pdf",
        f"{FIG_DIR}/angiogenesis_boxplot.pdf",
        f"{FIG_DIR}/monotherapy_approval_boxplot.pdf",
        f"{FIG_DIR}/HRmedian_boxplot.pdf",
        f"{FIG_DIR}/placebo_survival_plots.pdf",
        f"{FIG_DIR}/relative_doses.pdf",
        f"{FIG_DIR}/suppl_additive_survival_plots.pdf",
        f"{FIG_DIR}/suppl_between_hsa_survival_plots.pdf",
        f'{FIG_DIR}/HR_combo_control_scatterplot.pdf',
        f'{TABLE_DIR}/HR_predicted_vs_control.csv',
        f'{FIG_DIR}/explain_HSA_additive_difference.pdf',
        f'{TABLE_DIR}/added_benefit_hsa_add_syn.csv',
        f'{FIG_DIR}/hsa_additivity_sigma.pdf',        
        f'{FIG_DIR}/CRC_cetuximab_5FU_BestAvgResponse_corr.pdf',
        f'{FIG_DIR}/Dabrafenib_Trametinib_AUC_corr.pdf',
        f'{FIG_DIR}/CTRPv2_corr_distributions.pdf',
        f'{TABLE_DIR}/experimental_correlation_report.csv'


rule preprocess:
    input:
        expand(f"{RAW_COMBO_DIR}/{{trial}}.csv", trial=trials_list),
        expand(f"{RAW_PLACEBO_DIR}/{{placebo}}.csv", placebo=placebo_list),
    output:
        f"{FIG_DIR}/preprocess_sanity_check.png",
        ALL_TRIAL_FILES,
        ALL_PLACEBO_FILES,
    conda:
        "env/environment_short.yml"
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

rule forest_plot:
    input:
        COX_RESULT
    output:
        f'{FIG_DIR}/forest_plot.pdf'
    shell:
        "python src/plotting/forest_plot.py {output}"

rule main_survival_plots:
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

rule qqplots:
    input:
        COX_RESULT,
        ALL_PRED_FILES,
        ALL_TRIAL_FILES
    output:
        f"{FIG_DIR}/all_combo_qqplot.pdf",
        f"{FIG_DIR}/qqplot_legend.pdf"
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/all_survival_qqplot.py"

rule performance_histogram:
    input:
        COX_RESULT,
        ALL_PRED_FILES,
        ALL_TRIAL_FILES
    output:
        f"{FIG_DIR}/r2_histogram.pdf",
        f"{FIG_DIR}/msd_histogram.pdf",
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/r2_msd_histogram.py"

rule subgroup_boxplots:
    input:
        COX_RESULT
    output:    
        f"{FIG_DIR}/ici_boxplot.pdf",
        f"{FIG_DIR}/angiogenesis_boxplot.pdf",
        f"{FIG_DIR}/monotherapy_approval_boxplot.pdf",
        f"{FIG_DIR}/HRmedian_boxplot.pdf"
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/subgroup_boxplots.py"

rule placebo_survival_plots:
    input:
        ALL_PLACEBO_FILES
    output:
        f"{FIG_DIR}/placebo_survival_plots.pdf"
    script:
        "src/plotting/plot_placebo.py"

rule relative_dose_plot:
    input:
        config['relative_doses']
    output:
        f"{FIG_DIR}/relative_doses.pdf"
    script:
        "src/plotting/plot_doses_difference.py"

rule suppl_survival_plots:
    input:
        ALL_TRIAL_FILES,
        ALL_PRED_FILES,
        COX_RESULT
    output:
        f"{FIG_DIR}/suppl_additive_survival_plots.pdf",
        f"{FIG_DIR}/suppl_between_hsa_survival_plots.pdf"
    script:
        "src/plotting/plot_survival_curves_suppl.py"

rule predict_success:
    input:
        ALL_TRIAL_FILES,
        ALL_PRED_FILES,
        COX_RESULT
    output:
        f'{FIG_DIR}/HR_combo_control_scatterplot.pdf',
        f'{TABLE_DIR}/HR_predicted_vs_control.csv'
    script:
        "src/predict_success.py"

rule HSA_additive_diff:
    input:
        ALL_TRIAL_FILES,
        ALL_PRED_FILES,
        COX_RESULT
    output:
        f'{FIG_DIR}/explain_HSA_additive_difference.pdf',
        f'{TABLE_DIR}/added_benefit_hsa_add_syn.csv',
        f'{FIG_DIR}/hsa_additivity_sigma.pdf'
    shell:
        """
        python src/lognormal_examples.py;
        python src/hsa_add_diff.py;
        """

rule experimental_correlation:
    input:
        f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_CCL.csv',
        f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_CCL.csv',
        f'{EXPERIMENTAL_DATA_DIR}/Recalculated_CTRP_12_21_2018.txt',
        f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_clincal_active_drug_pairwise_corr.csv',
        f'{EXPERIMENTAL_DATA_DIR}/Gao2015_suppl_table.xlsx'
    output:
        f'{FIG_DIR}/CRC_cetuximab_5FU_BestAvgResponse_corr.pdf',
        f'{FIG_DIR}/Dabrafenib_Trametinib_AUC_corr.pdf',
        f'{FIG_DIR}/CTRPv2_corr_distributions.pdf',
        f'{TABLE_DIR}/experimental_correlation_report.csv'
    script:
        "src/experimental_correlation.py"
