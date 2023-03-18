configfile: "config.yaml"

import pandas as pd
import glob
import os
import numpy as np

datasets = ['approved', 'all_phase3']

# DIRECTORIES
EXPERIMENTAL_DATA_DIR = config['experimental_dir']
FIG_DIR = config['fig_dir']
TABLE_DIR = config['table_dir']

def get_trial_list(dataset):
    trials_df = pd.read_csv(config[dataset]['metadata_sheet'], 
                            sep='\t', header=0)
    if dataset == 'placebo':
        trial_list = list(placebo_df['File prefix'])
    else:
        trials_set = set()
        for i in range(trials_df.shape[0]):
            for arm in ['Experimental', 'Control', 'Combination']:  
                t = trials_df.at[i, arm]
                trials_set.add(t)
        trial_list = list(trials_set)
    return trial_list

def get_raw_files(dataset):
    dir = config[dataset]['raw_dir']
    trial_list = get_trial_list(dataset)
    file_list = [f"{dir}/{trial}.csv" for trial in trial_list]
    return file_list

def get_trial_files(dataset):
    dir = config[dataset]['data_dir']
    trial_list = get_trial_list(dataset)
    file_list = [f"{dir}/{trial}.clean.csv" for trial in trial_list]
    return file_list

def get_pred_list(dataset):
    trials_df = pd.read_csv(config[dataset]['metadata_sheet'], 
                            sep='\t', header=0)
    pred_list = []
    for i in range(trials_df.shape[0]):
        name_a = trials_df.at[i, 'Experimental']
        name_b = trials_df.at[i, 'Control']
        pred_list.append(f'{name_a}-{name_b}')
    return pred_list

def get_pred_files(dataset):
    pred_list = get_pred_list(dataset)
    pred_dir = config[dataset]['pred_dir']
    return 

placebo_df = pd.read_csv(config['placebo']['metadata_sheet'], sep='\t', header=0)
placebo_list = list(placebo_df['File prefix'])

# FILE LISTS
ALL_APPROVED_TRIALS = expand("{output_file}", output_file=get_trial_files("approved"))
ALL_PLACEBO_TRIALS = expand("{output_file}", output_file=get_trial_files("placebo"))
ALL_PHASE3_TRIALS = expand("{output_file}", output_file=get_trial_files("all_phase3"))
ALL_APPROVED_PRED_FILES =  expand(f"{config['approved']['pred_dir']}/{{pred}}_combination_predicted_{{model}}.csv", 
                                  pred=get_pred_list("approved"), model=['ind', 'add'])
ALL_PHASE3_PRED_FILES =  expand(f"{config['all_phase3']['pred_dir']}/{{pred}}_combination_predicted_{{model}}.csv", 
                                  pred=get_pred_list("all_phase3"), model=['ind', 'add'])
ALL_PRED_FILES = list(set(ALL_APPROVED_PRED_FILES) | set(ALL_PHASE3_PRED_FILES))


rule all:
    input:
        f"{config['approved']['fig_dir']}/forest_plot.pdf",
        f"{config['approved']['fig_dir']}/additive_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/between_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/hsa_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/all_combo_qqplot.pdf",
        f"{config['approved']['fig_dir']}/r2_histogram.pdf",
        f"{config['approved']['fig_dir']}/msd_histogram.pdf",
        f"{config['approved']['fig_dir']}/ici_boxplot.pdf",
        f"{config['approved']['fig_dir']}/angiogenesis_boxplot.pdf",
        f"{config['approved']['fig_dir']}/monotherapy_approval_boxplot.pdf",
        f"{config['approved']['fig_dir']}/HRmedian_boxplot.pdf",
        f"{config['placebo']['fig_dir']}/placebo_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/relative_doses.pdf",
        f"{config['approved']['fig_dir']}/suppl_additive_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/suppl_between_hsa_survival_plots.pdf",
        f"{config['fig_dir']}/HR_combo_control_scatterplot.pdf",
        f"{config['table_dir']}/HR_predicted_vs_control.csv",
        f"{config['approved']['fig_dir']}/explain_HSA_additive_difference.pdf",
        f"{config['approved']['table_dir']}/added_benefit_hsa_add_syn.csv",
        f"{config['approved']['fig_dir']}/hsa_additivity_sigma.pdf",        
        f"{config['fig_dir']}/CRC_cetuximab_5FU_BestAvgResponse_corr.pdf",
        f"{config['fig_dir']}/Dabrafenib_Trametinib_AUC_corr.pdf",
        f"{config['fig_dir']}/CTRPv2_corr_distributions.pdf",
        f"{config['table_dir']}/experimental_correlation_report.csv",
        f"{config['approved']['table_dir']}/AIC.csv",
        f"{config['all_phase3']['fig_dir']}/roc_curve.pdf"


rule preprocess:
    input:
        f"{config['approved']['metadata_sheet']}",
        f"{config['all_phase3']['metadata_sheet']}",
        expand("{input_file}", input_file=get_raw_files("approved")),
        expand("{input_file}", input_file=get_raw_files("all_phase3")),
        expand("{input_file}", input_file=get_raw_files("placebo"))
    output:
        ALL_APPROVED_TRIALS,
        ALL_PLACEBO_TRIALS,
        ALL_PHASE3_TRIALS,
        f"{config['approved']['fig_dir']}/preprocess_sanity_check.png",
        f"{config['all_phase3']['fig_dir']}/preprocess_sanity_check.png"
    conda:
        "env/environment_short.yml"
    shell:
        "python src/preprocessing.py approved; "
        "python src/preprocessing.py placebo; "
        "python src/preprocessing.py all_phase3"

rule find_seeds:
    input:
        f"{config['approved']['metadata_sheet']}",
        f"{config['all_phase3']['metadata_sheet']}",
        ALL_APPROVED_TRIALS,
        ALL_PHASE3_TRIALS
    output:
        f"{config['approved']['metadata_sheet_seed']}",
        f"{config['all_phase3']['metadata_sheet_seed']}"
    shell:
        "python src/find_median_sim.py approved; "
        "python src/find_median_sim.py all_phase3"        

rule hsa_additivity_prediction:
    input:
        config['approved']['metadata_sheet_seed'],
        config['all_phase3']['metadata_sheet_seed'],
        ALL_APPROVED_TRIALS,
        ALL_PHASE3_TRIALS
    output:
        ALL_PRED_FILES
    shell:
        "python src/hsa_additivity_model.py approved; "
        "python src/hsa_additivity_model.py all_phase3"

rule cox_ph_test:
    input:
        ALL_PRED_FILES
    output:
        config['approved']['cox_result'],
        config['all_phase3']['cox_result']
    shell:
        "python src/coxhazard_test.py approved; "
        "python src/coxhazard_test.py all_phase3"

rule forest_plot:
    input:
        config['approved']['cox_result']
    output:
        f"{config['approved']['fig_dir']}/forest_plot.pdf"
    shell:
        "python src/plotting/forest_plot.py {output}"

rule main_survival_plots:
    input:
        config['approved']['cox_result'],
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
    output:
        f"{config['approved']['fig_dir']}/additive_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/between_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/hsa_survival_plots.pdf"
    script:
        "src/plotting/plot_survival_curves.py"

rule qqplots:
    input:
        config['approved']['cox_result'],
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
    output:
        f"{config['approved']['fig_dir']}/all_combo_qqplot.pdf",
        f"{config['approved']['fig_dir']}/qqplot_legend.pdf"
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/all_survival_qqplot.py"

rule performance_histogram:
    input:
        config['approved']['cox_result'],
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
    output:
        f"{config['approved']['fig_dir']}/r2_histogram.pdf",
        f"{config['approved']['fig_dir']}/msd_histogram.pdf",
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/r2_msd_histogram.py"

rule subgroup_boxplots:
    input:
        config['approved']['cox_result']
    output:    
        f"{config['approved']['fig_dir']}/ici_boxplot.pdf",
        f"{config['approved']['fig_dir']}/angiogenesis_boxplot.pdf",
        f"{config['approved']['fig_dir']}/monotherapy_approval_boxplot.pdf",
        f"{config['approved']['fig_dir']}/HRmedian_boxplot.pdf"
    conda:
        "env/environment_short.yml"
    script:
        "src/plotting/subgroup_boxplots.py"

rule placebo_survival_plots:
    input:
        ALL_PLACEBO_TRIALS
    output:
        f"{config['placebo']['fig_dir']}/placebo_survival_plots.pdf"
    script:
        "src/plotting/plot_placebo.py"

rule relative_dose_plot:
    input:
        config['relative_doses']
    output:
        f"{config['approved']['fig_dir']}/relative_doses.pdf"
    script:
        "src/plotting/plot_doses_difference.py"

rule suppl_survival_plots:
    input:
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        config['approved']['cox_result']
    output:
        f"{config['approved']['fig_dir']}/suppl_additive_survival_plots.pdf",
        f"{config['approved']['fig_dir']}/suppl_between_hsa_survival_plots.pdf"
    shell:
        "python src/plotting/plot_survival_curves_suppl.py approved"

rule predict_success:
    input:
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        f"{config['approved']['cox_result']}",
        f"{config['all_phase3']['cox_result']}"
    output:
        f"{config['fig_dir']}/HR_combo_control_scatterplot.pdf",
        f"{config['table_dir']}/HR_predicted_vs_control.csv"
    shell:
        "python src/predict_success.py both"

rule HSA_additive_diff:
    input:
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        config['approved']['cox_result']
    output:
        f"{config['approved']['fig_dir']}/explain_HSA_additive_difference.pdf",
        f"{config['approved']['table_dir']}/added_benefit_hsa_add_syn.csv",
        f"{config['approved']['fig_dir']}/hsa_additivity_sigma.pdf"
    shell:
        "python src/lognormal_examples.py; "
        "python src/hsa_add_diff.py"

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

rule AIC:
    input:
        ALL_APPROVED_PRED_FILES,
        f"{config['approved']['cox_result']}"
    output:
        f"{config['approved']['table_dir']}/AIC.csv"
    script:
        "src/AIC_calculation.py"

rule all_phase3_predictive_power:
    input:
        ALL_PHASE3_TRIALS,
        ALL_PHASE3_PRED_FILES,
        f"{config['all_phase3']['metadata_sheet_seed']}"
    output:
        f"{config['all_phase3']['table_dir']}/predictive_power.csv",
        f"{config['all_phase3']['fig_dir']}/roc_curve.pdf",
        f"{config['all_phase3']['fig_dir']}/precision-recall_curve.pdf",
        f"{config['all_phase3']['fig_dir']}/additivity_prob_success_swarm_plot.pdf"
    script:
        "src/all_phase3_predictive_power.py"
