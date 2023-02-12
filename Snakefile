configfile: "config.yaml"

import pandas as pd
import glob
import os


datasets = ['approved', 'placebo', 'all_phase3']

# DIRECTORIES
#EXPERIMENTAL_DATA_DIR = config['dir']['experimental_data']
#FIG_DIR = config['dir']['figures']
#TABLE_DIR = config['dir']['tables']
#PFS_PRED_DIR = config['dir']['PFS_prediction']

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
placebo_df = pd.read_csv(config['metadata_sheet']['placebo'], sep='\t', header=0)
placebo_list = list(placebo_df['File prefix'])

# FILE LISTS
ALL_APPROVED_TRIALS = expand("{output_file}", output_file=get_trial_files("approved"))
ALL_PLACEBO_TRIALS = expand("{output_file}", output_file=get_trial_files("placebo"))
ALL_PHASE3_TRIALS = expand("{output_file}", output_file=get_trial_files("all_phase3"))
ALL_APPROVED_PRED_FILES =  expand(f"{PFS_PRED_DIR}/{{pred}}_combination_predicted_{{model}}.csv", 
                                  pred=phase3_pred_list, model=['ind', 'add'])
ALL_PHASE3_PRED_FILES =  expand(f"{PFS_PRED_DIR}/{{pred}}_combination_predicted_{{model}}.csv", 
                                  pred=approved_pred_list, model=['ind', 'add'])

# OTHER RESULTS
#APPROVED_COX_RESULT = config['cox_result']
#PHASE3_COX_RESULT = config['phase3_cox_result']

""""
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
"""


rule preprocess:
    input:
        lambda wildcards: expand("{input_file}", input_file=get_raw_files(wildcards.dataset))
    output:
        expand("{output_file}", output_file=get_trial_files("approved")),
        expand("{output_file}", output_file=get_trial_files("all_phase3")),
        expand("{output_file}", output_file=get_trial_files("placebo"))
    conda:
        "env/environment_short.yml"
    shell:
        "python src/preprocessing.py approved "
        "python src/preprocessing.py placebo "
        "python src/preprocessing.py all_phase3"


rule find_seeds:
    input:
        sheet = config['{dataset}']['metadata_sheet'],
        pred_dir = config['{dataset}']['pred_dir'],
        data_dir = config['{dataset}']['data_dir'],
        ALL_APPROVED_TRIALS,
        ALL_PHASE3_TRIALS
    output:
        config['{dataset}']['metadata_sheet_seed']
    shell:
        "python src/find_median_sim.py -s {input.sheet} -p {input.pred_dir} -d {input.data_dir} -o {output}"
"""
rule hsa_additivity_prediction:
    input:
        config['{dataset}']['metadata_sheet_seed'],
        ALL_APPROVED_TRIALS
    output:
        ALL_APPROVED_PRED_FILES
    script:
        "src/hsa_additivity_model.py {dataset}"

rule cox_ph_test:
    input:
        ALL_APPROVED_PRED_FILES
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
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
    output:
        f"{FIG_DIR}/additive_survival_plots.pdf",
        f"{FIG_DIR}/between_survival_plots.pdf",
        f"{FIG_DIR}/hsa_survival_plots.pdf"
    script:
        "src/plotting/plot_survival_curves.py"

rule qqplots:
    input:
        COX_RESULT,
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
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
        ALL_APPROVED_PRED_FILES,
        ALL_APPROVED_TRIALS
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
        ALL_PLACEBO_TRIALS
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
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        COX_RESULT
    output:
        f"{FIG_DIR}/all_phase3_suppl_additive_survival_plots.pdf",
        f"{FIG_DIR}/all_phase3_suppl_between_hsa_survival_plots.pdf"
    script:
        "src/plotting/plot_survival_curves_suppl.py"

rule predict_success:
    input:
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        COX_RESULT
    output:
        f'{FIG_DIR}/HR_combo_control_scatterplot.pdf',
        f'{TABLE_DIR}/HR_predicted_vs_control.csv'
    script:
        "src/predict_success.py"

rule HSA_additive_diff:
    input:
        ALL_APPROVED_TRIALS,
        ALL_APPROVED_PRED_FILES,
        COX_RESULT
    output:
        f'{FIG_DIR}/explain_HSA_additive_difference.pdf',
        f'{TABLE_DIR}/added_benefit_hsa_add_syn.csv',
        f'{FIG_DIR}/hsa_additivity_sigma.pdf'
    shell:
        "python src/lognormal_examples.py "
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
        COX_RESULT
    output:
        f'{TABLE_DIR}/approved_combinations_AIC.csv'
    script:
        "src/AIC_calculation.py"
"""