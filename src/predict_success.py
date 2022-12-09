import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from plotting.plot_utils import import_input_data
from coxhazard_test import create_ipd, get_cox_results
from plotting.plot_predict_success import plot_predict_success
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_DATA_DIR = CONFIG['dir']['combo_data']
PFS_PRED_DIR = CONFIG['dir']['PFS_prediction']
FIG_DIR = CONFIG['dir']['figures']
TABLE_DIR = CONFIG['dir']['tables']


def predict_success():
    """Predict whether the trial would have been successful based
    on additivity and HSA predictions by Cox-PH test.

    Returns:
        pd.DataFrame: predicted results
    """    
    input_df = import_input_data()
    # output dataframe
    results = pd.DataFrame(index=input_df.index,
                          columns=['p_ind', 'HR_ind', 'HRlower_ind', 'HRupper_ind',
                                   'p_add', 'HR_add', 'HRlower_add', 'HRupper_add'])
    for i in range(input_df.shape[0]):
        name_a = input_df.at[i, 'Experimental']
        name_b = input_df.at[i, 'Control']
        name_ab = input_df.at[i, 'Combination']
        n_control = input_df.at[i, 'N_control'].astype(int)
        n_combo = input_df.at[i, 'N_combination'].astype(int)
        # observed data
        df_a = pd.read_csv(f'{COMBO_DATA_DIR}/{name_a}.clean.csv')
        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')

        try:
            ipd_control = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}_indiv.csv')

        except FileNotFoundError:
            ipd_control = create_ipd(df_b, n=n_control)

        # import prediction
        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        tmax = np.amin([df_ab['Time'].max(), independent['Time'].max(), 
                        df_a['Time'].max(), df_b['Time'].max()])
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        ipd_add = create_ipd(additive, n=n_combo)
        ipd_ind = create_ipd(independent, n=n_combo)

        # additive
        p, hr, lower, upper = get_cox_results(ipd_control, ipd_add)
        results.at[i, 'p_add'] = p
        results.at[i, 'HR_add'] = hr
        results.at[i, 'HRlower_add'] = lower
        results.at[i, 'HRupper_add'] = upper

        # independent
        p, hr, lower, upper = get_cox_results(ipd_control, ipd_ind)
        results.at[i, 'p_ind'] = p
        results.at[i, 'HR_ind'] = hr
        results.at[i, 'HRlower_ind'] = lower
        results.at[i, 'HRupper_ind'] = upper
    
    results.loc[:, 'success_ind'] = results['p_ind'] < 0.05
    results.loc[:, 'success_add'] = results['p_add'] < 0.05

    return pd.concat([input_df.iloc[:, :21], results], axis=1)


def calc_correlation(model, results):
    """Calculate pearsonr correlation between predicted HR and observed HR.

    Args:
        model (str): which model to use. 'additivity' or 'HSA'
        results (pd.DataFrame): predicted success results

    Returns:
        (float, float): pearson rvalue, pvalue
    """    
    if model == 'HSA':
        return pearsonr(results['HR(combo/control)'], results['HR_ind'])
    elif model == 'additivity':
        return pearsonr(results['HR(combo/control)'], results['HR_add'])
    else:
        print("Wrong model argument")
        return


def main():
    results = predict_success()
    results.to_csv(f'{TABLE_DIR}/HR_predicted_vs_control.csv')
    r_hsa, p_hsa = calc_correlation('HSA', results)
    r_add, p_add = calc_correlation('additivity', results)
    print("r_hsa={0:.02f}, p_hsa={1:.03f}, r_add={2:.02f}, p_add={3:.03f}".format(
        r_hsa, p_hsa, r_add, p_add))
    fig = plot_predict_success(results)
    fig.savefig(f'{FIG_DIR}/HR_combo_control_scatterplot.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()