import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
from plotting.plot_utils import import_input_data
from coxhazard_test import create_ipd, get_cox_results
from plotting.plot_predict_success import plot_predict_success
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


def predict_success(input_df: pd.DataFrame, data_dir: str, pred_dir: str) -> pd.DataFrame:
    """Predict whether the trial would have been successful based
    on additivity and HSA predictions by Cox-PH test.

    Args:
        input_df (pd.DataFrame): _description_
        data_dir (str): directory path to observed survival data
        pred_dir (str): directory path to predicted survival data

    Returns:
        pd.DataFrame: predicted results
    """    
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
        df_a = pd.read_csv(f'{data_dir}/{name_a}.clean.csv')
        df_b = pd.read_csv(f'{data_dir}/{name_b}.clean.csv')
        df_ab = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')

        try:
            ipd_control = pd.read_csv(f'{data_dir}/{name_b}_indiv.csv')

        except FileNotFoundError:
            ipd_control = create_ipd(df_b, n=n_control)

        # import prediction
        independent = pd.read_csv(
            f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, 
                        help='Dataset to use (approved, all_phase3, placebo')
    args = parser.parse_args()
    
    config_dict = CONFIG[args.dataset]
    if args.dataset == 'approved':
        cox_df = import_input_data()
    else:
        cox_df = pd.read_csv(config_dict['cox_result'])
    data_dir = config_dict['data_dir']
    pred_dir = config_dict['pred_dir']
    table_dir = config_dict['table_dir']
    fig_dir = config_dict['fig_dir']
    results = predict_success(cox_df, data_dir, pred_dir)
    results.to_csv(f'{table_dir}/HR_predicted_vs_control.csv', index=False)
    
    r_hsa, p_hsa = calc_correlation('HSA', results)
    r_add, p_add = calc_correlation('additivity', results)
    
    print("r_hsa={0:.02f}, p_hsa={1:.03f}, r_add={2:.02f}, p_add={3:.03f}".format(
        r_hsa, p_hsa, r_add, p_add))
    
    fig = plot_predict_success(results)
    fig.savefig(f'{fig_dir}/HR_combo_control_scatterplot.pdf',
                bbox_inches='tight', pad_inches=0.1)



if __name__ == '__main__':
    main()