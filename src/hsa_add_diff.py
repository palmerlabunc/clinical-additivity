import numpy as np
import pandas as pd
from coxhazard_test import get_cox_results, create_ipd
from utils import interpolate
from lognormal_fitting import fit_lognormal
from plotting.plot_hsa_add_diff import plot_hsa_add_diff_vs_lognormal, corr_hsa_add_diff_vs_lognormal
from plotting.plot_utils import import_input_data
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_DATA_DIR = CONFIG['dir']['combo_data']
PFS_PRED_DIR = CONFIG['dir']['PFS_prediction']
FIG_DIR = CONFIG['dir']['figures']
TABLE_DIR = CONFIG['dir']['tables']


def hsa_add_diff():
    """Compute difference between additivity and HSA by calculating
    normalized difference between the two curves and performing
    Cox-PH model between the two survival curves for each combination.

    Returns:
        pd.DataFrame: difference between HSA and additivity
    """    
    cox_df = import_input_data()
    diff_df = pd.DataFrame(index=cox_df.index, columns=[
                           'HSA - Control', 'Additivity - HSA',
                           'p', 'HR', 'HRlower', 'HRupper'])
    n = 5000
    for i in range(cox_df.shape[0]):
        name_a = cox_df.at[i, 'Experimental']
        name_b = cox_df.at[i, 'Control']

        # import data
        control = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')
        tmax = np.amin(
            [control['Time'].max(), independent['Time'].max(), additive['Time'].max()])

        tmax = np.amin([additive['Time'].max(), independent['Time'].max()])
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        ipd_add = create_ipd(additive, n=n)
        ipd_ind = create_ipd(independent, n=n)

        # set up data for Cox regress
        p, hr, hr_lower, hr_upper = get_cox_results(ipd_add, ipd_ind)
        diff_df.at[i, 'p'] = p
        diff_df.at[i, 'HR'] = hr
        diff_df.at[i, 'HRlower'] = hr_lower
        diff_df.at[i, 'HRupper'] = hr_upper

        f_ctrl = interpolate(control, x='Time', y='Survival')
        f_ind = interpolate(independent, x='Time', y='Survival')
        f_add = interpolate(additive, x='Time', y='Survival')

        timepoints = np.linspace(0, tmax, n)

        diff_df.at[i,
                   'HSA - Control'] = sum(f_ind(timepoints) - f_ctrl(timepoints))/n
        diff_df.at[i,
                   'Additivity - HSA'] = sum(f_add(timepoints) - f_ind(timepoints))/n

    return diff_df.astype(np.float64)


def added_benefit_hsa_add_syn():
    """Compute normalized difference between survival curves. 
    Combo - Control, HSA - Control, Additivity - HSA, Combo - Additivity.

    Returns:
        pd.DataFrame: result dataframe
    """    
    cox_df = import_input_data()
    diff_df = pd.DataFrame(index=cox_df.index, 
                           columns=['Combo - Control', 
                                    'HSA - Control', 
                                    'Additivity - HSA', 
                                    'Combo - Additivity'])
    diff_df.loc[:, 'Model'] = cox_df['Model']
    diff_df.loc[:, 'Combination'] = cox_df['Combination']

    n = 5000
    for i in range(cox_df.shape[0]):
        name_a = cox_df.at[i, 'Experimental']
        name_b = cox_df.at[i, 'Control']
        name_ab = cox_df.at[i, 'Combination']

        # import data
        control = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        obs_combo = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')
        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')
        tmax = np.amin(
            [control['Time'].max(), independent['Time'].max(), additive['Time'].max()])
        f_ctrl = interpolate(control, x='Time', y='Survival')
        f_ind = interpolate(independent, x='Time', y='Survival')
        f_add = interpolate(additive, x='Time', y='Survival')
        f_obs = interpolate(obs_combo, x='Time', y='Survival')
        timepoints = np.linspace(0, tmax, n)

        diff_df.at[i,
                'Combo - Control'] = sum(f_obs(timepoints) - f_ctrl(timepoints))/n
        diff_df.at[i,
                'HSA - Control'] = sum(f_ind(timepoints) - f_ctrl(timepoints))/n
        diff_df.at[i,
                'Additivity - HSA'] = sum(f_add(timepoints) - f_ind(timepoints))/n
        diff_df.at[i, 'Combo - Additivity'] = sum(
            f_obs(timepoints) - f_add(timepoints))/n
        
    return diff_df


def main():
    added_df = added_benefit_hsa_add_syn()
    added_df.round(5).to_csv(
        f'{TABLE_DIR}/added_benefit_hsa_add_syn.csv', index=False)
    lognorm_df = fit_lognormal()
    diff_df = hsa_add_diff()
    r, p = corr_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    print(f'pearsonr={r}\npvalue={p}')
    fig = plot_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    fig.savefig(f'{FIG_DIR}/hsa_additivity_sigma.pdf')


if __name__ == '__main__':
    main()
