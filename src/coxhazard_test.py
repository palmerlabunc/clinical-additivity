import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
#from src.utils import interpolate
from utils import interpolate
from statsmodels.stats.multitest import multipletests
import sys
import yaml
import argparse

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

def create_ipd(df: pd.DataFrame, n=500) -> pd.DataFrame:
    #FIXME works fine as is, but can be problematic if you don't preprocess the additiivty
    # and HSA predictions that the survival curves go down to zero (which is misleading)
    # In current version, you need to trim the end of the curve before tmax
    """Creates individual patient data (IPD) for a given survival curve.
    The survival curve is broken into n equal survival intervals. All events are considered
    as failed before the end of follow-up. All events after follow-up are considered
    censored.

    Args:
        df (pd.DataFrame): survival data points
        n (int, optional): number of patients to generate. Defaults to 500.

    Returns:
        pd.DataFrame: individual patient data
    """    
    interp = interpolate(df, x='Survival', y='Time')
    # censoring due to loss of follow-up at the tail
    min_surv = np.round(np.ceil(df['Survival'].min())/100, 2)
    events = np.hstack((np.repeat(0, round(min_surv * n)), 
                        np.repeat(1, round((1 - min_surv) * n))))
    if len(events) > n:
        events = events[len(events) - n:]

    t = interp(np.linspace(0, 100, n))
    return pd.DataFrame({'Time': t, 'Event': events})


def get_cox_results(ipd_base: pd.DataFrame, ipd_test: pd.DataFrame) -> tuple:
    """Perform Cox PH test. IPD should have columns Time, Event.
    HR < 1 indicates that test has less hazard (i.e., better than) base.

    Args:
        ipd_base (pd.DataFrame): IPD of control arm.
        ipd_test (pd.DataFrame): IPD of test arm. 

    Returns:
        (float, float, float, float): p, HR, lower 95% CI, upper 95% CI
    """    
    cph = CoxPHFitter()
    ipd_base.loc[:, 'Arm'] = 0
    ipd_test.loc[:, 'Arm'] = 1
    merged = pd.concat([ipd_base, ipd_test],
                        axis=0).reset_index(drop=True)
    cph.fit(merged, duration_col='Time', event_col='Event')
    return tuple(cph.summary.loc['Arm', ['p', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])


def cox_ph_test(dataset: str) -> pd.DataFrame:
    config_dict = CONFIG[dataset]
    sheet = config_dict['metadata_sheet_seed']
    data_dir = config_dict['data_dir']
    raw_dir = config_dict['raw_dir']
    pred_dir = config_dict['pred_dir']
    
    input_df = pd.read_csv(sheet, sep='\t')
    tmp = input_df
    # output dataframe
    cox_df = pd.DataFrame(index=tmp.index, 
                          columns=['p_ind', 'HR_ind', 'HRlower_ind', 'HRupper_ind', 
                                   'p_add', 'HR_add', 'HRlower_add', 'HRupper_add', 'Model'])
    cox_df = pd.concat([input_df, cox_df], axis=1)
    for i in range(tmp.shape[0]):

        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']
        n_combo = tmp.at[i, 'N_combination']
        print(i, n_combo)
        # observed data
        df_a = pd.read_csv(f'{data_dir}/{name_a}.clean.csv').dropna()
        df_b = pd.read_csv(f'{data_dir}/{name_b}.clean.csv').dropna()
        df_ab = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv').dropna()
        
        try:
            ipd_ab = pd.read_csv(f'{raw_dir}/{name_ab}_indiv.csv')
            print("used IPD")
        except FileNotFoundError:
            ipd_ab = create_ipd(df_ab, n=n_combo)

        # import prediction
        independent = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv').dropna()
        additive = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv').dropna()

        tmax = np.amin([df_ab['Time'].max(), independent['Time'].max(), df_a['Time'].max(), df_b['Time'].max()])
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]
        
        ipd_add = create_ipd(additive)
        ipd_ind = create_ipd(independent)

        # additive
        p, hr, hr_lower_add, hr_upper_add = get_cox_results(ipd_add, ipd_ab)
        cox_df.at[i, 'p_add'] = p
        cox_df.at[i, 'HR_add'] = hr
        cox_df.at[i, 'HRlower_add'] = hr_lower_add
        cox_df.at[i, 'HRupper_add'] = hr_upper_add

        # independent
        p, hr, hr_lower_ind, hr_upper_ind = get_cox_results(ipd_ind, ipd_ab)
        cox_df.at[i, 'p_ind'] = p
        cox_df.at[i, 'HR_ind'] = hr
        cox_df.at[i, 'HRlower_ind'] = hr_lower_ind
        cox_df.at[i, 'HRupper_ind'] = hr_upper_ind
    
    # assign model
    cond_add = (cox_df['HRupper_ind'] < 1) & (
        cox_df['HRlower_add'] <= 1) & (cox_df['HRupper_add'] >= 1)
    cond_ind = (cox_df['HRlower_add'] > 1) & (
        cox_df['HRlower_ind'] <= 1) & (cox_df['HRupper_ind'] >= 1)
    cond_syn = cox_df['HRupper_add'] < 1
    cond_bad = cox_df['HRlower_ind'] > 1
    cond_btn = (cox_df['HRlower_ind'] <= 1) & (
        cox_df['HRupper_ind'] >= 1) & (
        cox_df['HRlower_add'] <= 1) & (cox_df['HRupper_add'] > 1)

    cox_df.loc[cond_add, 'Model'] = 'additive'
    cox_df.loc[cond_ind, 'Model'] = 'independent'
    cox_df.loc[cond_syn, 'Model'] = 'synergy'
    cox_df.loc[cond_bad, 'Model'] = 'worse than independent'
    cox_df.loc[cond_btn, 'Model'] = 'between'

    # assign figure
    cox_df.loc[:, 'Figure'] = cox_df['Model']
    cox_df.loc[cox_df['Main analysis'] == 0, 'Figure'] = 'suppl'

    return cox_df


def apply_fdr(df):
    _, p_ind_adj, _, _ = multipletests(df['p_ind'], method='fdr_bh')
    _, p_add_adj, _, _ = multipletests(df['p_add'], method='fdr_bh')
    df.loc[:, 'p_ind_bh'] = p_ind_adj
    df.loc[:, 'p_add_bh'] = p_add_adj
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, 
                        help='Dataset to use (approved, all_phase3, placebo')
    args = parser.parse_args()

    outfile = CONFIG[args.dataset]['cox_result']
    results = cox_ph_test(args.dataset)
    results = apply_fdr(results)
    results.to_csv(outfile, index=False)


if __name__ == '__main__':
    main()
