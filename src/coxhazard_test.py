import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from utils import interpolate
import sys
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_SEED_SHEET = CONFIG['metadata_sheet']['combo_seed']
COMBO_DATA_DIR = CONFIG['dir']['combo_data']
RAW_COMBO_DATA_DIR = CONFIG['dir']['raw_combo_data']
PFS_PRED_DIR = CONFIG['dir']['PFS_prediction']
OUTDIR = CONFIG['dir']['tables']

def create_ipd(df: pd.DataFrame, n=500) -> pd.DataFrame:
    #FIXME works fine as is, but can be problematic if you don't preprocess the additiivty
    # and HSA predictions that the survival curves go down to zero (which is misleading)
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
    events = np.hstack((np.repeat(0, round(min_surv * n)), np.repeat(1, round((1 - min_surv) * n))))
    t = interp(np.linspace(0, 100, n))
    return pd.DataFrame({'Time': t, 'Event': events})


def get_cox_results(ipd_base, ipd_test):
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


def cox_ph_test(input_df: pd.DataFrame) -> pd.DataFrame:
    tmp = input_df
    # output dataframe
    cox_df = pd.DataFrame(index=tmp.index, 
                                columns=['p_ind', 'HR_ind', 'HRlower_ind', 'HRupper_ind', 
                                         'p_add', 'HR_add', 'HRlower_add', 'HRupper_add', 'Model'])

    for i in range(tmp.shape[0]):
        print(i)
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']

        # observed data
        path = tmp.at[i, 'Path'] + '/'
        df_a = pd.read_csv(path + tmp.at[i, 'Experimental'] + '.clean.csv').dropna()
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv').dropna()
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv').dropna()
        
        try:
            ipd_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '_indiv.csv')
            print("used IPD")
        except FileNotFoundError:
            ipd_ab = create_ipd(df_ab, n=200)

        # import prediction
        independent = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b)).dropna()
        additive = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b)).dropna()

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


def main():
    indf = pd.read_csv(COMBO_SEED_SHEET, sep='\t')
    results = cox_ph_test(indf)
    # default save when output file name is not given
    if len(sys.argv) == 1:
        results.to_csv(f'{OUTDIR}/cox_ph_test.csv', index=False)
    else:
        results.to_csv(sys.argv[1], index=False)


if __name__ == '__main__':
    main()
