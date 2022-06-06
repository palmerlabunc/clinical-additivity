import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from utils import populate_N_patients, fit_rho3

OUTDIR = '../analysis/PFS_predictions/{}/'.format(date.today())
new_directory = Path(OUTDIR)
new_directory.mkdir(parents=True, exist_ok=True)

def sample_joint_response_add(ori_a, ori_b, subtracted, scan_time):
    """Calculates predicted PFS time for n-patients in combination therapy under additivity.

    Args:
        ori_a (pd.DataFrame): survival data for treatment A
        ori_b (pd.DataFrame): survival data for treatment B
        subtracted (str): treatment to subtract initial scan time from ('a' or 'b')
        scan_time (float): first scan time

    Returns:
        list: Sorted PFS times
    """
    # ensure that a + b > a and a + b > b
    if subtracted == 'a':
        adjusted = ori_a - scan_time
        adjusted[adjusted < 0] = 0
        predicted = np.maximum(adjusted + ori_b, ori_a)
    elif subtracted == 'b':
        adjusted = ori_b - scan_time
        adjusted[adjusted < 0] = 0
        predicted = np.maximum(ori_a + adjusted, ori_b)
    else:
        print("wrong argument")
    return sorted(predicted, reverse=True)


def sample_joint_response(ori_a, ori_b):
    """ Calculate predicted PFS time for n-patients in combination therapy under HSA.

    Args:
        ori_a (pd.DataFrame): survival data for treatment A
        ori_b (pd.DataFrame): survival data for treatment B

    Returns:
        list: list of predicted PFS for n-patients in combination therapy under HSA
    """
    return sorted(np.maximum(ori_a, ori_b), reverse=True)


def predict_both(df_a, df_b, name_a, name_b, subtracted, scan_time, N=5000, rho=0.3):
    """ Predict combination effect using HSA and additivity model and writes csv output.

    Args:
        df_a (pd.DataFrame): survival data for treatment A (Experimental)
        df_b (pd.DataFrame): survival data for treatment B (Control)
        name_a (str): treatment A name
        name_b (str): treatment B name
        subtracted (str): treatment arm to substract first scan time from ('a' or 'b')
        scan_time (float): first scan time
        N (int, optional): number of virtual patients. Defaults to 5000.
        rho (float, optional): correlation value. Defaults to 0.3.
    """
    a = populate_N_patients(df_a, N)
    b = populate_N_patients(df_b, N)

    # if the shorter curve goes down to < 5%, ok to set different max time
    # if not, set universal max time as min of max times
    min_idx = np.argmin([df_a['Time'].max(), df_b['Time'].max()])
    tmax_flag = True
    if min_idx == 0 and df_a['Survival'].min() < 5:
        tmax_flag = False
    elif min_idx == 1 and df_b['Survival'].min() < 5:
        tmax_flag = False
    
    if tmax_flag:
        print(name_a, name_b)
        tmax = min(a['Time'].max(), b['Time'].max())
        a.loc[a['Time'] > tmax, 'Time'] = tmax
        b.loc[b['Time'] > tmax, 'Time'] = tmax
    
    patients = a['Survival'].values
    new_a, new_b = fit_rho3(a['Time'].values, b['Time'].values, rho)
    additivity = pd.DataFrame({'Time': sample_joint_response_add(new_a, new_b, subtracted, scan_time),
                               'Survival': a['Survival'].values})
    independent = pd.DataFrame({'Time': sample_joint_response(new_a, new_b), 
                                'Survival': a['Survival'].values})

    additivity = additivity.sort_values('Survival', ascending=True).reset_index(drop=True)
    independent = independent.sort_values('Survival', ascending=True).reset_index(drop=True)

    additivity.round(5).to_csv(
        OUTDIR + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b), index=False)
    independent.round(5).to_csv(
        OUTDIR + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b), index=False)


def main():
    indf = pd.read_csv('../data/trials/final_input_list.txt', sep='\t')
    for i in indf.index:
        print(i)
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        path = indf.at[i, 'Path'] + '/'
        corr = indf.at[i, 'Corr']
        df_a = pd.read_csv(path + indf.at[i, 'Experimental'] + '.clean.csv', 
                           header=0, index_col=False)
        df_b = pd.read_csv(path + indf.at[i, 'Control'] + '.clean.csv', 
                           header=0, index_col=False)

        # subtract initial scan time of the larger one
        scan_a = indf.at[i, 'Experimental First Scan Time']
        scan_b = indf.at[i, 'Control First Scan Time']
        if scan_a == 9999:
            scan_a = -9999
        if scan_b == 9999:
            scan_b = -9999
        if scan_a < scan_b:
            scan_time = scan_b
            subtracted = 'b'
        else:
            scan_time = scan_a
            subtracted = 'a'

        predict_both(df_a, df_b, name_a, name_b, subtracted, scan_time, rho=corr)

if __name__ == "__main__":
    main()