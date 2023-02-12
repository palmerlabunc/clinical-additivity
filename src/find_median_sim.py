import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from hsa_additivity_model import predict_both, subtract_which_scan_time

NRUN = 100

def make_predictions_diff_seeds(indf: pd.DataFrame, data_dir: str, pred_dir: str):
    for i in indf.index:
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        name_ab = indf.at[i, 'Combination']
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value

        df_a = pd.read_csv(f'{data_dir}/{name_a}.clean.csv',
                           header=0, index_col=False)
        df_b = pd.read_csv(f'{data_dir}/{name_b}.clean.csv',
                           header=0, index_col=False)
        df_ab = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv',
                            header=0, index_col=False)
        # subtract initial scan time of the larger one
        scan_a = indf.at[i, 'Experimental First Scan Time']
        scan_b = indf.at[i, 'Control First Scan Time']

        subtracted, scan_time = subtract_which_scan_time(scan_a, scan_b)
        for k in range(NRUN):
            seed = k
            ind, add = predict_both(df_a, df_b, name_a, name_b, 
                                    subtracted, scan_time,
                                    df_ab=df_ab, rho=corr, 
                                    seed_ind=seed, seed_add=seed, 
                                    save=False)
            add.to_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add_run{seed:02d}.csv')
            ind.to_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind_run{seed:02d}.csv')


def find_median_sim(indf: pd.DataFrame, pred_dir: str, save=True, outfile=None) -> pd.DataFrame:
    med_df = indf.copy()
    med_df.loc[:, 'ind_median_std'] = np.nan
    med_df.loc[:, 'add_median_std'] = np.nan
    med_df.loc[:, 'ind_median_run'] = 0
    med_df.loc[:, 'add_median_run'] = 0

    for i in range(indf.shape[0]):
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        ind_arr = np.zeros(NRUN)
        add_arr = np.zeros(NRUN)
        for seed in range(NRUN):
            ind = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind_run{seed:02d}.csv')
            add = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add_run{seed:02d}.csv')
            ind_arr[seed] = ind.loc[2499:2500, 'Time'].mean()
            add_arr[seed] = add.loc[2499:2500, 'Time'].mean()
        med_df.loc[i, 'ind_median_std'] = np.std(ind_arr)
        med_df.loc[i, 'add_median_std'] = np.std(add_arr)
        # save run# of the median
        ind_idx = np.argsort(ind_arr)[len(ind_arr)//2]
        add_idx = np.argsort(add_arr)[len(add_arr)//2]
        med_df.loc[i, 'ind_median_run'] = ind_idx
        med_df.loc[i, 'add_median_run'] = add_idx

    if save:
        med_df.to_csv(outfile, index=False, sep='\t')
    
    return med_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sheet' '-s', type=str, 
                        help='Dataset to use (approved, all_phase3, placebo')
    parser.add_argument('--pred-dir' '-p', type=str,
                        help='Dataset to use (approved, all_phase3, placebo')
    parser.add_argument('--data-dir' '-d', type=str,
                        help='Dataset to use (approved, all_phase3, placebo')
    parser.add_argument('--outfile' '-o', type=str,
                        help='Dataset to use (approved, all_phase3, placebo')
    args = parser.parse_args()
    indf = pd.read_csv(args.sheet)
    make_predictions_diff_seeds(indf, args.data_dir, args.pred_dir)
    find_median_sim(indf, args.pred_dir, save=True, outfile=args.outfile)
