import pandas as pd
import numpy as np
from pathlib import Path
from hsa_additivity_model import predict_both, subtract_which_scan_time
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_INPUT_SHEET = CONFIG['metadata_sheet']['combo']
COMBO_SEED_SHEET = CONFIG['metadata_sheet']['combo_seed']
RAW_COMBO_DIR = CONFIG['dir']['raw_combo_data']
COMBO_DATA_DIR = CONFIG['dir']['combo_data']
OUTDIR = CONFIG['dir']['temp'] + '/find_median_sim'
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
NRUN = 100

def make_predictions_diff_seeds():
    indf = pd.read_csv(COMBO_INPUT_SHEET, sep='\t')
    for i in indf.index:
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        name_ab = indf.at[i, 'Combination']
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value

        df_a = pd.read_csv(f'{COMBO_DATA_DIR}/{name_a}.clean.csv',
                           header=0, index_col=False)
        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv',
                           header=0, index_col=False)
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv',
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
            add.to_csv(f'{OUTDIR}/{name_a}-{name_b}_combination_predicted_add_run{seed:02d}.csv')
            ind.to_csv(f'{OUTDIR}/{name_a}-{name_b}_combination_predicted_ind_run{seed:02d}.csv')


def find_median_sim(save=True) -> pd.DataFrame:
    indf = pd.read_csv(COMBO_INPUT_SHEET, sep='\t')
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
            ind = pd.read_csv(f'{OUTDIR}/{name_a}-{name_b}_combination_predicted_ind_run{seed:02d}.csv')
            add = pd.read_csv(f'{OUTDIR}/{name_a}-{name_b}_combination_predicted_add_run{seed:02d}.csv')
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
        med_df.to_csv(COMBO_SEED_SHEET, index=False, sep='\t')
    
    return med_df

if __name__ == '__main__':
    make_predictions_diff_seeds()
    find_median_sim()
