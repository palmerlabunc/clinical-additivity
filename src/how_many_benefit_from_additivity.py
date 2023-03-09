import numpy as np
import pandas as pd
import yaml
from utils import populate_N_patients
from hsa_additivity_model import subtract_which_scan_time, set_tmax

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)
config_dict = CONFIG['approved']


def calculate_how_many_benefit_from_additivity(df_baseline: pd.DataFrame, 
                                               df_added: pd.DataFrame, scan_time: float, tmax: float) -> float:
    # what proportion of patients have room for added benefit?
    N = 1000
    baseline = populate_N_patients(df_baseline, N)
    added = populate_N_patients(df_added, N)
    room_for_benefit = (baseline['Time'] < tmax).sum() / baseline.shape[0]
    print(room_for_benefit)
    # what proportion of patients had something to add past the first scan time?
    added_something = (added['Time'] > scan_time).sum() / added.shape[0]
    print(added_something)
    print()
    return room_for_benefit * added_something

def import_survival_data(filepath: str, N=5000) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return populate_N_patients(df, N)

def main():
    data_dir = config_dict['data_dir']
    sheet = pd.read_csv(config_dict['metadata_sheet_seed'], 
                        sep='\t', header=0, index_col=None)
    result_sheet = sheet.copy()
    result_sheet.loc[:, "how_many_benefit_from_additivity"] = np.nan

    for i in range(sheet.shape[0]):
        name_a = sheet.at[i, 'Experimental']
        name_b = sheet.at[i, 'Control']
        name_ab = sheet.at[i, 'Combination']
        print(name_ab)

        obs_ab = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')
        obs_exp = pd.read_csv(f'{data_dir}/{name_a}.clean.csv')
        obs_ctrl = pd.read_csv(f'{data_dir}/{name_b}.clean.csv')
        
        scan_a = sheet.at[i, 'Experimental First Scan Time']
        scan_b = sheet.at[i, 'Control First Scan Time']
        
        tmax = set_tmax(obs_exp, obs_ctrl, obs_ab)
        subtracted_drug, scan_time = subtract_which_scan_time(scan_a, scan_b)

        if subtracted_drug == 'a':  # experimental drug
            result = calculate_how_many_benefit_from_additivity(obs_ctrl, obs_exp, scan_time, tmax)
        else:
            result = calculate_how_many_benefit_from_additivity(
                obs_exp, obs_ctrl, scan_time, tmax)
        result_sheet.loc[i, "how_many_benefit_from_additivity"] = result
    
    result_sheet.to_csv(f"{config_dict['table_dir']}/how_many_benefit_from_additivity.csv", index=False)
    
if __name__ == '__main__':
    main()
        


