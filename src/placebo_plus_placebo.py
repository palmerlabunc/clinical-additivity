from pathlib import Path
import pandas as pd
from hsa_additivity_model import predict_both



def placebo_plus_pacebo():
    placebo_input = pd.read_csv('../data/placebo/placebo_input_list.txt', 
                                sep='\t', header=0)
    for i in range(placebo_input.shape[0]):
        placebo_path = placebo_input.at[i, 'Path'] + '/'
        file_prefix = placebo_input.at[i, 'File prefix']
        placebo = pd.read_csv(placebo_path + file_prefix + '.clean.csv')
        scan_time = placebo_input.at[i, 'First scan time (months)']
        for rho in [0.3, 0.6, 1]:
            hsa, add = predict_both(placebo, placebo, 'Placebo_A', 'Placebo_B', 'a', 
                                    scan_time, rho=rho)
            hsa.round(5).to_csv(f'../analysis/placebo_plus_placebo/{file_prefix}_hsa_{rho}.csv',
                                index=False)
            add.round(5).to_csv(f'../analysis/placebo_plus_placebo/{file_prefix}_add_{rho}.csv',
                                index=False)
                

if __name__ == '__main__':
    outdir = '../analysis/placebo_plus_placebo/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)
    placebo_plus_pacebo()