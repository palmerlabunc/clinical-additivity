import numpy as np
import pandas as pd
from experimental_correlation import get_all_pairs_95_range
from hsa_additivity_model import predict_both
from utils import interpolate


def diff_12month(high_df, low_df):
    """Note: this results in some combinations having 0 difference
    because the survival curve never reaches 12 months.

    Args:
        high_df (_type_): _description_
        low_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    f_high = interpolate(high_df)
    f_low = interpolate(low_df)
    return abs(f_high(12) - f_low(12))


def diff_average(high_df, low_df):
    tmax = np.max([high_df['Time'].max(), low_df['Time'].max()])
    f_high = interpolate(high_df)
    f_low = interpolate(low_df)
    timepoints = np.linspace(0, tmax, 500)
    return abs(np.mean(f_high(timepoints) - f_low(timepoints)))


def diff_medianPFS(high_df, low_df):
    """Note: this results in some combinations having 0 difference
    because the survival curve never reaches 50%.

    Args:
        high_df (_type_): _description_
        low_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    med = int(high_df.shape[0]/2)
    return abs(high_df.at[med, 'Time'] - low_df.at[med, 'Time'])


def calcualte_uncertainty():
    range95 = get_all_pairs_95_range()
    low_corr, high_corr = range95[0], range95[1]

    indf = pd.read_csv(
        '../data/trials/final_input_list_with_seed.txt', sep='\t')
    
    results = indf[['Experimental', 'Control', 'Combination', 'Corr']]
    
    for i in indf.index:
        print(i)
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        name_ab = indf.at[i, 'Combination']
        path = indf.at[i, 'Path'] + '/'
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value
        if corr != 0.3:
            continue
        
        # random generator seed that results in median of 100 simulations
        seed_ind = indf.at[i, 'ind_median_run']
        seed_add = indf.at[i, 'add_median_run']
        df_a = pd.read_csv(path + name_a + '.clean.csv',
                        header=0, index_col=False)
        df_b = pd.read_csv(path + name_b + '.clean.csv',
                        header=0, index_col=False)
        df_ab = pd.read_csv(path + name_ab + '.clean.csv',
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
        
        ori_corr_hsa, ori_corr_add = predict_both(df_a, df_b, name_a, name_b,
                                                  subtracted, scan_time,
                                                  df_ab=df_ab, rho=0.3,
                                                  seed_ind=seed_ind, seed_add=seed_add)
        low_corr_hsa, low_corr_add = predict_both(df_a, df_b, name_a, name_b, 
                                                    subtracted, scan_time,
                                                    df_ab=df_ab, rho=low_corr, 
                                                    seed_ind=seed_ind, seed_add=seed_add)
        high_corr_hsa, high_corr_add = predict_both(df_a, df_b, name_a, name_b,
                                                    subtracted, scan_time,
                                                    df_ab=df_ab, rho=high_corr,
                                                    seed_ind=seed_ind, seed_add=seed_add)
    
        results.at[i, 'avg_high2low_add'] = diff_average(high_corr_add, low_corr_add)
        results.at[i, 'avg_high2low_HSA'] = diff_average(high_corr_hsa, low_corr_hsa)
        results.at[i, 'avg_high2mid_HSA'] = diff_average(high_corr_hsa, ori_corr_hsa)
        results.at[i, 'avg_high2mid_add'] = diff_average(high_corr_add, ori_corr_add)
        results.at[i, 'avg_low2mid_HSA'] = diff_average(ori_corr_hsa, low_corr_hsa)
        results.at[i, 'avg_low2mid_add'] = diff_average(ori_corr_add, low_corr_add)


    # drop combinations that are not using all pairs
    results = results[results['Corr'] == 0.3]
    results.to_csv(
        "../figures/correlation_uncertainty_range95_avg.txt", sep='\t', index=False)
    return results


if __name__ == '__main__':
    calcualte_uncertainty()
