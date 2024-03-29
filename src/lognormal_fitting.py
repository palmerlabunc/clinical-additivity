import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from plotting.plot_utils import import_input_data
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_DATA_DIR = CONFIG['approved']['data_dir']


def lognormal_survival(x, mu, sigma):
    """Compute log-normal survival distribution/

    Args:
        x (np.ndarray): x values
        mu (float): mean of the log-normal distribution
        sigma (flaot): standard deviation of the log-normal distribution

    Returns:
        np.ndarray: 1-D array of survival distribution (y values)
    """    
    y = (1 - norm.cdf((np.log(x) - mu)/sigma))
    return y


def mse(pred, true):
    """Compute mean squared error between pred and true.

    Args:
        pred (np.ndarray): 1-D array of prediction values
        true (np.ndarray): 1-D array of true values

    Returns:
        float: mean squared error
    """    
    return np.mean(np.power(pred - true, 2))


def fit_lognormal():
    """Fit log-normal distribution to each monotherapy survival curves.

    Returns:
        pd.DataFrame: fitted parameters
    """    
    cox_df = import_input_data()
    lognorm_df = pd.DataFrame(index=cox_df.index, 
                              columns=['mu_a', 'sigma_a', 'mse_a', 'mu_b', 'sigma_b', 'mse_b'])
    for i in range(cox_df.shape[0]):
        name_a = cox_df.at[i, 'Experimental']
        name_b = cox_df.at[i, 'Control']

        # import data
        df_a = pd.read_csv(f'{COMBO_DATA_DIR}/{name_a}.clean.csv')
        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')

        popt_a, cov_a = curve_fit(
            lognormal_survival, df_a['Time'], df_a['Survival']/100)
        popt_b, cov_b = curve_fit(
            lognormal_survival, df_b['Time'], df_b['Survival']/100)

        mu_a, sigma_a = popt_a
        mu_b, sigma_b = popt_b

        lognorm_df.at[i, 'mu_a'] = mu_a
        lognorm_df.at[i, 'sigma_a'] = sigma_a
        lognorm_df.at[i, 'mu_b'] = mu_b
        lognorm_df.at[i, 'sigma_b'] = sigma_b
        lognorm_df.at[i, 'mse_a'] = mse(lognormal_survival(
            df_a['Time'], mu_a, sigma_a), df_a['Survival']/100)
        lognorm_df.at[i, 'mse_b'] = mse(lognormal_survival(
            df_b['Time'], mu_b, sigma_b), df_b['Survival']/100)
    
    return lognorm_df.astype(np.float64)
