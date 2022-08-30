import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from plotting.plot_utils import import_input_data


def lognormal_survival(x, mu, sigma):
    y = (1 - norm.cdf((np.log(x) - mu)/sigma))
    return y


def mse(pred, true):
    return np.mean(np.power(pred - true, 2))


def fit_lognormal():
    indir, cox_df = import_input_data()
    lognorm_df = pd.DataFrame(index=cox_df.index, 
                              columns=['mu_a', 'sigma_a', 'mse_a', 'mu_b', 'sigma_b', 'mse_b'])
    for i in range(cox_df.shape[0]):
        name_a = cox_df.at[i, 'Experimental']
        name_b = cox_df.at[i, 'Control']
        path = cox_df.at[i, 'Path'] + '/'

        # import data
        df_a = pd.read_csv(path + name_a + '.clean.csv')
        df_b = pd.read_csv(path + name_b + '.clean.csv')

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
