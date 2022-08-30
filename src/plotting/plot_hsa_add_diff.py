import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def plot_hsa_add_diff_vs_lognormal(lognorm_df, diff_df):
    """Plot average sigma of the lognormal fits of the monotherapy curves vs.
    hazard ratio (HSA vs. additivity).

    Args:
        lognorm_df (pd.DataFrame): _description_
        diff_df (pd.DataFrame): _description_

    Returns:
        sns.JointGrid: jointgrid plot of regression plot and univariate distributions
    """    
    results = pd.concat([lognorm_df[['sigma_a', 'sigma_b']], diff_df], axis=1)
    results.loc[:, 'sigma_avg'] = np.sqrt((results['sigma_a']**2 + results['sigma_b']**2)/2)
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300, constrained_layout=True)
    sns.despine()
    sns.regplot('sigma_avg', 'HR', data=results, scatter_kws={"s": 7}, ax=ax)
    ax.set_xlabel(r'$\bar{\sigma}$')
    ax.set_ylabel('HR(additivity vs. HSA)')
    ax.set_xticks([1, 1.5, 2])
    ax.set_yticks([0.6, 0.8, 1])

    return fig


def reg_hsa_add_diff_vs_lognormal(lognorm_df, diff_df):
    avg_sigma = np.sqrt((lognorm_df['sigma_a']**2 + lognorm_df['sigma_b']**2)/2)
    reg = linregress(avg_sigma, diff_df['HR'])
    return reg
