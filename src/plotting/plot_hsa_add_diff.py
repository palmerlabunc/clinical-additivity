import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from scipy.stats import linregress, pearsonr


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
    sns.regplot(x=results['sigma_avg'], y=np.log2(results['HR']), scatter_kws={"s": 7}, ax=ax)
    ax.set_xlabel(r'$\bar{\sigma}$')
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_ylabel('HR(HSA vs. Additivity)')
    ax.set_ylim(np.log2(0.9), np.log2(2))
    y_major_ticklocs = np.log2(np.array([1, 1.5, 2]))
    y_major_ticklabels = [1, 1.5, 2]
    ax.yaxis.set_major_locator(plticker.FixedLocator(y_major_ticklocs))
    ax.yaxis.set_major_formatter(plticker.FixedFormatter(y_major_ticklabels))
    ax.yaxis.set_minor_locator(plticker.FixedLocator(np.log2(np.arange(0.9, 2, 0.1))))
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())

    return fig


def reg_hsa_add_diff_vs_lognormal(lognorm_df, diff_df):
    """Perform linear regression between average standard deviation
    of log-normal fit vs. HR(additivity vs. HSA).

    Args:
        lognorm_df (pd.DataFrame): log-normal parameters of combinations
        diff_df (pd.DataFrame): HSA additivity difference

    Returns:
        LinregressResult: contains slope, intercept, rvalue, pvalue
    """    
    avg_sigma = np.sqrt((lognorm_df['sigma_a']**2 + lognorm_df['sigma_b']**2)/2)
    reg = linregress(avg_sigma, diff_df['HR'])
    return reg


def corr_hsa_add_diff_vs_lognormal(lognorm_df, diff_df):
    """Perform Pearson correlation between average standard deviation
    of log-normal fit vs. HR(additivity vs. HSA).

    Args:
        lognorm_df (pd.DataFrame): log-normal parameters of combinations
        diff_df (pd.DataFrame): HSA additivity difference

    Returns:
        LinregressResult: contains slope, intercept, rvalue, pvalue
    """
    avg_sigma = np.sqrt(
        (lognorm_df['sigma_a']**2 + lognorm_df['sigma_b']**2)/2)
    r, p = pearsonr(avg_sigma, np.log(diff_df['HR']))
    return r, p
