import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from scipy.stats import linregress, pearsonr
from .plot_utils import get_model_colors
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

plt.style.use('env/publication.mplstyle')

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


def hsa_add_contribution_stacked_barplot():
    #FIXME This will probably break; need to change file paths
    diff_df = pd.read_csv('../analysis/additivity_HSA_similarity/difference.csv').round(5)
    less_than_HSA = diff_df['Combo - Control'] < diff_df['HSA - Control'] 
    less_than_Add = diff_df['Combo - Control'] < diff_df['Additivity - HSA'] + diff_df['HSA - Control']

    # if less than HSA, plot Combo - Control instead of HSA - Control
    print(diff_df.loc[less_than_HSA, 'Combination'])
    diff_df.loc[less_than_HSA, 'HSA - Control'] = diff_df.loc[less_than_HSA, 'Combo - Control']
    # if more than HSA and less than additivity, plot Combo - HSA instead of Additivity - HSA
    diff_df.loc[less_than_Add, 'Additivity - HSA'] = diff_df.loc[less_than_Add, 'Combo - Control'] - diff_df.loc[less_than_Add, 'HSA - Control']
    diff_df.loc[less_than_HSA, 'Additivity - HSA'] = 0
    diff_df.loc[diff_df['Additivity - HSA'] < 0, 'Additivity - HSA'] = 0
    # if more than additivity,
    diff_df.loc[diff_df['Combo - Additivity'] < 0, 'Combo - Additivity'] = 0
    color_dict = get_model_colors()
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    sns.despine()
    plot_df = diff_df.sort_values('Combo - Control')[['HSA - Control', 'Additivity - HSA', 'Combo - Additivity']]
    print(diff_df.loc[plot_df.index, 'Combination'])
    print(plot_df.mean())
    plot_df.plot.bar(stacked=True, color=[color_dict['HSA'], color_dict['additive'], 'purple'], ax=ax)
    ax.set_ylabel('Normalized Difference \nin Survival (%)')
    ax.set_ylim(0, 50)
    ax.set_xticks([])
    ax.legend(loc='upper left')
    fig.savefig('../figures/normalized_diff_in_survival.pdf')


def hsa_add_contribution_stacked_barplot_test():
    #FIXME This will probably break; need to change file paths
    diff_df = pd.read_csv('../analysis/additivity_HSA_similarity/difference.csv').round(5)
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    sns.despine()
    diff_df.sort_values('Combo - Control')['Combo - Control'].plot.bar(ax=ax)
    ax.set_ylabel('Normalized Difference \nin Survival (%)')
    ax.set_ylim(0, 50)
    ax.set_xticks([])
    ax.legend(loc='upper left')
    fig.savefig('../figures/normalized_diff_in_survival.test.pdf')


if __name__ == '__main__':
    hsa_add_contribution_stacked_barplot()
    #hsa_add_contribution_stacked_barplot_test()
