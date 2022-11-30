import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from plot_utils import import_input_data_include_suppl
import warnings
import sys
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

FIG_DIR = CONFIG['dir']['figures']

warnings.filterwarnings("ignore")

def forest_plot() -> plt.figure:
    """Generate forest plot and return figure.

    Returns:
        plt.figure : figure
    """    
    cox_df = import_input_data_include_suppl()
    
    add_df = cox_df[(cox_df['Model'] == 'additive') |
                    (cox_df['Model'] == 'synergy')]


    add_df = add_df.sort_values(by=['Model', 'Combination'], ascending=[
                                False, True]).reset_index()
    between_df = cox_df[(cox_df['Model'] == 'between')]
    between_df = between_df.sort_values('Combination').reset_index()
    ind_df = cox_df[(cox_df['Model'] == 'independent') | (
        cox_df['Model'] == 'worse than independent')]
    ind_df = ind_df.sort_values(by=['Model', 'Combination'], ascending=[
                                True, True]).reset_index()
    tmp = pd.concat([add_df, between_df, ind_df], axis=0).reset_index(
        drop=True)[::-1]  # reverse order

    fig, axes = plt.subplots(1, 2, figsize=(7, 8), dpi=300, constrained_layout=True)
    sns.despine()

    labels = tmp['label'].values

    # independence
    hr_vals = tmp['HR_ind'].values
    ci = np.array([tmp['HR_ind'].values - tmp['HRlower_ind'].values,
                tmp['HRupper_ind'].values - tmp['HR_ind'].values])
    axes[0].errorbar(x=hr_vals,
                    y=labels,
                    xerr=ci,
                    color='black',  capsize=3, linestyle='None',
                    linewidth=1, marker="o", markersize=4, mfc="black", mec="black")
    axes[0].set_title('HSA')

    # additivity
    hr_vals = tmp['HR_add'].values
    ci = np.array([tmp['HR_add'].values - tmp['HRlower_add'].values,
                tmp['HRupper_add'].values - tmp['HR_add'].values])
    axes[1].errorbar(x=hr_vals,
                    y=[str(i) for i in range(len(hr_vals))],
                    xerr=ci,
                    color='black',  capsize=3, linestyle='None',
                    linewidth=1, marker="o", markersize=4, mfc="black", mec="black")
    axes[1].set_title('Additivity')
    axes[1].axes.yaxis.set_visible(False)
    for ax in axes:
        ax.axvline(x=1, linewidth=0.8, linestyle='--', color='red', alpha=0.5)
        ax.set_xscale('log', basex=2)
        x_major = [0.25, 0.5, 1, 2]
        ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
        ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_formatter(plticker.NullFormatter())
        ax.set_xlim(0.25, 3)

    return fig


def main():
    fig = forest_plot()
    if len(sys.argv) == 1:
        outfile = f'{FIG_DIR}/cox_ph_test.csv'
    else:
        outfile = sys.argv[1]
    fig.savefig(outfile)


if __name__ == '__main__':
    main()
