import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plot_utils import import_input_data, get_model_colors
from scipy.stats import spearmanr
import matplotlib.ticker as plticker
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

plt.style.use('env/publication.mplstyle')

def plot_HR_additivity_HR_combo(cox_df: pd.DataFrame, ax: plt.axes) -> plt.axes:
    additive = cox_df['Model'].isin(['additive', 'between'])
    synergy = cox_df['Model'] == 'synergy'
    cox_df.loc[:, 'Group'] = 'worse than additive'
    cox_df.loc[additive, 'Group'] = 'additive'
    cox_df.loc[synergy, 'Group'] = 'better than additive'

    cox_df.loc[:, 'Additive'] = cox_df['Model'].isin(['additive', 'between'])
    #ci = np.array([cox_df['HR_add'].values - cox_df['HRlower_add'].values,
    #               cox_df['HRupper_add'].values - cox_df['HR_add'].values])
    #ax.errorbar(x=cox_df['HR(combo/control)'],
    #            y=cox_df['HR_add'],
    #            yerr=ci,
    #            color='black',  capsize=3, linestyle='None',
    #            linewidth=1, marker="o", markersize=0, mfc="black", mec="black")
    color_dict = get_model_colors()
    sns.scatterplot(x='HR(combo/control)', y='HR_add',
                    hue='Group', hue_order=['better than additive', 'additive', 'worse than additive'],
                    palette=['violet', color_dict['additive'], color_dict['HSA']],
                    data=cox_df, zorder=0, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('HR(combo vs. control)')
    ax.set_ylabel('HR(addtivity vs. combo)')
    ax.set_xlim(0.2, 1.1)
    ax.set_ylim(0.4, 2)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    x_major = [0.25, 0.5, 1, 1.5]
    y_major = [0.5, 1, 1.5, 2]

    ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.yaxis.set_major_locator(plticker.FixedLocator(y_major))
    ax.yaxis.set_major_formatter(plticker.FixedFormatter(y_major))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    

def main():
    cox_df = import_input_data()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    plot_HR_additivity_HR_combo(cox_df, ax=ax)
    fig_dir = CONFIG['approved']['fig_dir']
    print(spearmanr(np.log(cox_df['HR(combo/control)']), np.log(cox_df['HR_add'])))
    fig.savefig(f'{fig_dir}/HR_additivity-HR_combo_scatterplot.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
