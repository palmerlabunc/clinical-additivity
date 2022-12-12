import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from plot_utils import get_model_colors, import_input_data_include_suppl
import warnings
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_DATA_DIR = CONFIG['dir']['combo_data']
PFS_PRED_DIR = CONFIG['dir']['PFS_prediction']
FIG_DIR = CONFIG['dir']['figures']

warnings.filterwarnings("ignore")


def set_tmax(df_a, df_b):
    """Set maximum follow-up time.

    Args:
        df_a (pd.DataFrame): survival data of drug A (experimental)
        df_b (pd.DataFrame): survival data of drug B (control)

    Returns:
        float: maximum follow-up time
    """    
    if df_a.at[0, 'Survival'] < 5 and df_b.at[0, 'Survival'] < 5:
        tmax = max(df_a.at[0, 'Time'], df_b.at[0, 'Time'])
    else:
        tmax = min(df_a['Time'].max(), df_b['Time'].max())
    return tmax


def make_label(name):
    """Format file name to '{Cancer type} {Drugs}\n{Author} et al. {Year}' for figure label.

    Args:
        name (str): input file prefix in the format of '{Cancer}_{Drug}_{Author}{Year}_PFS'

    Returns:
        str: formatted label
    """
    tokens = name.split('_')
    cancer = tokens[0]
    drugA, drugB = tokens[1].split('-')
    author, year = tokens[2][:-4], tokens[2][-4:]
    return f"{cancer} {drugA}+{drugB}\n({author} et al. {year})"


def plot_survivals(df_control, df_exp, df_combo, df_add, df_ind, ax, label=None):
    """Helper function to plot survival curves. Plots survival on Axes object.

    Args:
        df_control (pd.DataFrame): control drug survival data
        df_exp (pd.DataFrame): exerimental drug survival data
        df_combo (pd.DataFrame): combination survival data
        df_add (pd.DataFrame): additivity survival data
        df_ind (pd.DataFrame): HSA survival data
        ax (plt.axes): axes to plot the survival curves on
        label (str, optional): _description_. Defaults to None.

    Returns:
        plt.axes: plotted axes
    """    
    ticks = [0, 50, 100]
    color_dict = get_model_colors()
    # set same max time
    tmax = set_tmax(df_exp, df_control)

    ### plot
    ax.plot(df_control['Time'], df_control['Survival'],
            alpha=0.5, color=color_dict['control'], linewidth=1)
    ax.plot(df_exp['Time'], df_exp['Survival'],
            alpha=0.5, color=color_dict['experimental'], linewidth=1)
    ax.plot(df_combo['Time'], df_combo['Survival'], color=color_dict['combo'], linewidth=1.5)
    ax.plot(df_ind['Time'], df_ind['Survival'], color=color_dict['HSA'], linewidth=1.5)
    ax.plot(df_add['Time'], df_add['Survival'], color=color_dict['additive'], linewidth=1.5)
    if label is not None:
        ax.set_title(make_label(label))
    ax.set_xlabel('')
    ax.set_xlim(0, tmax - 0.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(6))
    ax.axes.xaxis.set_ticklabels([])

    return ax


def plot_additivity_suppl():
    cox_df = import_input_data_include_suppl()
    tmp = cox_df[(cox_df['Model'] == 'additive') |
                 (cox_df['Model'] == 'synergy')]

    # sort by cancer types
    tmp = tmp.sort_values(by=['Model', 'Combination'],
                          ascending=[False, True]).reset_index()

    rows, cols = 6, 3
    fig, axes = plt.subplots(rows, cols, sharey=True, 
                             figsize=(7, 10), subplot_kw=dict(box_aspect=0.5), dpi=300)
    sns.despine()
    flat_axes = axes.flatten()

    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']

        # import data
        obs_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')
        obs_exp = pd.read_csv(f'{COMBO_DATA_DIR}/{name_a}.clean.csv')
        obs_ctrl = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        plot_survivals(obs_ctrl, obs_exp, obs_ab, additive,
                    independent, flat_axes[i], label=name_ab)

    return fig


def plot_between_hsa_suppl():
    cox_df = import_input_data_include_suppl()
    tmp1 = cox_df[(cox_df['Model'] == 'between')]
    # sort by cancer types
    tmp1 = tmp1.sort_values('Combination').reset_index(drop=True)

    tmp2 = cox_df[(cox_df['Model'] == 'independent') | (
        cox_df['Model'] == 'worse than independent')]
    # sort by cancer types
    tmp2 = tmp2.sort_values(['Model', 'Combination'], ascending=[
                            True, True]).reset_index(drop=True)
    tmp = pd.concat([tmp1, tmp2], axis=0).reset_index(drop=True)

    rows, cols = 8, 3

    fig, axes = plt.subplots(rows, cols, sharey=True, figsize=(7, 13), 
                             subplot_kw=dict(box_aspect=0.5), dpi=300)
    sns.despine()
    flat_axes = axes.flatten()

    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']

        # import data
        obs_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')
        obs_exp = pd.read_csv(f'{COMBO_DATA_DIR}/{name_a}.clean.csv')
        obs_ctrl = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        independent = pd.read_csv(f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        plot_survivals(obs_ctrl, obs_exp, obs_ab, additive,
                    independent, flat_axes[i], label=name_ab)

    return fig


def main():
    fig_add = plot_additivity_suppl()
    fig_btn = plot_between_hsa_suppl()

    fig_add.savefig(f'{FIG_DIR}/suppl_additive_survival_plots.pdf',
                    bbox_inches='tight', pad_inches=0.1)
    fig_btn.savefig(f'{FIG_DIR}/suppl_between_hsa_survival_plots.pdf',
                    bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()