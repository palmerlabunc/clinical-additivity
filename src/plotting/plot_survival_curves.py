import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from plot_utils import import_input_data, get_model_colors
import warnings
import yaml

plt.style.use('env/publication.mplstyle')

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

config_dict = CONFIG['approved']
COMBO_SEED_SHEET = config_dict['metadata_sheet_seed']
COMBO_DATA_DIR = config_dict['data_dir']
PFS_PRED_DIR = config_dict['pred_dir']
FIG_DIR = config_dict['fig_dir']
warnings.filterwarnings("ignore")


def plot_survivals(df_control: pd.DataFrame, df_combo: pd.DataFrame, 
                   df_add: pd.DataFrame, df_ind: pd.DataFrame, ax: plt.axes, 
                   label=None) -> plt.axes:
    """Helper function to plot survival curves. Plots survival on plt.Axes object.

    Args:
        df_control (pd.DataFrame): control arm survival data
        df_combo (pd.DataFrame): combination arm survival data
        df_add (pd.DataFrame): additivity survival data
        df_ind (pd.DataFrame): HSA survival data
        ax (plt.axes): axes to plot on
        label (str, optional): label of the combination. Defaults to None.

    Returns:
        plt.axes: plotted axes
    """    
    ticks = [0, 50, 100]
    # define colors
    color_dict = get_model_colors()
    # set same max time
    tmax = np.amin(
        [df_combo['Time'].max(), df_ind['Time'].max(), df_control['Time'].max()])

    ### plot
    ax.plot(df_control['Time'], df_control['Survival'],
            alpha=1, color=color_dict['control'], linewidth=1)
    ax.plot(df_combo['Time'], df_combo['Survival'], color=color_dict['combo'], linewidth=1)
    ax.plot(df_ind['Time'], df_ind['Survival'], color=color_dict['HSA'], linewidth=1)
    ax.plot(df_add['Time'], df_add['Survival'], color=color_dict['additive'], linewidth=1)

    if label is not None:
        ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                fontweight='bold', va='top', ha='right')
    ax.set_xlabel('')
    ax.set_xlim(0, tmax - 0.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(6))
    ax.axes.xaxis.set_ticklabels([])

    return ax


def plot_additive_survival() -> plt.figure:
    """Generates a figure with all combination survival curves that are only
    consistent with or surpass additivity.
    Returns:
        plt.figure:
    """
    cox_df = import_input_data()
    tmp = cox_df[(cox_df['Figure'] == 'additive') |
                 (cox_df['Figure'] == 'synergy')]
    # sort by cancer types
    tmp = tmp.sort_values(by=['Figure', 'Combination'],
                          ascending=[False, True]).reset_index()

    fig, axes = plt.subplots(4, 5, figsize=(6, 6 * 4/5),
                             subplot_kw=dict(box_aspect=0.7),
                             sharey=True, constrained_layout=True)
    sns.despine()
    flat_axes = axes.flatten()

    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']

        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')

        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(
            i+1), transform=flat_axes[i].transAxes, fontweight='bold')

    for k in range(17, 20):
        flat_axes[k].axes.xaxis.set_ticklabels([])

    return fig


def plot_between_survival() -> plt.figure:
    """Generates a figure with all combination survival curves that are
    consistent with both additivity and HSA.
    Returns:
        plt.figure:
    """
    cox_df = import_input_data()
    tmp = cox_df[(cox_df['Figure'] == 'between')]
    # sort by cancer types
    tmp = tmp.sort_values('Combination').reset_index()

    fig, axes = plt.subplots(2, 5, figsize=(6,  6 * 2/5),
                             subplot_kw=dict(box_aspect=0.7),
                             sharey=True, constrained_layout=True)

    sns.despine()
    flat_axes = axes.flatten()

    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']

        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')

        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(
            i+18), transform=flat_axes[i].transAxes, fontweight='bold')

    return fig


def plot_hsa_survival() -> plt.figure:
    """Generates a figure with all combination survival curves that are
    only consistent with or inferior to HSA.
    Returns:
        plt.figure:
    """
    cox_df = import_input_data()
    tmp = cox_df[(cox_df['Figure'] == 'independent') | (
        cox_df['Figure'] == 'worse than independent')]
    # sort by cancer types
    tmp = tmp.sort_values(['Model', 'Combination'], ascending=[
        True, True]).reset_index()

    fig, axes = plt.subplots(2, 5, figsize=(6, 6 * 2/5),
                             subplot_kw=dict(box_aspect=0.7),
                             sharey=True, constrained_layout=True)
    sns.despine()
    flat_axes = axes.flatten()

    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']

        df_b = pd.read_csv(f'{COMBO_DATA_DIR}/{name_b}.clean.csv')
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv')

        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv')

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(
            i+28), transform=flat_axes[i].transAxes, fontweight='bold')
    return fig


def main():
    additive_fig = plot_additive_survival()
    between_fig = plot_between_survival()
    hsa_fig = plot_hsa_survival()

    additive_fig.savefig(f'{FIG_DIR}/additive_survival_plots.pdf')
    between_fig.savefig(f'{FIG_DIR}/between_survival_plots.pdf')
    hsa_fig.savefig(f'{FIG_DIR}/hsa_survival_plots.pdf')


if __name__ == '__main__':
    main()
