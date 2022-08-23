import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from .plot_utils import import_input_data
import warnings
warnings.filterwarnings("ignore")


def plot_survivals(df_control, df_combo, df_add, df_ind, ax, label=None):
    ticks = [0, 50, 100]
    # define colors
    blue = [i/255 for i in (0, 128, 255)]
    orange = [i/255 for i in (255, 219, 40)]
    red = [i/255 for i in (200, 0, 50)]
    # set same max time
    tmax = np.amin(
        [df_combo['Time'].max(), df_ind['Time'].max(), df_control['Time'].max()])

    ### plot
    ax.plot(df_control['Time'], df_control['Survival'],
            alpha=1, color=orange, linewidth=1)
    ax.plot(df_combo['Time'], df_combo['Survival'], color='k', linewidth=1)
    ax.plot(df_ind['Time'], df_ind['Survival'], color=blue, linewidth=1)
    ax.plot(df_add['Time'], df_add['Survival'], color=red, linewidth=1)

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


def plot_additive_survival():
    indir, cox_df = import_input_data()
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

        path = tmp.at[i, 'Path'] + '/'
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv')
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv')

        independent = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        additive = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(
            i+1), transform=flat_axes[i].transAxes, fontweight='bold')

    for k in range(17, 20):
        flat_axes[k].axes.xaxis.set_ticklabels([])

    return fig


def plot_between_survival():
    indir, cox_df = import_input_data()
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

        path = tmp.at[i, 'Path'] + '/'
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv')
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv')

        independent = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        additive = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(i+18), transform=flat_axes[i].transAxes, fontweight='bold')

    return fig


def plot_hsa_survival():
    indir, cox_df = import_input_data()
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

        path = tmp.at[i, 'Path'] + '/'
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv')
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv')

        independent = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        additive = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))

        plot_survivals(df_b, df_ab, additive, independent, flat_axes[i])
        flat_axes[i].text(0.85, 0.8, str(
            i+28), transform=flat_axes[i].transAxes, fontweight='bold')

    return fig
