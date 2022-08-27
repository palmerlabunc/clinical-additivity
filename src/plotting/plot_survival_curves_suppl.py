import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from .plot_utils import import_input_data, get_model_colors, set_figsize
import warnings
warnings.filterwarnings("ignore")


def set_tmax(df_a, df_b, df_ab):
    if df_a.at[0, 'Survival'] < 5 and df_b.at[0, 'Survival'] < 5:
        tmax = max(df_a.at[0, 'Time'], df_b.at[0, 'Time'])
    else:
        tmax = min(df_a['Time'].max(), df_b['Time'].max())
    return tmax


def make_label(name):
    tokens = name.split('_')
    cancer = tokens[0]
    drugA, drugB = tokens[1].split('-')
    author, year = tokens[2][:-4], tokens[2][-4:]
    return f"{cancer} {drugA}+{drugB}\n({author} et al. {year})"


def plot_survivals(df_control, df_exp, df_combo, df_add, df_ind, ax, label=None):
    ticks = [0, 50, 100]
    color_dict = get_model_colors()
    # set same max time
    tmax = set_tmax(df_exp, df_control, df_combo)

    ### plot
    ax.plot(df_control['Time'], df_control['Survival'],
            alpha=0.5, color=color_dict['control'], linewidth=1)
    ax.plot(df_exp['Time'], df_exp['Survival'],
            alpha=0.5, color=color_dict['experimental'], linewidth=1)
    ax.plot(df_combo['Time'], df_combo['Survival'], color=color_dict['combo'], linewidth=1.5)
    ax.plot(df_ind['Time'], df_ind['Survival'], color=color_dict['HSA'], linewidth=1.5)
    ax.plot(df_add['Time'], df_add['Survival'], color=color_dict['additive'], linewidth=1.5)

    ax.set_title(make_label(label))
    ax.set_xlabel('')
    ax.set_xlim(0, tmax - 0.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(6))
    ax.axes.xaxis.set_ticklabels([])

    return ax


def plot_additivity_suppl():
    indir, cox_df = import_input_data()
    rows, cols = 6, 3
    fig, axes = plt.subplots(rows, cols, sharey=True, 
                             figsize=(8, 4), subplot_kw=dict(box_aspect=0.5), dpi=300)
    
    tmp = cox_df[(cox_df['Model'] == 'additive') |
                 (cox_df['Model'] == 'synergy')]

    # sort by cancer types
    tmp = tmp.sort_values(by=['Model', 'Combination'],
                        ascending=[False, True]).reset_index()
    
    
