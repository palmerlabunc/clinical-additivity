import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from .plot_utils import get_model_colors, import_input_data

plt.style.use('env/publication.mplstyle')

def get_observed_survival(cox_df, i):
    """Retreive observed survival curves for the combination, experimental,
    and control arms.

    Args:
        cox_df (pd.DataFrame): result data
        i (int): index
    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: observed combination, 
                                                  experimental, and control arms
    """    
    name_a = cox_df.at[i, 'Experimental']
    name_b = cox_df.at[i, 'Control']
    name_ab = cox_df.at[i, 'Combination']
    path = cox_df.at[i, 'Path'] + '/'
    
    obs_ab = pd.read_csv(path + name_ab + '.clean.csv')
    obs_exp = pd.read_csv(path + name_a + '.clean.csv')
    obs_ctrl = pd.read_csv(path + name_b + '.clean.csv')
    
    return (obs_ab, obs_exp, obs_ctrl)


def plot_abema_vs_palbo():
    """Plot Abemaciclib + NSAI vs. Palbociclib + NSAI survival curves.

    Returns:
        plt.figure: plotted figure
    """    
    _, cox_df = import_input_data()
    abema_nsai_idx = 1
    palbo_nsai_idx = 7
    color_dict = get_model_colors()
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.despine()

    obs_ab, obs_exp, obs_ctrl = get_observed_survival(cox_df, abema_nsai_idx)
    ax.plot(obs_ctrl['Time'], obs_ctrl['Survival'],
            alpha=0.8, color=color_dict['control'], linewidth=1, linestyle='--')
    ax.plot(obs_exp['Time'], obs_exp['Survival'],
            alpha=0.8, color=color_dict['experimental'], linewidth=1, linestyle='--')
    ax.plot(obs_ab['Time'], obs_ab['Survival'],
            alpha=0.8, color='purple', linewidth=1, linestyle='--')
    
    obs_ab, obs_exp, obs_ctrl = get_observed_survival(cox_df, palbo_nsai_idx)
    ax.plot(obs_ab['Time'], obs_ab['Survival'],
            alpha=0.8, color='purple', linewidth=1, label='Combination')
    ax.plot(obs_ctrl['Time'], obs_ctrl['Survival'],
            alpha=0.8, color=color_dict['control'], linewidth=1, label='Letrozole/Anastrozole')
    ax.plot(obs_exp['Time'], obs_exp['Survival'],
            alpha=0.8, color=color_dict['experimental'], linewidth=1, label='CDK4/6i')


    ax.plot([0, 0], [1, 0], linewidth=1, color='k', label='Palbociclib')
    ax.plot([0, 0], [1, 0], linewidth=1,
            linestyle='--', color='k', label='Abemaciclib')
    
    yticks = [0, 50, 100]
    xticks = list(range(0, 31, 6))
    ax.legend(bbox_to_anchor=(1.01, 1))
    ax.set_xlabel('Time (months)')
    ax.set_xlim(0)
    ax.set_ylabel('PFS (%)')
    ax.set_ylim(0, 105)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    return fig
