import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import import_input_data, set_figsize, interpolate
import warnings
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_SEED_SHEET = CONFIG['metadata_sheet']['combo_seed']
COMBO_DATA_DIR = CONFIG['dir']['combo_data']
PFS_PRED_DIR = CONFIG['dir']['PFS_prediction']
FIG_DIR = CONFIG['dir']['figures']

warnings.filterwarnings("ignore")


def prepare_data_for_qqplot(cox_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for plotting.
    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv file

    Returns:
        pd.DataFrame: dataframe for plotting
    """    
    tmp = cox_df[['Path', 'Experimental', 'Control',
                  'Combination', 'Figure', 'Model']]

    tmp.loc[:, 'ind_label'] = tmp['Combination'].str.split('_', expand=True)[0]
    tmp.loc[:, 'add_label'] = tmp['Combination'].str.split('_', expand=True)[0]

    tmp.loc[~tmp['Model'].isin(['independent', 'between']), 'ind_label'] = 'Fits to other model'
    tmp.loc[~tmp['Model'].isin(['additive', 'between']), 'add_label'] = 'Fits to other model'
    return tmp


def plot_all_survival_qqplot() -> plt.figure:
    """Plot QQ-plot (expected vs. observed) of all PFS curves in the dataset.
    
    Returns:
        plt.figure : figure object
    """    
    cox_df = import_input_data()
    tmp = prepare_data_for_qqplot(cox_df)

    custom_dict = {'Fits to other model': 0, 'Breast': 1, 'Cervical': 2, 
                   'Colorectal': 3, 'Gastric': 4, 'HeadNeck': 5,
                   'Leukemia': 6, 'Lung': 7, 'Melanoma': 8, 'Mesothelioma': 9, 
                   'Myeloma': 10, 'Ovarian': 11, 'Pancreatic': 12, 'Prostate': 13}

    pal = np.array(['lightgray']  + list(sns.color_palette("hls", 13)), dtype='object')

    cols = 2 
    fig_width, fig_height = set_figsize(1, 1, cols)

    fig, axes = plt.subplots(1, cols, sharey=True, figsize=(fig_width, fig_height), subplot_kw=dict(box_aspect=1), dpi=300)
    sns.despine()
    ticks = [0, 50, 100]

    for i in range(2):
        axes[i].plot([0, 100], [0, 100], color='k')
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_yticks(ticks)
        axes[i].set_xticks(ticks)

    for i in tmp.sort_values('ind_label', key=lambda x: x.map(custom_dict)).index:
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        path = tmp.at[i, 'Path'] + '/'
        ind_color = pal[custom_dict[tmp.at[i, 'ind_label']]]

        # import data
        obs = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv')
        independent = pd.read_csv(indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max()]) - 0.1

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)

        time_points = np.linspace(0, tmax, 100)
        axes[0].plot(f_ind(time_points), f_obs(time_points),
                    lw=0.7, alpha=0.7, c=ind_color)

    for i in tmp.sort_values('add_label', key=lambda x: x.map(custom_dict)).index:
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        path = tmp.at[i, 'Path'] + '/'
        add_color = pal[custom_dict[tmp.at[i, 'add_label']]]
        # import data
        obs = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv')
        additive = pd.read_csv(indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))
        # set tmax
        tmax = np.amin([obs['Time'].max(), additive['Time'].max()]) - 0.1

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        time_points = np.linspace(0, tmax, 100)
        axes[1].plot(f_add(time_points), f_obs(time_points),
                    lw=0.7, alpha=0.7, c=add_color)

    axes[0].set_ylabel('Observed PFS (%)')
    axes[0].set_xlabel('Expected PFS (%)')

    return fig


def qqplot_legends() -> plt.figure:
    """Generate legend figure for the QQ-plot.
    
    Returns:
        plt.figure: legend figure
    """    
    custom_dict = {'Fits to other model': 0, 'Breast': 1, 'Cervical': 2,
                   'Colorectal': 3, 'Gastric': 4, 'HeadNeck': 5,
                   'Leukemia': 6, 'Lung': 7, 'Melanoma': 8, 'Mesothelioma': 9,
                   'Myeloma': 10, 'Ovarian': 11, 'Pancreatic': 12, 'Prostate': 13}
    pal = np.array(['lightgray'] +
                   list(sns.color_palette("hls", 13)), dtype='object')
    
    # figure legend
    fig, ax = plt.subplots()
    legendFig = plt.figure("Legend plot")

    for key in custom_dict.keys():
        val = custom_dict[key]
        ax.plot(val, val, c=pal[val], label=key)

    legendFig.legend(ax.get_legend_handles_labels()[
                    0], ax.get_legend_handles_labels()[1], ncol=3)

    return legendFig
