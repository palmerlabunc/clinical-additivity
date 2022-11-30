import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
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


def prepare_data_for_r2_histogram(cox_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe to plot R2 hsitogram.

    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """    

    r2_df = cox_df[['Experimental', 'Control',
                    'Combination', 'label', 'Figure', 'Model']]

    r2_df.loc[:, 'r2_ind'] = np.nan
    r2_df.loc[:, 'r2_add'] = np.nan


    for i in range(r2_df.shape[0]):
        name_a = r2_df.at[i, 'Experimental']
        name_b = r2_df.at[i, 'Control']

        path = r2_df.at[i, 'Path'] + '/'
        # import data
        obs = pd.read_csv(path + r2_df.at[i, 'Combination'] + '.clean.csv')
        independent = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        additive = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))

        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max(),
                        additive['Time'].max()]) - 0.1
        obs = obs[obs['Time'] < tmax]
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        time_points = np.linspace(0, tmax, 100)
        ori = f_obs(time_points)
        ind = f_ind(time_points)
        add = f_add(time_points)
        r2_ind = r2_score(ori, ind)
        r2_add = r2_score(ori, add)
        # convert r2 < 0 to r2=0 for plotting purposes
        if r2_ind < 0:
            r2_ind = 0
        if r2_add < 0:
            r2_add = 0
        r2_df.at[i, 'r2_ind'] = r2_ind
        r2_df.at[i, 'r2_add'] = r2_add
    
    r2_df.loc[:, 'HSA'] = 'No'
    r2_df.loc[r2_df['Model'].isin(['independent', 'between']), 'HSA'] = 'Yes'
    r2_df.loc[:, 'Additivity'] = 'No'
    r2_df.loc[r2_df['Model'].isin(['additive', 'between']), 'Additivity'] = 'Yes'
    
    return r2_df


def plot_r2_histogram():
    """Generate a plot for R2 histogram.

    Returns:
        plt.figure: plotted figure
    """    
    indir, cox_df = import_input_data()
    r2_df = prepare_data_for_r2_histogram(indir, cox_df)
    cols = 2
    fig_width, fig_height = set_figsize(1, 1, cols)
    fig, axes = plt.subplots(1, cols, sharey=True, figsize=(
        fig_width, fig_height), subplot_kw=dict(box_aspect=0.5), dpi=300)
    sns.despine()
    sns.histplot(data=r2_df, multiple='stack', x='r2_ind', hue='HSA',
                 hue_order=['No', 'Yes'],
                 palette={'Yes': [i/255 for i in (0, 128, 255)], 'No': 'lightgray'},
                 binwidth=0.05, binrange=(0, 1), ax=axes[0])

    sns.histplot(data=r2_df, multiple='stack', x='r2_add', hue='Additivity',
                 hue_order=['No', 'Yes'],
                 palette={'Yes': [i/255 for i in (200, 0, 50)], 'No': 'lightgray'}, edgecolor='k',
                 binwidth=0.05, binrange=(0, 1), ax=axes[1])
    xticks = [0, 0.5, 1]
    yticks = [0, 10, 20]
    for i in range(2):
        axes[i].set_xticks(xticks)
        axes[i].set_yticks(yticks)
    axes[0].set_xlabel('R2')
    
    return fig

def prepare_data_for_msd_histogram(indir, cox_df):
    """Preprocess dataframe to plot MSD (mean signed difference) hsitogram.

    Args:
        indir (str): input directory path
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    error_df = cox_df[['Path', 'Experimental', 'Control',
                       'Combination', 'label', 'Figure', 'Model']]
    error_df.loc[:, 'error_ind'] = np.nan
    error_df.loc[:, 'error_add'] = np.nan

    for i in range(error_df.shape[0]):
        name_a = error_df.at[i, 'Experimental']
        name_b = error_df.at[i, 'Control']

        path = error_df.at[i, 'Path'] + '/'
        # import data
        obs = pd.read_csv(path + error_df.at[i, 'Combination'] + '.clean.csv')
        independent = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b))
        additive = pd.read_csv(
            indir + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b))

        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max(),
                        additive['Time'].max()]) - 0.1
        obs = obs[obs['Time'] < tmax]
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        n = 100
        time_points = np.linspace(0, tmax, n)
        ori = f_obs(time_points)
        ind = f_ind(time_points)
        add = f_add(time_points)

        error_ind = np.sum(ori - ind)/n
        error_add = np.sum(ori - add)/n

        error_df.at[i, 'error_ind'] = error_ind
        error_df.at[i, 'error_add'] = error_add
    
    error_df.loc[:, 'HSA'] = 'No'
    error_df.loc[error_df['Model'].isin(['independent', 'between']), 'HSA'] = 'Yes'
    error_df.loc[:, 'Additivity'] = 'No'
    error_df.loc[error_df['Model'].isin(
        ['additive', 'between']), 'Additivity'] = 'Yes'
    return error_df


def plot_msd_histogram():
    """Generate a plot for MSD histogram.

    Returns:
        plt.figure: plotted figure
    """
    indir, cox_df = import_input_data()
    error_df = prepare_data_for_msd_histogram(indir, cox_df)
    cols = 2
    fig_width, fig_height = set_figsize(1, 1, cols)

    fig, axes = plt.subplots(1, cols, sharey=True, figsize=(
        fig_width, fig_height), subplot_kw=dict(box_aspect=0.5), dpi=300)
    sns.despine()

    sns.histplot(data=error_df, multiple='stack', x='error_ind', hue='HSA',
                 hue_order=['No', 'Yes'],
                 palette={'Yes': [i/255 for i in (0, 128, 255)], 'No': 'lightgray'},
                 binwidth=2, binrange=(-15, 15), ax=axes[0])

    sns.histplot(data=error_df, multiple='stack', x='error_add', hue='Additivity',
                 hue_order=['No', 'Yes'],
                 palette={'Yes': [i/255 for i in (200, 0, 50)], 'No': 'lightgray'}, edgecolor='k',
                 binwidth=2, binrange=(-15, 15), ax=axes[1])

    xticks = [-10, 0, 10]
    yticks = [0, 5, 10]

    for i in range(2):
        axes[i].get_legend().remove()
        axes[i].set_xticks(xticks)
        axes[i].set_yticks(yticks)
    axes[0].set_xlabel('Mean Signed Difference (%)')

    return fig
