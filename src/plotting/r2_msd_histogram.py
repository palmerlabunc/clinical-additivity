#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from plot_utils import import_input_data, set_figsize, interpolate
import warnings
from scipy.stats import pearsonr
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

warnings.filterwarnings("ignore")
plt.style.use('env/publication.mplstyle')

def prepare_data_for_r2_histogram(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> pd.DataFrame:
    """Preprocess dataframe to plot R2 hsitogram.

    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe
        data_dir (str): path to clinical trial survival data
        pred_dir (str): path to HSA, additivity prediction data
    
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
        name_ab = r2_df.at[i, 'Combination']

        # import data
        obs = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')
        independent = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max(),
                        additive['Time'].max()]) - 0.1
        obs = obs[obs['Time'] < tmax]
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        time_points = np.linspace(0, tmax, 5000)
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


def calc_total_R2(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> tuple:
    """Calculates R2 across all combinations.

    Returns:
        (float, float): R2 for HSA, R2 for Additivity
    """    
    obs_all = []
    ind_all = []
    add_all = []

    for i in range(cox_df.shape[0]):
        name_a = cox_df.at[i, 'Experimental']
        name_b = cox_df.at[i, 'Control']
        name_ab = cox_df.at[i, 'Combination']

        # import data
        obs = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')
        independent = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max(),
                        additive['Time'].max()]) - 0.1
        obs = obs[obs['Time'] < tmax]
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        time_points = np.linspace(0, tmax, 5000)
        ori = f_obs(time_points)
        ind = f_ind(time_points)
        add = f_add(time_points)
        obs_all += list(ori)
        ind_all += list(ind)
        add_all += list(add)
    r2_ind = r2_score(obs_all, ind_all)
    r2_add = r2_score(obs_all, add_all)

    return r2_ind, r2_add



def prepare_data_for_r_histogram(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> tuple:
    """Preprocess dataframe to plot correlation hsitogram.

    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe
        data_dir (str): path to clinical trial survival data
        pred_dir (str): path to HSA, additivity prediction data

    Returns:
        pd.DataFrame: preprocessed dataframe
        float: observed vs. HSA pearson r
        float: observed vs. additivity pearson r
    """    

    r2_df = cox_df[['Experimental', 'Control',
                    'Combination', 'label', 'Figure', 'Model']]

    r2_df.loc[:, 'r_ind'] = np.nan
    r2_df.loc[:, 'r_add'] = np.nan
    obs_everything = []
    ind_everything = []
    add_everything = []
    for i in range(r2_df.shape[0]):
        name_a = r2_df.at[i, 'Experimental']
        name_b = r2_df.at[i, 'Control']
        name_ab = r2_df.at[i, 'Combination']

        # import data
        obs = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')
        independent = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

        # set tmax
        tmax = np.amin([obs['Time'].max(), independent['Time'].max(),
                        additive['Time'].max()]) - 0.1
        obs = obs[obs['Time'] < tmax]
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]

        f_obs = interpolate(x='Time', y='Survival', df=obs)
        f_ind = interpolate(x='Time', y='Survival', df=independent)
        f_add = interpolate(x='Time', y='Survival', df=additive)

        time_points = np.linspace(0, tmax, 5000)
        ori = f_obs(time_points)
        ind = f_ind(time_points)
        add = f_add(time_points)
        obs_everything += list(ori)
        ind_everything += list(ind)
        add_everything += list(add)
        r_ind, _ = pearsonr(ori, ind)
        r_add, _ = pearsonr(ori, add)

        r2_df.at[i, 'r_ind'] = r_ind
        r2_df.at[i, 'r_add'] = r_add
    
    r2_df.loc[:, 'HSA'] = 'No'
    r2_df.loc[r2_df['Model'].isin(['independent', 'between']), 'HSA'] = 'Yes'
    r2_df.loc[:, 'Additivity'] = 'No'
    r2_df.loc[r2_df['Model'].isin(['additive', 'between']), 'Additivity'] = 'Yes'
    return r2_df, pearsonr(obs_everything, ind_everything), pearsonr(obs_everything, add_everything)


def plot_r2_histogram(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> plt.figure:
    """Generate a plot for R2 histogram.

    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe
        data_dir (str): path to clinical trial survival data
        pred_dir (str): path to HSA, additivity prediction data

    Returns:
        plt.figure: plotted figure
    """    
    r2_df = prepare_data_for_r2_histogram(cox_df, data_dir, pred_dir)
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


def prepare_data_for_msd_histogram(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> pd.DataFrame:
    """Preprocess dataframe to plot MSD (mean signed difference) hsitogram.

    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe
        data_dir (str): path to clinical trial survival data
        pred_dir (str): path to HSA, additivity prediction data

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    error_df = cox_df[['Experimental', 'Control',
                       'Combination', 'label', 'Figure', 'Model']]
    error_df.loc[:, 'error_ind'] = np.nan
    error_df.loc[:, 'error_add'] = np.nan

    for i in range(error_df.shape[0]):
        name_a = error_df.at[i, 'Experimental']
        name_b = error_df.at[i, 'Control']
        name_ab = error_df.at[i, 'Combination']

        # import data
        obs = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv')
        independent = pd.read_csv(
            f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
        additive = pd.read_csv(
            f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

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


def plot_msd_histogram(cox_df: pd.DataFrame, data_dir: str, pred_dir: str) -> plt.figure:
    """Generate a plot for MSD histogram.
    
    Args:
        cox_df (pd.DataFrame): cox_ph_test.csv dataframe
        data_dir (str): path to clinical trial survival data
        pred_dir (str): path to HSA, additivity prediction data
    
    Returns:
        plt.figure: plotted figure
    """
    
    error_df = prepare_data_for_msd_histogram(cox_df, data_dir, pred_dir)
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


def calc_r(cox_df: pd.DataFrame, data_dir: str, pred_dir: str):
    print(calc_total_R2(cox_df))
    r2_df = prepare_data_for_r2_histogram(cox_df, data_dir, pred_dir)
    print(r2_df['r2_ind'].mean(), r2_df['r2_add'].mean())
    df, ind, add = prepare_data_for_r_histogram(cox_df, data_dir, pred_dir)
    print("ind", ind)
    print("add", add)
    print("avg ind", df['r_ind'].mean())
    print(df['r_ind'])
    print("avg add", df['r_add'].mean())
    print("Add", df[df['Additivity'] == 'Yes']['r_add'].mean())
    print("No Add", df[df['Additivity'] == 'No']['r_add'].mean())


def year_by_r2(r2_df):
    author_year = r2_df['Combination'].str.split('_', expand=True)[2]
    r2_df.loc[:, "Publication Year"] = author_year.str[-4:].astype(int)
    fig, ax = plt.subplots(figsize=(3, 1.5))
    sns.scatterplot(x="Publication Year", y="r2_add", data=r2_df, alpha=0.8, ax=ax, s=5)
    ax.set_ylabel('R2')
    return fig

def main():
    config_dict = CONFIG["approved"]
    cox_df = import_input_data()
    data_dir = config_dict['data_dir']
    pred_dir = config_dict['pred_dir']
    fig_dir = config_dict['fig_dir']
    table_dir = config_dict['table_dir']
    r2_df = prepare_data_for_r2_histogram(cox_df, data_dir, pred_dir)
    fig = year_by_r2(r2_df)
    fig.savefig(f'{fig_dir}/year_by_r2.pdf',
                bbox_inches='tight', pad_inches=0.1)
    r2_df.to_csv(f'{table_dir}/r2.csv', index=False)

    r2_fig = plot_r2_histogram(cox_df, data_dir, pred_dir)
    r2_fig.savefig(f'{fig_dir}/r2_histogram.pdf',
                 bbox_inches='tight', pad_inches=0.1)

    msd_fig = plot_msd_histogram(cox_df, data_dir, pred_dir)
    msd_fig.savefig(f'{fig_dir}/msd_histogram.pdf',
                    bbox_inches='tight', pad_inches=0.1)
    r2_ind_all, r2_add_all = calc_total_R2(cox_df, data_dir, pred_dir)
    print("HSA R2 across all trials: ", r2_ind_all)
    print("Additivity R2 across all trials: ", r2_add_all)


if __name__ == '__main__':
    main()
    #calc_r()
