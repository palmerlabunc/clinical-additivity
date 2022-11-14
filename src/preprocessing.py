import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import date

def raw_import(filepath):
    with open(filepath, 'r') as f:
        cols = len(f.readline().split(','))
    if cols == 2:
        df = pd.read_csv(filepath, header=0)
        # change column names
        if not ('Time' in df.columns and 'Survival' in df.columns):
            df.columns = ['Time', 'Survival']
        # if survival is in 0-1 scale, convert to 0-100
        if df['Survival'].max() <= 1.1:
            df.loc[:, 'Survival'] = df['Survival'] * 100
    elif cols == 1:
        # TODO remove repeating points at the end of the tail
        df = pd.read_csv(filepath, sep=',', header=None, names=['Time'])
        df = df.sort_values('Time', ascending=False).reset_index(drop=True)
        df.loc[:, 'Survival'] = np.linspace(0, 100, num=df.shape[0])
    return df


def preprocess_survival_data(filepath):
    """ Import survival data either having two columns (time, survival) or one
    column (time)

    Args:
        filepath (str): path to survival data file

    Returns:
        pd.DataFrame: returned data frame
    """
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'r') as f:
        tokens = f.readline().strip().split(',')
        cols = len(tokens)
    if cols == 2:
        df = pd.read_csv(filepath, header=0)
        # change column names
        if not ('Time' in df.columns and 'Survival' in df.columns):
            df.columns = ['Time', 'Survival']
        # if survival is in 0-1 scale, convert to 0-100
        if df['Survival'].max() <= 1.1:
            df.loc[:, 'Survival'] = df['Survival'] * 100

    elif cols == 1:
        # TODO remove repeating points at the end of the tail
        df = pd.read_csv(filepath, sep=',', header=None, names=['Time'])
        df = df.sort_values('Time', ascending=False).reset_index(drop=True)
        df.loc[:, 'Survival'] = np.linspace(0, 100, num=df.shape[0])

    ### Clean up
    # normalize everything to [0, 100]
    df.loc[:, 'Survival'] = 100 * df['Survival'] / df['Survival'].max()
    df.loc[df['Survival'] < 0, 'Survival'] = 0
    df.loc[df['Time'] <= 0, 'Time'] = 0.00001

    # make sure survival is in increasing order
    if df.iat[-1, 1] < df.iat[0, 1]:
        df = df.sort_values(['Survival'], ascending=True).drop_duplicates()
        df = df.reset_index(drop=True)

    # enforce monotinicity
    df.loc[:, 'Survival'] = np.maximum.accumulate(
        df['Survival'].values)  # monotonic increasing
    df.loc[:, 'Time'] = np.minimum.accumulate(
        df['Time'].values)  # monotonic decreasing
    return df


def sanity_check_plot(ori, new, ax):
    ax.plot(ori['Time'],  ori['Survival'], linewidth=0.5)
    ax.plot(new['Time'],  new['Survival'], linewidth=0.5)
    ax.set_ylim(0, 105)
    return ax


def sanity_check_everything():
    OUTDIR = '../analysis/preprocessing/{}/'.format(date.today())
    new_directory = Path(OUTDIR)
    new_directory.mkdir(parents=True, exist_ok=True)
    indf = pd.read_csv('../data/trials/final_input_list.txt', sep='\t')
    cols = ['Experimental', 'Control', 'Combination']
    fig, axes = plt.subplots(indf.shape[0], 3, figsize=(6, 30))
    for i in range(indf.shape[0]):
        path = indf.at[i, 'Path'] + '/'
        for k in range(len(cols)):
            try:
                name = indf.at[i, cols[k]]
                ori = raw_import(path + name + '.csv')
                ori.columns = ['Time', 'Survival']
                new = preprocess_survival_data(path + name + '.csv')
                axes[i, k] = sanity_check_plot(ori, new, axes[i, k])
            except:
                print(name)
    fig.savefig(OUTDIR + 'sanity_check.png')


def preprocess_everything():
    indf = pd.read_csv('../data/trials/final_input_list.txt', sep='\t')
    cols = ['Experimental', 'Control', 'Combination']
    for i in range(indf.shape[0]):
        path = indf.at[i, 'Path'] + '/'
        for k in range(len(cols)):
            name = indf.at[i, cols[k]]
            new = preprocess_survival_data(path + name + '.csv')
            new.round(5).to_csv(path + name + '.clean.csv', index=False)


def preprocess_placebo():
    indf = pd.read_csv('../data/placebo/placebo_input_list.txt', sep='\t', header=0)
    for i in range(indf.shape[0]):
        path = indf.at[i, 'Path'] + '/'
        name = indf.at[i, 'File prefix']
        print(name)
        new = preprocess_survival_data(path + name + '.csv')
        new.round(5).to_csv(path + name + '.clean.csv', index=False)


def main():
    sanity_check_everything()
    preprocess_everything()
    preprocess_placebo()


if __name__ == '__main__':
    main()
