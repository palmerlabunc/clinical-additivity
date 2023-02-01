import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COMBO_INPUT_SHEET = CONFIG['metadata_sheet']['combo']
PLACEBO_INPUT_SHEET = CONFIG['metadata_sheet']['placebo']
PHASE3_INPUT_SHEET = CONFIG['metadata_sheet']['all_phase3']

RAW_COMBO_DIR = CONFIG['dir']['raw_combo_data']
RAW_PLACEBO_DIR = CONFIG['dir']['raw_placebo_data']
RAW_PHASE3_DIR = CONFIG['dir']['raw_all_phase3_data']

COMBO_DATA_DIR = CONFIG['dir']['combo_data']
PLACEBO_DATA_DIR = CONFIG['dir']['placebo_data']
PHASE3_DATA_DIR = CONFIG['dir']['all_phase3_data']

FIG_DIR = CONFIG['dir']['figures']

def raw_import(filepath: str) -> pd.DataFrame:
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


def preprocess_survival_data(filepath: str) -> pd.DataFrame:
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


def sanity_check_plot(ori: pd.DataFrame, new: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    ax.plot(ori['Time'],  ori['Survival'], linewidth=0.5)
    ax.plot(new['Time'],  new['Survival'], linewidth=0.5)
    ax.set_ylim(0, 105)
    return ax


def sanity_check_everything():
    indf = pd.read_csv(COMBO_INPUT_SHEET, sep='\t')
    cols = ['Experimental', 'Control', 'Combination']
    fig, axes = plt.subplots(indf.shape[0], 3, figsize=(6, 30))
    for i in range(indf.shape[0]):
        for k in range(len(cols)):
            try:
                name = indf.at[i, cols[k]]
                ori = raw_import(f'{RAW_COMBO_DIR}/{name}.csv')
                ori.columns = ['Time', 'Survival']
                new = preprocess_survival_data(f'{RAW_COMBO_DIR}/{name}.csv')
                axes[i, k] = sanity_check_plot(ori, new, axes[i, k])
            except:
                print(name)
    fig.savefig(f'{FIG_DIR}/preprocess_sanity_check.png')


def preprocess_combinations(indf_filepath: str, raw_dir: str, output_dir: str):
    """_summary_

    Args:
        indf_filepath (str): metadata sheet filepath
        raw_dir (str): raw file directory
        output_dir (str): output directory for cleaned survival curves
    """    
    indf = pd.read_csv(indf_filepath, sep='\t')
    cols = ['Experimental', 'Control', 'Combination']
    for i in range(indf.shape[0]):
        for k in range(len(cols)):
            name = indf.at[i, cols[k]]
            try:
                new = preprocess_survival_data(f'{raw_dir}/{name}.csv')
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError in {name}")
            new.round(5).to_csv(f'{output_dir}/{name}.clean.csv', index=False)


def preprocess_placebo():
    indf = pd.read_csv(PLACEBO_INPUT_SHEET, sep='\t', header=0)
    for i in range(indf.shape[0]):
        name = indf.at[i, 'File prefix']
        new = preprocess_survival_data(f'{RAW_PLACEBO_DIR}/{name}.csv')
        new.round(5).to_csv(f'{PLACEBO_DATA_DIR}/{name}.clean.csv', index=False)


def stand_alone():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, 
                        help='Path to CSV file of the digitized KM curve')
    parser.add_argument('-o', '--output', default=None,
                        help="Output file name.")
    
    args = parser.parse_args()
    cleaned = preprocess_survival_data(args.input)
    if args.output is None:
        tokens = args.input.rsplit('/', 1)
        indir, filename = tokens[0], tokens[1]
        file_prefix = filename.rsplit('.', 1)[0]
        cleaned.to_csv(f'{indir}/{file_prefix}.clean.csv', index=False)
    else:
        cleaned.to_csv(args.output, index=False)


if __name__ == '__main__':
    sanity_check_everything()
    preprocess_combinations(COMBO_INPUT_SHEET, RAW_COMBO_DIR, COMBO_DATA_DIR)
    #preprocess_combinations(PHASE3_INPUT_SHEET, RAW_PHASE3_DIR, PHASE3_DATA_DIR)
    preprocess_placebo()
