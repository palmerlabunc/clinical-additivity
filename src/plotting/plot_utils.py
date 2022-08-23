import pandas as pd
from scipy.interpolate import interp1d


def interpolate(df, x='Time', y='Survival', kind='zero'):
    return interp1d(df[x], df[y], kind=kind, fill_value='extrapolate')

def import_input_data():
    indir = '../data/PFS_predictions/'
    cox_df = pd.read_csv(indir + 'cox_ph_test.csv', index_col=False)
    # remove supplementary
    cox_df = cox_df[cox_df['Figure'] != 'suppl'].reset_index()
    return indir, cox_df


def import_input_data_include_suppl():
    indir = '../data/PFS_predictions/'
    cox_df = pd.read_csv(indir + 'cox_ph_test.csv', index_col=False)
    return indir, cox_df


def set_figsize(scale, rows, cols):
    subplot_abs_width = 2 * scale  # Both the width and height of each subplot
    # The width of the spacing between subplots
    subplot_abs_spacing_width = 0.2 * scale
    # The height of the spacing between subplots
    subplot_abs_spacing_height = 0.2 * scale
    # The width of the excess space on the left and right of the subplots
    subplot_abs_excess_width = 0.3 * scale
    # The height of the excess space on the top and bottom of the subplots
    subplot_abs_excess_height = 0.3 * scale

    fig_width = (cols * subplot_abs_width) + ((cols-1) *
                                              subplot_abs_spacing_width) + subplot_abs_excess_width
    fig_height = (rows * subplot_abs_width) + ((rows-1) * 
                                                subplot_abs_spacing_height) + subplot_abs_excess_height
    return (fig_width, fig_height)
