import pandas as pd
from scipy.interpolate import interp1d
import yaml
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

COX_RESULT = CONFIG['approved']['cox_result']


def interpolate(df, x='Time', y='Survival', kind='zero'):
    """Wrapper function for scipy.interpolate.interp1d.

    Args:
        df (pd.DataFrame): data to make interpolation.
        x (str, optional): column name to use as x values. Defaults to 'Time'.
        y (str, optional): column name to use as y values. Defaults to 'Survival'.
        kind (str, optional): Specifies the kind of interpolation. 
                              Feeds in to kind argument of interp1d function.. Defaults to 'zero'.

    Returns:
        _type_: _description_
    """    
    return interp1d(df[x], df[y], kind=kind, fill_value='extrapolate')


def import_input_data() -> pd.DataFrame:
    """Import input data excluding supplementary combinations.

    Returns:
        pd.DataFrame: a tuple of input directory path and input data
    """
    cox_df = pd.read_csv(COX_RESULT, index_col=False)
    # remove supplementary
    cox_df = cox_df[cox_df['Figure'] != 'suppl'].reset_index()
    return cox_df


def import_input_data_include_suppl() -> pd.DataFrame:
    """Import input data including supplementary combinations.

    Returns:
        pd.DataFrame: a tuple of input directory path and input data
    """
    cox_df = pd.read_csv(COX_RESULT, index_col=False)
    return cox_df


def set_figsize(scale: float, rows: int, cols: int, 
                spacing_width_scale=0.2, spacing_height_scale=0.2):
    """Set figure size.

    Args:
        scale (float): scale
        rows (int): number of rows in a figure
        cols (int): number of columns in a figure

    Returns:
        (float, float): (width, height) of the figure
    """    
    subplot_abs_width = 2 * scale  # Both the width and height of each subplot
    # The width of the spacing between subplots
    subplot_abs_spacing_width = spacing_width_scale * scale
    # The height of the spacing between subplots
    subplot_abs_spacing_height = spacing_height_scale * scale
    # The width of the excess space on the left and right of the subplots
    subplot_abs_excess_width = 0.3 * scale
    # The height of the excess space on the top and bottom of the subplots
    subplot_abs_excess_height = 0.3 * scale

    fig_width = (cols * subplot_abs_width) + ((cols-1) *
                                              subplot_abs_spacing_width) + subplot_abs_excess_width
    fig_height = (rows * subplot_abs_width) + ((rows-1) * 
                                                subplot_abs_spacing_height) + subplot_abs_excess_height
    return (fig_width, fig_height)


def get_model_colors() -> dict:
    """Returns preset colors for trial arms. Keywords are HSA, additive, control, experimental, and combo.

    Returns:
        dict: color dictionary
    """    
    # define colors
    blue = [i/255 for i in (0, 128, 255)]  # hsa
    red = [i/255 for i in (200, 0, 50)]  # additivity

    color_dict = {'HSA': blue, 'additive': red, 'control': 'orange', 'experimental': 'green', 'combo': 'black'}
    return color_dict


def make_axes_logscale_for_HR(ax: plt.axes, x_major: list, y_major: list) -> plt.axes:
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.yaxis.set_major_locator(plticker.FixedLocator(y_major))
    ax.yaxis.set_major_formatter(plticker.FixedFormatter(y_major))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    return ax
