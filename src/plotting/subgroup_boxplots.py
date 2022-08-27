import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from .plot_utils import import_input_data
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")

def plot_box(df, group, order, model='add'):
    """Helper function for boxplot.

    Args:
        df (pd.DataFrame): input data
        group (str): column name (categorical data) to plot
        order (list of strings): order to plot the categorical levels in
        model (str, optional): Model to use between HSA and additivity (ind or add). Defaults to 'add'.

    Returns:
        (plt.figure, plt.axes): tuple of figure and its associated axes
    """    
    fig, ax = plt.subplots(figsize=(2, 1))

    if model == 'ind':
        hr = 'HR_ind'
        xlab = 'HRHSA'
    elif model == 'add':
        hr = 'HR_add'
        xlab = 'HRAdditivity'

    sns.boxplot(x=hr, y=group, order=order, data=df,
                palette=sns.color_palette("Set2"), ax=ax)
    sns.swarmplot(x=hr, y=group, order=order, data=df,
                  color='k', size=3, alpha=0.7, ax=ax)
    ax.axvline(1, linewidth=0.5, color='gray', alpha=0.5)
    ax.set_xscale('log', base=2)
    x_major = [0.25, 0.5, 1, 2]
    ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.set_xlim(0.25, 2)
    ax.set_title(group)
    ax.set_ylabel("")
    ax.set_xlabel(xlab)
    return (fig, ax)

def plot_box_wrapper(feature):
    """Wrapper function to boxplot. Does the preprocessing, plotting, and adding p-values.

    Args:
        feature (str): Column name (categorical data) to plot the data

    Returns:
        plt.figure: boxplot figure
    """    
    _, df = import_input_data()
    df.loc[:, feature] = df[feature].replace({0: 'No', 1: 'Yes'})
    _, p_yes = wilcoxon(np.log(df[df[feature] == 'Yes']['HR_add']))
    _, p_no = wilcoxon(np.log(df[df[feature] == 'No']['HR_add']))
    fig, ax = plot_box(df, feature, ['Yes', 'No'], model='ind')
    ax.text(0.28, 0, 'p={:.3f}'.format(p_yes), size=9)
    ax.text(0.28, 1, 'p={:.3f}'.format(p_no), size=9)
    return fig

def plot_ici_boxplot():
    """Generate a boxplot for ICI combinations yes/no.

    Returns:
        plt.figure: plotted figure
    """    
    fig = plot_box_wrapper("ICI Combo")
    return fig


def plot_angio_boxplot():
    """Generate a boxplot for anti-angiogenesis combinations yes/no.

    Returns:
        plt.figure: plotted figure
    """
    fig = plot_box_wrapper("Anti-angio")
    return fig


def plot_mono_approval_boxplot():
    """Generate a boxplot for monotherapy approval yes/no.

    Returns:
        plt.figure: plotted figure
    """
    fig = plot_box_wrapper("Monotherapy Approval")
    return fig


def plot_HR_boxplot():
    """Generate a boxplot for HR > 0.6 yes/no.

    Returns:
        plt.figure: plotted figure
    """
    _, df = import_input_data()
    df.loc[:, 'HR binary'] = 'HR <= 0.6'
    df.loc[df['HR(combo/control)'] > 0.6, 'HR binary'] = 'HR > 0.6'
    _, p_high = wilcoxon(np.log(df[df['HR binary'] == 'HR > 0.6']['HR_add']))
    _, p_low = wilcoxon(np.log(df[df['HR binary'] == 'HR <= 0.6']['HR_add']))
    fig, ax = plot_box(df, 'HR binary', ['HR <= 0.6', 'HR > 0.6'], model='add')
    ax.text(0.28, 0, 'p={:.3f}'.format(p_low), size=9)
    ax.text(0.28, 1, 'p={:.3f}'.format(p_high), size=9)
    ax.set_title('HR(combination vs. control)')
    return fig
