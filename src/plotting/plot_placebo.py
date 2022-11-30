import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def make_label(name: str) -> str:
    """Format file name to '{Cancer type} {Author} et al. {Year}' for figure label.

    Args:
        name (str): input file prefix in the format of '{Cancer}_{Drug}_{Author}{Year}_PFS'

    Returns:
        str: formatted label
    """    
    tokens = name.split('_')
    cancer = tokens[0]
    author, year = tokens[2][:-4], tokens[2][-4:]
    return f"{cancer} Cancer\n({author} et al. {year})"


def plot_one_placebo(df: pd.DataFrame, scan_time: float, ax: plt.axes, label=None) -> plt.axes:
    """ Plot one placebo curve to an axes object.

    Args:
        df (pd.DataFrame): input survival data
        scan_time (float): first scan time in months
        ax (plt.axes): axes to plot the data
        label (str, optional): axes title. Defaults to None.

    Returns:
        plt.axes: plotted axes
    """    
    xticks = [0, 6, 12]
    yticks = [0, 50, 100]

    ### plot
    ax.plot(df['Time'], df['Survival'], linewidth=1, color='k')
    ax.axvline(scan_time, color='r', linewidth=1)
    ax.set_title(make_label(label))
    ax.set_xlabel('')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 105)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_locator(plt.MultipleLocator(6))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2))

    return ax


def plot_placebo() -> plt.figure:
    """Plot all placebo curves in one figure.

    Returns:
        plt.figure: plotted figure of all placebo survival curves 
    """    
    indf = pd.read_csv('../data/placebo/placebo_input_list.txt', sep='\t', header=0)
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, sharey=True, 
                             figsize=(6, 6), 
                             subplot_kw=dict(box_aspect=0.5), dpi=300)

    sns.despine()
    flat_axes = axes.flatten()

    for i in range(indf.shape[0]):
        path = indf.at[i, 'Path'] + '/'
        file_prefix = indf.at[i, 'File prefix']
        scan_time = indf.at[i, 'First scan time (months)']
        placebo = pd.read_csv(path + file_prefix + '.clean.csv', header=0)
        flat_axes[i] = plot_one_placebo(placebo, scan_time, flat_axes[i], label=file_prefix)

    return fig
