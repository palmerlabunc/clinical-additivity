import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import get_model_colors

plt.style.use('env/publication.mplstyle')

def plot_lognormal_examples(less_variable, more_variable):
    """Plot demonstrative figure of how the HSA and additivity predictions differ
    based on the variabilibity of monotherapy drug responses.

    Args:
        less_variable (dict): dictionary of dataframes containg A, B, HSA, and Addivitivity
                              for less variable lognormal survival curves
        more_variable (dict): dictionary of dataframes containg A, B, HSA, and Addivitivity
                              for more variable lognormal survival curves

    Returns:
        plt.figure: plotted figure
    """    
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(3, 3), 
                             subplot_kw=dict(box_aspect=0.75), dpi=300)
    sns.despine()
    color_dict = get_model_colors()
    # column 1: monotherapy curves
    # top: more variable
    axes[0, 0].plot('Time', 'Survival', data=more_variable['A'],
                    color=color_dict['experimental'], label='Drug A')
    axes[0, 0].plot('Time', 'Survival', data=more_variable['B'],
                    color=color_dict['control'], label='Drug B')
    # bottom: less variable
    axes[1, 0].plot('Time', 'Survival', data=less_variable['A'], 
                    color=color_dict['experimental'])
    axes[1, 0].plot('Time', 'Survival', data=less_variable['B'], 
                    color=color_dict['control'])


    # column 2: HSA and additivity
    axes[0, 1].plot('Time', 'Survival', data=more_variable['HSA'],
                    color=color_dict['HSA'], label='HSA')
    axes[0, 1].plot('Time', 'Survival', data=more_variable['Additivity'],
                    color=color_dict['additive'], label='Additivity')
    axes[0, 1].fill_betweenx(more_variable['HSA']['Survival'], 
                             more_variable['HSA']['Time'], 
                             more_variable['Additivity']['Time'],
                             color='lightgray', alpha=0.7)
    
    axes[1, 1].plot('Time', 'Survival', data=less_variable['HSA'],
                    color=color_dict['HSA'], label='HSA')
    axes[1, 1].plot('Time', 'Survival', data=less_variable['Additivity'],
                    color=color_dict['additive'], label='Additivity')
    axes[1, 1].fill_betweenx(less_variable['HSA']['Survival'],
                             less_variable['HSA']['Time'],
                             less_variable['Additivity']['Time'],
                             color='lightgray', alpha=0.7)

    for i in range(2):
        for k in range(2):
            axes[i, k].set_xlim(0, 19)
            axes[i, k].set_xticks([0, 5, 10, 15])
            axes[i, k].set_ylim(0, 105)

    axes[0, 0].legend()
    axes[0, 1].legend()
    
    return fig
