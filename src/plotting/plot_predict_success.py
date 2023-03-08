import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import set_figsize

plt.style.use('env/publication.mplstyle')

def plot_predict_success(results):
    """Plot scatterplot of HR(obs combo vs. control) vs. HR(exp combo vs. control).

    Args:
        results (pd.DataFrame): dataframe containing HR(exp vs. control)

    Returns:
        plt.figure: plotted figure
    """    
    rows, cols = 1, 2
    ticks = [0, 0.5, 1]
    fig_width, fig_height = set_figsize(1, rows, cols)
    fig, axes = plt.subplots(rows, cols, sharey=True, figsize=(fig_width, fig_height),
                            subplot_kw=dict(box_aspect=1), dpi=300)

    sns.scatterplot(x='HR(combo/control)', y='HR_ind', data=results, zorder=2,
                    hue='success_ind', hue_order=[True, False], palette={True: 'green', False: 'gray'}, ax=axes[0])
    sns.scatterplot(x='HR(combo/control)', y='HR_add', data=results, zorder=2,
                    hue='success_add', hue_order=[True, False], palette={True: 'green', False: 'gray'}, ax=axes[1])

    axes[0].set_ylabel('HR(Exp Combo vs. Control)')
    axes[0].set_title('HSA')
    axes[1].set_title('Additivity')
    axes[0].set_xticks(ticks)

    for ax in axes:
        ax.plot([0, 1], [0, 1], color='k', alpha=0.5, linewidth=1, zorder=1)
        ax.set_xlabel('HR(Obs Combo vs. Control)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks(ticks)


    return fig
