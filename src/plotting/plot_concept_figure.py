import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use('env/publication.mplstyle')

def plot_concept_figure():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3), 
                           dpi=300, constrained_layout=True)
    sns.despine()
    colors = sns.color_palette('colorblind')
    
    baseline = np.power(10, range(9, 11))
    a = np.power(10, range(8, 11))
    b = np.power(10, range(7, 11))

    axes[0].axhline(2*10**9, linestyle='--', color='k', alpha=0.5)
    axes[0].plot(baseline, lw=3, c=colors[0])
    axes[0].plot(a, lw=3, c=colors[1])
    axes[0].plot(b, lw=3, c=colors[2])
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(10**5, 2 * 10**10)

    axes[0].set_xticks([])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('# of Cancer Cells')

    baseline = np.power(10, range(9, 11))
    a = np.power(10, range(8, 11))
    b = np.power(10, range(7, 11))
    ab = np.power(10, range(6, 11))
    axes[1].axhline(2*10**9, linestyle='--', color='k', alpha=0.5)
    axes[1].plot(baseline, lw=3, c=colors[0])
    axes[1].plot(ab, lw=3, c=colors[3])
    axes[1].set_yscale('log')
    axes[1].set_xlim(0, 6)
    axes[1].set_ylim(10**5, 2 * 10**10)

    axes[1].set_xticks([])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('# of Cancer Cells')

    return fig
