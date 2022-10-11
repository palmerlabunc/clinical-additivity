import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
import warnings
warnings.filterwarnings("ignore")


def plot_uncertainty_stripplot():
    df = pd.read_csv(
        '../../figures/correlation_uncertainty_range95_avg.txt', sep='\t', header=0)
    # we want +/- XX % so, divide by 2
    df.loc[:, 'avg_high2low_add'] = df['avg_high2low_add']/2
    df.loc[:, 'avg_high2low_HSA'] = df['avg_high2low_HSA']/2
    melted = df.melt(id_vars=['Combination'], value_vars=['avg_high2low_add', 'avg_high2low_HSA'])
    fig, ax = plt.subplots(figsize=(3,3))
    sns.stripplot(x='variable', y='value', data=melted, ax=ax)
    print("Additivity mean: ", df['avg_high2low_add'].mean())
    print("HSA mean: ", df['avg_high2low_HSA'].mean())
    
    # plot the mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="variable",
                y="value",
                data=melted,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)
    fig.savefig('../../figures/correlation_uncertainty_range95_avg.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    plot_uncertainty_stripplot()
