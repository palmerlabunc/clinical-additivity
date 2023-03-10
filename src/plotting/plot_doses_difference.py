import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

DOSE_DIFF_SHEET = CONFIG['relative_doses']
FIG_DIR = CONFIG['approved']['fig_dir']

plt.style.use('env/publication.mplstyle')
warnings.filterwarnings("ignore")


def import_dose_data():
    """Import data for drawing relative dose between monotherapy and
    combination therapy.

    Returns:
        pd.DataFrame: relative dose data
    """    
    df = pd.read_excel(DOSE_DIFF_SHEET,
                       sheet_name='relative', header=0, index_col=None, engine='openpyxl')
    cat_order = CategoricalDtype(['Main', 'Suppl', 'No'], ordered=True)
    df['Included in Analysis'] = df['Included in Analysis'].astype(cat_order)
    df = df.sort_values(['Included in Analysis', 'Relative dose (monotherapy: 1)'])
    df = df.reset_index(drop=True).reset_index()
    return df


def plot_doses_difference():
    """Plot bar plot of relative doses between monotherapy and combination arm.

    Returns:
        plt.figure: plotted figure
    """    
    df = import_dose_data()
    fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    red = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    sns.barplot(y='Relative dose (monotherapy: 1)', x='index', data=df, dodge=False, ax=ax,
                hue='Included in Analysis', palette={'Main': 'k', 'Suppl': 'grey', 'No': red})
    ax.set_xticks([])
    ax.set_ylabel('Relative Dose')
    ax.set_xlabel('Combinations')
    return fig


def main():
    fig = plot_doses_difference()
    fig.savefig(f'{FIG_DIR}/relative_doses.pdf')


if __name__ == '__main__':
    main()
