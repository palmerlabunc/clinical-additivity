import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def import_dose_data():
    """Import data for drawing relative dose between monotherapy and
    combination therapy.

    Returns:
        pd.DataFrame: relative dose data
    """    
    df = pd.read_excel('../data/trials/differences_in_doses.xlsx',
                       sheet_name='relative', header=0, index_col=None, engine='openpyxl')
    cat_order = CategoricalDtype(['Main', 'Suppl', 'No'], ordered=True)
    df['Included in Analysis'] = df['Included in Analysis'].astype(cat_order)
    df = df.sort_values(['Included in Analysis', 'Relative dose (monotherapy: 1)'])
    df = df.reset_index(drop=True).reset_index()
    return df


def plot_doses_difference():
    """_summary_

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
