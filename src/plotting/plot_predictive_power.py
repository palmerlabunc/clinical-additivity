import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from .plot_utils import get_model_colors

plt.style.use('env/publication.mplstyle')

def calc_tpr_fpr(df: pd.DataFrame, model: str, threshold: float) -> tuple:
    actual_p = df['PFS_improvement'].sum()
    actual_n = df.shape[0] - actual_p
    tp = ((df['PFS_improvement'] == 1) & (df[f'prob_success_{model}'] >= threshold)).sum()
    tpr = tp / actual_p
    fp = ((df['PFS_improvement'] == 0) & (df[f'prob_success_{model}'] >= threshold)).sum()
    fpr = fp / actual_n
    return (tpr, fpr)


def plot_roc_curve(indf: pd.DataFrame):
    color_dict = get_model_colors()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.despine()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


    for model in ['ind', 'add']:
        fpr, tpr, thresholds = roc_curve(indf['PFS_improvement'], indf[f'prob_success_{model}'])
        auc = roc_auc_score(indf['PFS_improvement'], indf[f'prob_success_{model}'])
        auc = np.round(auc, 2)
        if model == 'ind':
            color = color_dict['HSA']
            txt = f'HSA AUC = {auc}'
            ax.text(1, 0.1, txt, c=color, 
                    horizontalalignment='right', verticalalignment='bottom')

        else:

            color = color_dict['additive']
            txt = f'Additivity AUC = {auc}'
            ax.text(1, 0, txt, c=color,
                    horizontalalignment='right', verticalalignment='bottom')
            #optimal_idx = np.argmax(tpr - fpr)
            tpr_at_50, fpr_at_50 = calc_tpr_fpr(indf, 'add', 0.5)
            ax.scatter(fpr_at_50, tpr_at_50, marker='o', s=5, color='k')
            optimal_txt = "Sensitivity = {0:.2f}\nSpecificity = {1:.2f}".format(tpr_at_50, 1 - fpr_at_50)
            ax.text(1, 0.5, optimal_txt, color='k',
                    horizontalalignment='right', verticalalignment='bottom')
        ax.plot(fpr, tpr, color=color)
    return fig


def plot_precision_recall_curve(indf):
    color_dict = get_model_colors()
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.despine()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0.5, 0.75, 1])
    
    for model in ['ind', 'add']:
        precision, recall, thresholds = precision_recall_curve(indf['PFS_improvement'], indf[f'prob_success_{model}'])
        if model == 'ind':
            color = color_dict['HSA']
            txt = 'HSA'
            ax.text(1, 0.6, txt, c=color, 
                    horizontalalignment='right', verticalalignment='bottom')
        else:
            color = color_dict['additive']
            txt = 'Additivity'
            ax.text(1, 0.5, txt, c=color,
                    horizontalalignment='right', verticalalignment='bottom')
        ax.plot(recall, precision, color=color)
    return fig


def plot_swarmplot(indf):
    tmp = indf.copy()
    tmp.loc[:, 'PFS_improvement'] = tmp['PFS_improvement'].replace({0: "No", 1: "Yes"})

    # calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(indf['PFS_improvement'], indf['prob_success_add'])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # plot
    fig, ax = plt.subplots(figsize=(2.5, 1.5))

    sns.despine()
    sns.swarmplot(x='prob_success_add', y='PFS_improvement', data=tmp, ax=ax,
                  palette=['#23b200', "#9900cc"], size=4, alpha=0.8)
    ax.set_xlabel('Prob(success) under Additivity')
    ax.set_ylabel('PFS improvement')
    ax.legend().remove()
    #ax.axvline(optimal_threshold, c='k')
    #print(f"Optimal threshold = {optimal_threshold}")
    return fig






