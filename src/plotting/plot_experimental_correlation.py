import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import sys
sys.path.insert(0, '..')
from experimental_correlation import get_pdx_corr_data, get_ctrp_corr_data


def draw_corr_pdx(df, drug1, drug2, metric='BestAvgResponse'):
    tmp = get_pdx_corr_data(df, drug1, drug2, metric=metric)
    r, p = spearmanr(tmp[drug1], tmp[drug2])
    fig, ax = plt.subplots(figsize=(2, 2))

    a = df[df['Treatment'] == drug1]['ResponseCategory'] == 'PD'
    b = df[df['Treatment'] == drug2]['ResponseCategory'] == 'PD'
    if metric == 'BestAvgResponse':
        ax.axvline(0, color='gray', linestyle='--')
        ax.axhline(0, color='gray', linestyle='--')
        if (a.sum() / a.shape[0] > 0.75) or (b.sum() / b.shape[0] > 0.75):
            print("WARNING: at least one of the drug is inactive!")
    tumor = tmp['Tumor Type'].unique()
    if tumor.size > 1:
        sns.scatterplot(x=drug1, y=drug2, hue='Tumor Type', data=tmp, ax=ax)
        ax.set_title('n={0} rho={1:.2f}'.format(tmp.shape[0], r))
    else:
        sns.scatterplot(x=drug1, y=drug2, data=tmp, ax=ax)
        ax.set_title('{0} n={1} rho={2:.2f}'.format(tumor[0], tmp.shape[0], r))
    ax.set_xlabel(drug1 + ' ({})'.format(metric))
    ax.set_ylabel(drug2 + ' ({})'.format(metric))

    return fig


def draw_ctrp_spearmanr_distribution(all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    sns.despine()
    sns.histplot(all_pairs, ax=ax, label='all pairs', color=sns.color_palette()[0])
    sns.histplot(targ_pairs, label='targeted pairs', color=sns.color_palette()[2])
    sns.histplot(cyto_targ_pairs, label='cytotoxic-targeted pairs',
                color=sns.color_palette()[3])
    sns.histplot(cyto_pairs, label='cytotoxic pairs', color=sns.color_palette()[1])
    ax.set_xlim(-0.5, 1)
    ax.set_xlabel('Spearman rho')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),)

    return fig


def draw_corr_cell(ctrp_df, cancer_type, drug1, drug2, 
                   metric='CTRP_AUC', only_cancer_type=None):
    dat = get_ctrp_corr_data(ctrp_df, cancer_type, drug1, drug2, metric)
    if only_cancer_type is not None:
        dat = dat[dat['Cancer_Type_HH'] == only_cancer_type]
    r, p = spearmanr(dat[drug1], dat[drug2], axis=0, nan_policy='omit')

    ticks = [0, 0.5, 1]

    fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True, 
                           subplot_kw=dict(box_aspect=1))
    sns.despine()
    sns.scatterplot(x=drug1, y=drug2, size=1, color='k', alpha=0.7,
                    data=dat, ax=ax)
    ax.get_legend().remove()
    ax.set_xlabel(drug1 + ' ' + metric)
    ax.set_ylabel(drug2 + ' ' + metric)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # identity line
    lims = [0, min(ax.get_xlim()[1], ax.get_ylim()[1])]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    ax.set_title('n={}, rho={:.2f}'.format(dat.shape[0], r))
    
    return fig
