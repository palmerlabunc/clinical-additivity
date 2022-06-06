import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from sklearn.metrics import r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import sys
sys.path.append("../../survival_benefit/src")
from utils import interpolate

mpl.rcParams['pdf.fonttype'] = 42

INDIR = '../analysis/PFS_predictions/2022-06-06/'
OUTDIR = '../analysis/figure/{}/'.format(date.today())
new_directory = Path(OUTDIR)
new_directory.mkdir(parents=True, exist_ok=True)

def find_deviations(expected, observed, threshold=10):
    """Returns all indices of sequences that surpassed the threshold.

    Args:
        expected (np.ndarray): expected survival (%) values under model
        observed (np.ndarray): observed survival (%) values
        threshold (int, optional):  Defaults to 10.

    Returns:
        list: _description_
    """
    # extends the sequence to all positive values
    tmp1 = observed - expected > threshold
    tmp2 = observed - expected > 0
    idx = tmp1.astype(int) + tmp2.astype(int)
    
    all_seq = []
    start1_idx = -1
    met2 = False
    met0_idx = -1
    prev = idx[0]
    for i in range(1,idx.size):
        curr = idx[i]
        if prev == 0 and curr == 1:
            start1_idx = i
        if curr == 2:
            met2 = True
        if met2 and (prev == 1 or prev == 2) and curr == 0:
            met0_idx = i

        if i == idx.size - 1:
            met0_idx = idx.size - 1
        if start1_idx != -1 and met2 and met0_idx != -1:
            print(start1_idx, met0_idx)
            all_seq = all_seq + list(range(start1_idx, met0_idx))
            # reset all flags
            start1_idx = -1
            met2 = False
            met0_idx = -1
        prev = curr
    return all_seq

def plot_three_panel_each_combo():
    indf = pd.read_csv('../data/trials/final_input_list.txt', sep='\t')
    tmp = indf
    nrows = tmp.shape[0]

    # Set plot parameters #
    mm = 1/25.4

    params = {'legend.fontsize':9,
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.title_fontsize': 9}

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    ticks = [0, 50, 100]


    rcParams.update(params)

    # define colors
    blue = sns.color_palette()[0]
    orange = sns.color_palette()[1]
    green = sns.color_palette()[2]
    red = sns.color_palette()[3]

    sns.despine()

    for i in range(nrows):
        fig, axes = plt.subplots(1, 3, figsize=(7, 1.8), dpi=300, gridspec_kw={"width_ratios":[1.5, 1, 1]})
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        
        path = tmp.at[i, 'Path'] + '/'
        df_a = pd.read_csv(path + tmp.at[i, 'Experimental'] + '.clean.csv', 
                           header=0, index_col=False)
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv', 
                           header=0, index_col=False)
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv', 
                            header=0, index_col=False)
        
        independent = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b), index_col=False)
        additive = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b), index_col=False)
        
        # set same max time
        #tmax = np.amin([df_ab['Time'].max(), independent['Time'].max(), df_a['Time'].max(), df_b['Time'].max()])
        tmax = min(df_ab['Time'].max(), independent['Time'].max()) - 1
        surv_points = np.linspace(0, tmax, 500)
        
        # calculate r2
        f_ori = interpolate(x='Time', y='Survival', df=df_ab[df_ab['Time'] < tmax])
        f_ind = interpolate(x='Time', y='Survival', df=independent[independent['Time'] < tmax])
        f_add = interpolate(x='Time', y='Survival', df=additive[additive['Time'] < tmax])
        
        ori = f_ori(surv_points)
        ind = f_ind(surv_points)
        add = f_add(surv_points)
        r2_ind = r2_score(ori, ind)
        r2_add = r2_score(ori, add)
        
        ### left pane: KM plot
        axes[0].plot(df_a['Time'], df_a['Survival'], alpha=0.3, color=blue)
        axes[0].plot(df_b['Time'], df_b['Survival'], alpha=0.3, color=orange)
        axes[0].plot(df_ab['Time'], df_ab['Survival'], 
                        color='k', label='Observed combination')
        axes[0].plot(independent['Time'], independent['Survival'], 
                        color=green, label='Independence prediction')
        axes[0].plot(additive['Time'], additive['Survival'], 
                        color=red, label='Additivity prediction')
        # add drug names
        axes[0].text(0.5, 2, name_a.split('_')[1], color=blue, size=8)
        axes[0].text(0.5, 9, name_b.split('_')[1], color=orange, size=8)
        
        axes[0].set_ylabel('PFS (%)')
        axes[0].set_xlabel('')
        axes[0].legend()
        axes[0].get_legend().remove()
        axes[0].set_xlim(0, tmax - 0.1)
        axes[0].set_ylim(0, 105)
        axes[0].set_yticks(ticks)
        
        ### middle pane: independence QQ-plot
        axes[1].plot([0, 100], [0, 100], linestyle='--', alpha=0.5, color='k')
        idx = find_deviations(ind, ori)
        if not idx is None:
            axes[1].fill_between(ind[idx], ori[idx], ind[idx],
                                    alpha=0.2, color='purple', label='Benefit exceeding independence')
        axes[1].plot(ind, ori, alpha=0.8, color=green)
        axes[1].text(70, 10, r'$r^2$={:.2f}'.format(r2_ind), size=9)
        axes[1].set_ylabel('Observed PFS (%)')
        axes[1].set_xlim(0, 105)
        axes[1].set_ylim(0, 105)
        axes[1].set_yticks(ticks)
        axes[1].legend()
        axes[1].get_legend().remove()
        
        #### right pane: additivity QQ-plot
        axes[2].plot([0, 100], [0, 100], linestyle='--', alpha=0.3, color='k')
        axes[2].plot(add, ori, alpha=0.8, color=red)
        axes[2].text(70, 10, r'$r^2$={:.2f}'.format(r2_add), size=9)
        axes[2].set_xlim(0, 105)
        axes[2].set_ylim(0, 105)
        axes[2].set_yticks(ticks)

        fig.tight_layout()
        fig.savefig(OUTDIR + '{0}-{1}.pdf'.format(name_a, name_b))
        plt.close()

if __name__ == '__main__':
    plot_three_panel_each_combo()
