import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import yaml
from scipy.stats import spearmanr
from lognormal_examples import get_lognormal_examples
from coxhazard_test import get_cox_results, create_ipd

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

plt.style.use('env/publication.mplstyle')

def corr_HR_associtation(cox_df):
    sns.scatterplot('Corr', 'HR_add', data=cox_df)
    print(spearmanr(cox_df['Corr'], cox_df['HR_add']))


def change_in_corr():
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), tight_layout=True)
    colors = list(sns.color_palette("rocket", 5))[::-1]
    rho_list = np.arange(0, 1, 0.2)
    models = ['HSA', 'Additivity']
    HR_arr = np.zeros((len(rho_list), len(models)))
    HRlow_arr = np.zeros((len(rho_list), len(models)))
    HRhigh_arr = np.zeros((len(rho_list), len(models)))

    for rho_idx in range(len(rho_list)):
        rho = rho_list[rho_idx]
        dic = get_lognormal_examples(20, 500, 1.2, 1.5, 1, rho=rho)
        ipd_control = create_ipd(dic['B'])
        for model_idx in range(len(models)):
            df = dic[models[model_idx]]
            ipd_model = create_ipd(df)
            p, HR, low, high = get_cox_results(ipd_control, ipd_model)
            HR_arr[rho_idx, model_idx] = HR
            HRlow_arr[rho_idx, model_idx] = low
            HRhigh_arr[rho_idx, model_idx] = high
            axes[0, model_idx].plot(df['Time'], df['Survival'], 
                                    color=colors[rho_idx], label="{:.1f}".format(rho))
    axes[0, 1].legend(title='rho', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    axes[0, 0].set_title('HSA')
    axes[0, 1].set_title('Additivity')
    axes[0, 0].set_ylabel('Survival (%)')
    axes[1, 0].set_ylabel('HR(model vs. control)')
    for i in range(2):
        axes[0, i].set_xlim(0, 20)
        axes[0, i].set_ylim(0, 105)
        axes[0, i].set_xlabel('Time (months)')
        axes[1, i].set_xlabel('rho')
        axes[1, i].set_xlim(-0.1, 1)
    axes[1, 0].set_ylim(0.5, 2)
    axes[1, 1].set_ylim(0.5, 2)
    
    for model_idx in range(len(models)):
        ci = np.array([HR_arr[:, model_idx] - HRlow_arr[:, model_idx], 
                       HRhigh_arr[:, model_idx] - HR_arr[:, model_idx]])
        axes[1, model_idx].plot(rho_list, HR_arr[:, model_idx], linewidth=1, color='black')
        axes[1, model_idx].errorbar(y=HR_arr[:, model_idx],
                                    x=rho_list,
                                    yerr=ci,
                                    color='black',  capsize=3, linestyle='None',
                                    linewidth=1, marker="o", markersize=4, mfc="black", mec="black")

    for ax in axes[1, :]:
        ax.set_yscale('log', basey=2)
        major = [0.25, 0.5, 1, 2]
        ax.yaxis.set_major_formatter(plticker.FixedFormatter(major))
        ax.yaxis.set_major_locator(plticker.FixedLocator(major))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    
    return fig


def main():
    cox_df = pd.read_csv(CONFIG['all_phase3']['cox_result'], index_col=None)
    corr_HR_associtation(cox_df)
    fig = change_in_corr()
    fig.savefig(f"{CONFIG['fig_dir']}/corr_impact_on_additivity.pdf",
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()