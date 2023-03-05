import numpy as np
import pandas as pd
import yaml
import warnings
from multiprocessing import Pool
from coxhazard_test import get_cox_results, create_ipd
from plotting.plot_predictive_power import plot_roc_curve, plot_precision_recall_curve, plot_swarmplot
warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

N = 5000
NRUN = 10000
SEED = 0
rng = np.random.default_rng(SEED)


def simulate_one_trial(sampled_patients: np.array, ipd_ori: pd.DataFrame, ipd_control: pd.DataFrame) -> int:
    ipd_sim = ipd_ori.reindex(sampled_patients)
    p, HR, low95, high95 = get_cox_results(ipd_control, ipd_sim)
    success = 0
    if p < 0.05 and high95 < 1:
        success = 1
    return success


def calculate_success_prob(input_df: pd.DataFrame, output_df: pd.DataFrame, i: int, data_dir: str, pred_dir: str) -> tuple:
    name_a = input_df.at[i, 'Experimental']
    name_b = input_df.at[i, 'Control']
    n_control = input_df.at[i, 'N_control'].astype(int)
    n_combo = input_df.at[i, 'N_combination'].astype(int)
    # observed data
    df_b = pd.read_csv(f'{data_dir}/{name_b}.clean.csv')
    try:
        ipd_control = pd.read_csv(f'{data_dir}/{name_b}_indiv.csv')

    except FileNotFoundError:
        ipd_control = create_ipd(df_b, n=n_control)

    # import prediction
    independent = pd.read_csv(
        f'{pred_dir}/{name_a}-{name_b}_combination_predicted_ind.csv')
    additive = pd.read_csv(
        f'{pred_dir}/{name_a}-{name_b}_combination_predicted_add.csv')

    independent = independent[independent['Time']
                              < independent['Time'].max() - 0.1]
    additive = additive[additive['Time'] < additive['Time'].max() - 0.1]

    ipd_add = create_ipd(additive, n=N)
    ipd_ind = create_ipd(independent, n=N)

    success_cnt_ind = 0
    success_cnt_add = 0
    for run in range(NRUN):
        sampled_patients = rng.integers(0, N, n_combo)
        success_cnt_ind += simulate_one_trial(sampled_patients, ipd_ind, ipd_control)
        success_cnt_add += simulate_one_trial(sampled_patients, ipd_add, ipd_control)
    
    return [i, success_cnt_ind / NRUN, success_cnt_add / NRUN]


def predictive_power(metadata, data_dir, pred_dir):
    outdf = metadata.iloc[:, :15].copy()
    ll = []
    with Pool(processes=12) as pool:
        args_list = [(metadata, outdf, i, data_dir, pred_dir) for i in range(metadata.shape[0])]
        for result in pool.starmap(calculate_success_prob, args_list):
            print(result)
            ll.append(result)
    tmp = pd.DataFrame(ll, columns=['idx','prob_success_ind', 'prob_success_add'])
    tmp = tmp.set_index('idx', drop=True)
    outdf = pd.concat([outdf, tmp], axis=1)
    return outdf


def main():
    config_dict = CONFIG['all_phase3']
    metadata = pd.read_csv(config_dict['metadata_sheet_seed'], sep='\t')
    data_dir = config_dict['data_dir']
    pred_dir = config_dict['pred_dir']
    table_dir = config_dict['table_dir']
    fig_dir = config_dict['fig_dir']

    outdf = predictive_power(metadata, data_dir, pred_dir)
    outdf.to_csv(f'{table_dir}/predictive_power.csv', index=False)

    outdf = pd.read_csv(f'{table_dir}/predictive_power.csv', index_col=None)
    fig = plot_roc_curve(outdf)
    fig.savefig(f'{fig_dir}/roc_curve.pdf',
                bbox_inches='tight', pad_inches=0.1)
    fig = plot_precision_recall_curve(outdf)
    fig.savefig(f'{fig_dir}/precision-recall_curve.pdf',
                bbox_inches='tight', pad_inches=0.1)
    fig = plot_swarmplot(outdf)
    fig.savefig(f'{fig_dir}/additivity_prob_success_swarm_plot.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
