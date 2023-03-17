import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import yaml
import warnings
from plotting.plot_utils import import_input_data
from weibull_fitting import weibull3_params_from_digitized, weibull3_pdf, weibull3_survival
from coxhazard_test import create_ipd

warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

config_dict = CONFIG['approved']
COMBO_SEED_SHEET = config_dict['metadata_sheet_seed']
COMBO_DATA_DIR = config_dict['data_dir']
RAW_COMBO_DATA_DIR = config_dict['raw_dir']
PFS_PRED_DIR = config_dict['pred_dir']
FIG_DIR = f"{config_dict['fig_dir']}/weibull_fit"
TABLE_DIR = config_dict['table_dir']
Path(FIG_DIR).mkdir(exist_ok=True, parents=True)


def calculate_NLL(df: pd.DataFrame, weibull_fit_fig=True) -> pd.DataFrame:
    """Calculate negative log-likelihood for HSA and additivity for each combination.

    Args:
        df (pd.DataFrame): input dataframe for combinations
        weibull_fit_fig (bool, optional): Option to generate figures to inspect weibull fits. Defaults to True.

    Returns:
        pd.DataFrame: calcuated NLL for each combination
    """
    # calculate negative log likelihood
    lik_df = pd.DataFrame(index=df.index, columns=['Name', 'HSA_NLL', 'Add_NLL'])
    # some survival curves have discrete steps at the end. This throws off Weibull fitting.
    # Manually force a Tmax for these cases. 
    tmax_dict = {4: 20, 5: 15, 8: 25, 11: 20, 13: 9,
                 14: 15, 15: 15, 17: 15, 19: 20, 20: 15,
                 21: 15, 22: 40, 23: 30, 34: 15, 35: 15}
    tmp = df
    tstep = 0.25
    
    for i in range(tmp.shape[0]):
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']
        name_ab = tmp.at[i, 'Combination']
        n_combo = tmp.at[i, 'N_combination']
        lik_df.at[i, 'Name'] = name_ab
        # observed data
        df_ab = pd.read_csv(f'{COMBO_DATA_DIR}/{name_ab}.clean.csv').dropna()
        try:
            ipd_ab = pd.read_csv(f'{RAW_COMBO_DATA_DIR}/{name_ab}_indiv.csv')
            #print("used IPD")
        except FileNotFoundError:
            ipd_ab = create_ipd(df_ab, n=n_combo)

        # import prediction
        independent = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_ind.csv').dropna()
        additive = pd.read_csv(
            f'{PFS_PRED_DIR}/{name_a}-{name_b}_combination_predicted_add.csv').dropna()
        models = [independent, additive]
        print(i, name_ab)

        
        if weibull_fit_fig:
            fig, ax = plt.subplots()
            ax.plot(additive['Time'], additive['Survival'], label='additive')
            ax.plot(independent['Time'], independent['Survival'], label='HSA')
            ax.set_title(i)
        for k in range(len(models)):
            model = models[k]
            tmax = tmax_dict.get(i)
            if tmax is None:
                tmax = model['Time'].max() - tstep
            t_event = ipd_ab[ipd_ab['Event'] == 1]['Time'].values
            t_censor = ipd_ab[ipd_ab['Event'] == 0]['Time'].values
            model = model.reindex(range(0, 5000, 20))  # sample to 1/10 number of patients
            wa, wb, wc = weibull3_params_from_digitized(model, tmax)
            # calculate negative log likelihood
            l_event = weibull3_pdf(t_event, wa, wb, wc) # use PDF for events
            l_censor = weibull3_survival(t_censor, wa, wb, wc) # use survival for censoring
            l_all = np.hstack((l_event, l_censor))
            neg_log_lik = np.round(np.sum(-np.log(l_all)), 3)
            if k == 0:
                lik_df.at[i, 'HSA_NLL'] = neg_log_lik
            elif k == 1:
                lik_df.at[i, 'Add_NLL'] = neg_log_lik
            
            if weibull_fit_fig:
                t = np.linspace(0, tmax, 100)
                ax.plot(t, weibull3_survival(t, wa, wb, wc) * 100)
        if weibull_fit_fig:
            plt.legend()
            fig.savefig(f'{FIG_DIR}/{name_ab}.weibull_fit.png')


    return lik_df


def calculate_AIC(lik_df, p=1):
    #p(int, optional): Number of parameters in the model. Defaults to 1.
    aic_df = lik_df.copy()
    aic_df.columns = ['Name', 'HSA_AIC', 'Add_AIC']
    aic_df['HSA_AIC'] = 2 * p + 2 * lik_df['HSA_NLL']
    aic_df['Add_AIC'] = 2 * p + 2 * lik_df['Add_NLL']
    return aic_df
    

def calculate_relative_AIC(aic_df: pd.DataFrame) -> float:
    """Calculate relative likelihood of HSA compared to Additivity.

    Args:
        aic_df (pd.DataFrame): AIC dataframe

    Returns:
        float: average relative likelihood of HSA compared to additivity
    """    
    aic = aic_df[['HSA_AIC', 'Add_AIC']].mean().values
    print(aic)
    hsa_aic, add_aic = aic[0], aic[1]
    return np.exp((add_aic - hsa_aic)/2)


def main():
    indf = import_input_data()
    lik_df = calculate_NLL(indf, weibull_fit_fig=True)
    aic_df = calculate_AIC(lik_df)
    aic_df.to_csv(f'{TABLE_DIR}/AIC.csv', index=False)
    print(calculate_relative_AIC(aic_df))


if __name__ == '__main__':
    main()