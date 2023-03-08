import numpy as np
import pandas as pd
from lognormal_fitting import lognormal_survival
from hsa_additivity_model import predict_both
from plotting.plot_lognormal_examples import plot_lognormal_examples
import yaml
import warnings


with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

FIG_DIR = CONFIG['approved']['fig_dir']

warnings.filterwarnings("ignore")

def get_lognorm_survival_dataframe(tmax, n, mu, sigma):
    """Generate survival dataframe for lognormal survival curve.

    Args:
        tmax (float): maximum follow-up time
        n (int): number of datapoints
        mu (float): mean of lognomral distribution
        sigma (float): standard deviation of lognormal distribution

    Returns:
        pd.DataFrame: lognormal survival data
    """    
    timepoints = np.linspace(0, tmax, n)
    survpoints = lognormal_survival(timepoints, mu, sigma) * 100
    df = pd.DataFrame({'Time': timepoints, 'Survival': survpoints})
    return df.iloc[::-1]


def get_lognormal_examples(tmax, n, mu_a, mu_b, sigma, rho=0.3):
    """Get demonstrative lognormal survival curves of monotherapies,
    HSA, and additivity predictions.

    Args:
        tmax (float): maximum follow-up time
        n (int): number of datapoints
        mu_a (float): mean of lognomral distribution for drug A
        mu_b (float): mean of lognomral distribution for drug B
        sigma (float): standard deviation of lognormal distribution
        rho (float, optional): Spearman rho for two drug responses. Defaults to 0.3.

    Returns:
        dict: dictonary of dataframes for A, B, HSA, and Additivity.
    """    
    drugA = get_lognorm_survival_dataframe(tmax, n, mu_a, sigma)
    drugB = get_lognorm_survival_dataframe(tmax, n, mu_b, sigma)
    hsa, add = predict_both(drugA, drugB, 'Drug A', 'DrugB', 'a', 1, rho=rho, N=500)
    hsa = hsa.append({'Time': 0, 'Survival': 100}, ignore_index=True)
    add = add.append({'Time': 0, 'Survival': 100}, ignore_index=True)
    return {'A': drugA, 'B': drugB, 'HSA': hsa, 'Additivity': add}


def main():
    less_variable = get_lognormal_examples(20, 500, 2, 2.2, 0.5)
    more_variable = get_lognormal_examples(20, 500, 1, 1.5, 2)
    fig = plot_lognormal_examples(less_variable, more_variable)
    fig.savefig(f'{FIG_DIR}/explain_HSA_additive_difference.pdf')


if __name__ == '__main__':
    main()
