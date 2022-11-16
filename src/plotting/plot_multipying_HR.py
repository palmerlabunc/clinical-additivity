import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, '..')
from hsa_additivity_model import predict_both


def exponential_survival(ts, mu):
    return pd.DataFrame({'Time': ts, 'Survival': 100 * np.exp(-mu * ts)})


def plot_multipying_HR_example(outdir):
    """_summary_

    Args:
        outdir (str): output directory

    Returns:
        plt.figure: plotted figure
    """    
    ts = np.linspace(0, 20, 500)
    placebo = exponential_survival(ts, 1)
    mu_a, mu_b = 0.2, 0.25
    a = exponential_survival(ts, mu_a)
    b = exponential_survival(ts, mu_b)
    multiplied = exponential_survival(ts, mu_a * mu_b)
    _, add = predict_both(a, b, 'A', 'B', subtracted='a',
                            scan_time=1, N=5000, rho=0, save=False)

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    sns.despine()
    ax.plot('Time', 'Survival', data=placebo, label=r'placebo ($\lambda$=1)')
    ax.plot('Time', 'Survival', data=a, label=r'A ($\lambda$=0.2)')
    ax.plot('Time', 'Survival', data=b, label=r'B ($\lambda$=0.25)')
    ax.plot('Time', 'Survival', data=multiplied, label='HR multiplied')
    ax.plot('Time', 'Survival', data=add, label='additive')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival (%)')
    ax.legend(bbox_to_anchor=(1, 1.05))
    ax.set_yticks([0, 50, 100])
    return fig


def main():
    outdir = '../../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)
    fig = plot_multipying_HR_example(outdir)
    fig.savefig(outdir + 'multipyling_HR_example.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()