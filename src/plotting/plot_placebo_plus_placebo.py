import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

plt.style.use('env/publication.mplstyle')
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

config_dict = CONFIG['placebo']
PLACEBO_INPUT_SHEET = config_dict['metadata_sheet']
PLACEBO_DATA_DIR = config_dict['data_dir']
FIG_DIR = config_dict['fig_dir']


def plot_placebo_plus_placebo():
    placebo_input = pd.read_csv(PLACEBO_INPUT_SHEET,
                                sep='\t', header=0)
    fig, axes = plt.subplots(3, 3, figsize=(5, 5), constrained_layout=True)
    axes = axes.flatten()
    colors = list(sns.color_palette("rocket", 4))[::-1]
    rho_list = [0.3, 0.6, 1]
    for i in range(placebo_input.shape[0]):
        ax = axes[i]
        file_prefix = placebo_input.at[i, 'File prefix']
        placebo = pd.read_csv(f'{PLACEBO_DATA_DIR}/{file_prefix}.clean.csv', header=0)

        ax.plot('Time', 'Survival', data=placebo, label='placebo', color='k', linestyle='--')
        for k in range(len(rho_list)):
            rho = rho_list[k]
            add = pd.read_csv(f'../analysis/placebo_plus_placebo/{file_prefix}_add_{rho}.csv', 
                              index_col=False, header=0)
            ax.plot(add['Time'], add['Survival'], label=f'rho={rho}', color=colors[k])
        ax.legend()
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 105)
        ax.set_title(
            f'{placebo_input.at[i, "Cancer Type"]}\n{placebo_input.at[i, "Trial"]}')
        ax.set_xticks(list(range(0, 20, 6)))
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival (%)')
    return fig


if __name__ == '__main__':
    fig = plot_placebo_plus_placebo()
    fig.savefig(f'{FIG_DIR}/placebo_plus_palcebo.pdf')
