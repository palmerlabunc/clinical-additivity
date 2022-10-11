import matplotlib.pyplot as plt
import seaborn as sns
from plotting.plot_utils import get_model_colors
from lognormal_examples import get_lognormal_examples
import numpy as np
from scipy.stats import norm
from utils import interpolate

less_variable = get_lognormal_examples(20, 500, 2, 2.2, 0.5)
more_variable =  get_lognormal_examples(20, 500, 1, 1.5, 2)
color_dict = get_model_colors()
"""
#### PDF ####
rows, cols = 1, 2
fig1, axes = plt.subplots(rows, cols, figsize=(4, 2), constrained_layout=True,
                          sharey=True, subplot_kw=dict(box_aspect=0.75), dpi=300)
sns.despine()
ts = np.linspace(0, 20, 500)


axes[0].plot(ts, norm.pdf((np.log(ts) - 2)/0.5), 
             color=color_dict['experimental'])
axes[0].plot(ts, norm.pdf((np.log(ts) - 2.2)/0.5),
             color=color_dict['control'])
axes[1].plot(ts, norm.pdf((np.log(ts) - 1)/2),
             color=color_dict['experimental'])
axes[1].plot(ts, norm.pdf((np.log(ts) - 1.5)/2),
             color=color_dict['control'])
axes[0].set_xscale('log')
axes[1].set_xscale('log')
fig1.savefig('../figures/lognormal_example_PDF.pdf')
"""
#### Survival ####
rows, cols = 1, 2
fig2, axes = plt.subplots(rows, cols, figsize=(4, 2), constrained_layout=True,
                          sharey=True, subplot_kw=dict(box_aspect=0.75), dpi=300)
sns.despine()
for i in range(2):
    axes[i].set_xlim(0, 19)
    axes[i].set_xticks([0, 5, 10, 15])
    axes[i].set_ylim(0, 105)


for i, dic in [(0, less_variable), (1, more_variable)]:
    axes[i].plot('Time', 'Survival', data=dic['A'],
                 color=color_dict['experimental'])
    axes[i].plot('Time', 'Survival', data=dic['B'],
                 color=color_dict['control'])

fig2.savefig('../figures/lognormal_example_monotherapy_survival.pdf')

#### HSA ####

fig2, axes = plt.subplots(rows, cols, figsize=(4, 2), constrained_layout=True,
                          sharey=True, subplot_kw=dict(box_aspect=0.75), dpi=300)
sns.despine()
for i in range(2):
    axes[i].set_xlim(0, 19)
    axes[i].set_xticks([0, 5, 10, 15])
    axes[i].set_ylim(0, 105)

survival_points = np.linspace(0, 100, 500)
for i, dic in [(0, less_variable), (1, more_variable)]:
    axes[i].plot('Time', 'Survival', data=dic['A'],
                 color=color_dict['experimental'], alpha=0.7)
    axes[i].plot('Time', 'Survival', data=dic['B'],
                 color=color_dict['control'], alpha=0.7)
    f_hsa = interpolate(dic['HSA'], x='Survival', y='Time')
    f_ctrl = interpolate(dic['B'], x='Survival', y='Time')
    axes[i].plot('Time', 'Survival', data=dic['HSA'],
                color=color_dict['HSA'])
    axes[i].fill_betweenx(survival_points,
                          f_ctrl(survival_points),
                          f_hsa(survival_points),
                          color=color_dict['HSA'], alpha=0.2)

fig2.savefig('../figures/lognormal_example_HSA_survival.pdf')

#### Additivity ####
for i, dic in [(0, less_variable), (1, more_variable)]:
    axes[i].plot('Time', 'Survival', data=dic['Additivity'],
                 color=color_dict['additive'])
    axes[i].fill_betweenx(dic['HSA']['Survival'],
                          dic['HSA']['Time'],
                          dic['Additivity']['Time'],
                          color=color_dict['additive'], alpha=0.2)

fig2.savefig('../figures/lognormal_example_additivity_survival.pdf')

