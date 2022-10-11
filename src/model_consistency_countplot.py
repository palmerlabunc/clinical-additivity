import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting.plot_utils import import_input_data


_, df = import_input_data()

df.loc[:, 'Model_for_fig'] = df['Model']
df.loc[df['Model'].isin(['additive', 'between']), 'Model_for_fig'] = "Additive"

fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)
sns.despine()
sns.countplot(data=df, y="Model_for_fig",
              order=['synergy', 'Additive', 'independent', 'worse than independent'])
fig.savefig('../figures/model_count.pdf')