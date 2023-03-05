import pandas as pd
import yaml
from plotting.plot_utils import import_input_data

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

def make_string(row, model):
    return "{0:.2f} [{1:.2f}, {2:.2f}]".format(row[f'HR_{model}'], row[f'HRlower_{model}'], row[f'HRupper_{model}'])

df = import_input_data()
df.loc[:, 'HSA_string'] = df.apply(make_string, args=('ind',), axis=1)
df.loc[:, 'HSA_add'] = df.apply(make_string, args=('add',), axis=1)

df.to_csv(f"{CONFIG['approved']['table_dir']}/cox_ph_test_string.csv", index=False)