configfile: "config.yaml"

import pandas as pd
import glob
import os

RAW_DIR = config['raw_trial_dir']
DATA_DIR = config['trial_dir']
RAW_FILE = f""
trials_df = pd.read_csv(config['combination_trial_sheet'], sep='\t', header=0)
trials_set = set()

for i in range(trials_df.shape[0]):
    for arm in ['Experimental', 'Control', 'Combination']:
        t = trials_df.at[i, arm]
        trials_set.add(t)
trials_list = list(trials_set)

rule all:
    input:
        expand(f"{DATA_DIR}/{{trial}}.clean.csv", trial=trials_list)
        expand(f"{DATA_DIR}")

rule preprocess_main_trials:
    input:
        f"{RAW_DIR}/{{trial}}.csv"
    output:
        f"{DATA_DIR}/{{trial}}.clean.csv"
    shell:
        "python src/preprocessing.py {input} --output {output}"

rule find_seeds:

#rule hsa_additivity_prediction:

#only need to parallelize up to this point

#####################

#rule cox_ph_test:

#rule figure1:

#rule figure2:

#rule figure3:

#rule figure4:

#rule figureS2:

#rule figureS3:

