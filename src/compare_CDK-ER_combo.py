from re import L
import numpy as np
import pandas as pd
from datetime import date
from coxhazard_test import get_cox_results
from plotting.plot_utils import import_input_data

def get_ipd(cox_df, i, arm):
    path = cox_df.at[i, 'Path'] + '/'
    file_prefix = cox_df.at[i, arm]
    try:
        ipd = pd.read_csv(path + file_prefix + '_indiv.csv')
    except FileNotFoundError:
        print("File Not Found.")
        return
    return ipd


def compare_abema_palbo(output_file):
    _, cox_df = import_input_data()
    abema_nsai_idx = 1
    palbo_nsai_idx = 7

    with open(output_file, 'w') as f:
        # sanity check
        f.write(f"Date of Analysis: {date.today()}\n")
        f.write(f"Drug 1: {cox_df.at[abema_nsai_idx, 'Combination']}\n")
        f.write(f"Drug 2: {cox_df.at[palbo_nsai_idx, 'Combination']}\n")
        
        for arm in ['Combination', 'Experimental', 'Control']:
            abema_ipd = get_ipd(cox_df, abema_nsai_idx, arm)
            palbo_ipd = get_ipd(cox_df, palbo_nsai_idx, arm)
            p, HR, lower, higher = get_cox_results(abema_ipd, palbo_ipd)
            f.write("{0} p={1:.03f}, HR={2:.02f}, 95% lower={3:.02f}, 95% higher={4:.02f}\n".format(arm, p, HR, lower, higher))


def main():
    compare_abema_palbo('../figures/Abemaciclib+NSAI_Palbociclib+Letrozole_HR.txt')


if __name__ == '__main__':
    main()