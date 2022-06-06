import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from utils import interpolate

INDIR = '../analysis/PFS_predictions/2022-06-06/'

def create_ipd(df, n=500):
    interp = interpolate(df, x='Survival', y='Time')
    # censoring due to loss of follow-up at the tail
    min_surv = np.round(np.ceil(df['Survival'].min())/100, 2)
    events = np.hstack((np.repeat(0, round(min_surv * n)), np.repeat(1, round((1 - min_surv) * n))))
    t = interp(np.linspace(0, 100, n))
    return pd.DataFrame({'Time': t, 'Event': events})

def cox_ph_test(input_df):
    tmp = input_df
    # output dataframe
    cox_df = pd.DataFrame(index=tmp.index, 
                                columns=['p_ind', 'HR_ind', 'HRlower_ind', 'HRupper_ind', 
                                         'p_add', 'HR_add', 'HRlower_add', 'HRupper_add', 'model'])
    cox_df.loc[:, 'model'] = tmp['Figure']
    for i in range(tmp.shape[0]):
        print(i)
        name_a = tmp.at[i, 'Experimental']
        name_b = tmp.at[i, 'Control']

        # observed data
        path = tmp.at[i, 'Path'] + '/'
        df_a = pd.read_csv(path + tmp.at[i, 'Experimental'] + '.clean.csv').dropna()
        df_b = pd.read_csv(path + tmp.at[i, 'Control'] + '.clean.csv').dropna()
        df_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '.clean.csv').dropna()
        
        try:
            ipd_ab = pd.read_csv(path + tmp.at[i, 'Combination'] + '_indiv.csv')
            print("used IPD")
        except FileNotFoundError:
            ipd_ab = create_ipd(df_ab, n=200)
        
        ipd_ab.loc[:, 'Arm'] = 1

        # import prediction
        independent = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_ind.csv'.format(name_a, name_b)).dropna()
        additive = pd.read_csv(INDIR + '{0}-{1}_combination_predicted_add.csv'.format(name_a, name_b)).dropna()

        tmax = np.amin([df_ab['Time'].max(), independent['Time'].max(), df_a['Time'].max(), df_b['Time'].max()])
        independent = independent[independent['Time'] < tmax]
        additive = additive[additive['Time'] < tmax]
        
        ipd_add = create_ipd(additive)
        ipd_ind = create_ipd(independent)

        # set up data for Cox regression
        cph = CoxPHFitter()
        
        # additive
        merged = pd.concat([ipd_ab, ipd_add], axis=0).reset_index(drop=True)
        merged = merged.fillna(0)
        merged.loc[:,'Arm'] = merged.loc[:,'Arm'].astype(int)
        cph.fit(merged, duration_col='Time', event_col='Event')
        cox_df.at[i, 'p_add'] = cph.summary.at['Arm', 'p']
        cox_df.at[i, 'HR_add'] = cph.summary.at['Arm', 'exp(coef)']
        cox_df.at[i, 'HRlower_add'] = cph.summary.at['Arm', 'exp(coef) lower 95%']
        cox_df.at[i, 'HRupper_add'] = cph.summary.at['Arm', 'exp(coef) upper 95%']
        
        # independent
        merged = pd.concat([ipd_ab, ipd_ind], axis=0).reset_index(drop=True)
        merged = merged.fillna(0)
        merged.loc[:,'Arm'] = merged.loc[:,'Arm'].astype(int)
        cph.fit(merged, duration_col='Time', event_col='Event')
        cox_df.at[i, 'p_ind'] = cph.summary.at['Arm', 'p']
        cox_df.at[i, 'HR_ind'] = cph.summary.at['Arm', 'exp(coef)']
        cox_df.at[i, 'HRlower_ind'] = cph.summary.at['Arm', 'exp(coef) lower 95%']
        cox_df.at[i, 'HRupper_ind'] = cph.summary.at['Arm', 'exp(coef) upper 95%']
        
    return cox_df

def main():
    indf = pd.read_csv('../data/trials/final_input_list.txt', sep='\t')
    cox_df = cox_ph_test(indf)
    results = pd.concat([indf, cox_df], axis=1)
    results.to_csv(INDIR + 'cox_ph_test.csv', index=False)

if __name__ == '__main__':
    main()
