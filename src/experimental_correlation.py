import pandas as pd
import numpy as np
from itertools import combinations
from plotting.plot_experimental_correlation import draw_corr_cell, draw_corr_pdx, draw_ctrp_spearmanr_distribution
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

EXPERIMENTAL_DATA_DIR = CONFIG['experimental_dir']
FIG_DIR = CONFIG['fig_dir']
TABLE_DIR = CONFIG['table_dir']


def import_ctrp_data():
    """Imports all CTRP data

    Returns:
        pd.DataFrame: cell line information
        pd.DataFrame: drug information
        pd.Series: cancer type information
        pd.DataFrame: viablility data
        pd.DataFrame: pairwise correlation data
    """
    # metadata
    cell_info = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_CCL.csv', index_col=0)
    drug_info = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_drug.csv', index_col=None)
    # drop non-cancer cell types
    noncancer = cell_info[cell_info['Cancer_Type_HH'] == 'Non-cancer'].index
    cell_info = cell_info.drop(noncancer)
    cancer_type = cell_info['Cancer_Type_HH'].squeeze()

    # viability data
    ctrp = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/Recalculated_CTRP_12_21_2018.txt',
                       sep='\t', index_col=0)

    # scale AUC
    ctrp.loc[:, 'CTRP_AUC'] = ctrp['CTRP_AUC'] / 16
    # drop non-cancer cell lines
    ctrp = ctrp[~ctrp['Harmonized_Cell_Line_ID'].isin(noncancer)]

    # keep only active drugs
    grouped = ctrp.groupby('Harmonized_Compound_Name')
    groups = np.array([name for name, unused_df in grouped])
    active_drugs = groups[grouped['CTRP_AUC'].quantile(0.1) < 0.8]
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(active_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(active_drugs)]

    # keep only drugs in clinical phases
    clinical_drugs = drug_info[~drug_info['Clinical Phase'].isin(
        ['Withdrawn', 'Preclinical', np.nan])]['Harmonized Name']
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(clinical_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(clinical_drugs)]
    return cell_info, drug_info, cancer_type, ctrp


def prepare_ctrp_agg_data(drug_info: pd.DataFrame):
    """Prepare data frame to plot distribution of correlations for drug pairs.

    Args:
        drug_info (pd.DataFrame): drug info data

    Returns:
        np.array: array of correlation values between all drug pairs
        np.array: array of correlation values between cytotoxic drug pairs
        np.array: array of correlation values between targeted drug pairs
        np.array: array of correlation values between cytotoxic-targeted drug pairs
    """
    df = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_clincal_active_drug_pairwise_corr.csv', 
                     index_col=0)
    vals = df.values.flatten()
    vals = vals[~np.isnan(vals)]

    # referenced from http://www.bccancer.bc.ca/pharmacy-site/Documents/Pharmacology_Table.pdf
    cyto_moa = ['DNA alkylating agent|DNA inhibitor',
                'DNA alkylating agent|DNA synthesis inhibitor',
                'DNA synthesis inhibitor',
                'ribonucleotide reductase inhibitor',
                'thymidylate synthase inhibitor',
                'dihydrofolate reductase inhibitor',
                'DNA alkylating agent',
                'DNA inhibitor',
                'topoisomerase inhibitor',
                'tubulin polymerization inhibitor',
                'src inhibitor|tubulin polymerization inhibitor',
                'HDAC inhibitor',
                'DNA methyltransferase inhibitor']

    targ_moa = ['src inhibitor',
                'MEK inhibitor',
                'EGFR inhibitor',
                'Abl kinase inhibitor|Bcr-Abl kinase inhibitor',
                'ALK tyrosine kinase receptor inhibitor',
                'RAF inhibitor|VEGFR inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RAF inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'EGFR inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'NFkB pathway inhibitor|proteasome inhibitor',
                'mTOR inhibitor',
                'mTOR inhibitor|PI3K inhibitor',
                'Bcr-Abl kinase inhibitor|ephrin inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|src inhibitor|tyrosine kinase inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'BCL inhibitor',
                'KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'PLK inhibitor',
                'Abl kinase inhibitor|Bcr-Abl kinase inhibitor|src inhibitor',
                'PI3K inhibitor',
                'AKT inhibitor',
                'BCL inhibitor|MCL1 inhibitor',
                'FLT3 inhibitor|JAK inhibitor',
                'RAF inhibitor',
                'Aurora kinase inhibitor|Bcr-Abl kinase inhibitor|FLT3 inhibitor|JAK inhibitor',
                'NFkB pathway inhibitor',
                'RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'VEGFR inhibitor',
                "Bruton's tyrosine kinase (BTK) inhibitor",
                'CDK inhibitor',
                'CHK inhibitor',
                'FGFR inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RAF inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|src inhibitor',
                'KIT inhibitor|VEGFR inhibitor',
                'proteasome inhibitor',
                'Abl kinase inhibitor|Aurora kinase inhibitor|FLT3 inhibitor',
                'CDK inhibitor|cell cycle inhibitor|MCL1 inhibitor',
                'cell cycle inhibitor|PLK inhibitor',
                'FGFR inhibitor',
                'FGFR inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'FGFR inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor',
                'JAK inhibitor']

    cyto_drugs = drug_info[drug_info['MOA'].isin(
        cyto_moa)]['Harmonized Name'].values
    targ_drugs = drug_info[drug_info['MOA'].isin(
        targ_moa)]['Harmonized Name'].values

    # cytotoxic - cytotoxic pairs
    cyto_vals = []
    for i, j in combinations(cyto_drugs, 2):
        cyto_vals.append(df.at[i, j])
    cyto_vals = np.array(cyto_vals)

    # targeted - targeted pairs
    targ_vals = []
    for i, j in combinations(targ_drugs, 2):
        targ_vals.append(df.at[i, j])
    targ_vals = np.array(targ_vals)
    targ_vals = targ_vals[~np.isnan(targ_vals)]

    # targeted - cytotoxic pairs
    mesh = np.array(np.meshgrid(targ_drugs, cyto_drugs))
    combi = mesh.T.reshape(-1, 2)
    cyto_targ_pairs = []
    for i, j in combi:
        v = df.at[i, j]
        if np.isnan(v):
            v = df.at[j, i]
        cyto_targ_pairs.append(v)
    cyto_targ_pairs = np.array(cyto_targ_pairs)
    cyto_targ_pairs = cyto_targ_pairs[~np.isnan(cyto_targ_pairs)]
    return vals, cyto_vals, targ_vals, cyto_targ_pairs


def import_pdx_data():
    """Import Gao et al. (2015) PDX suppl. data.

    Returns:
        (pd.DataFrame, pd.DataFrame): a tuple of drug response dataframe and cancer type info dataframe.
    """    
    info = pd.read_excel(f'{EXPERIMENTAL_DATA_DIR}/Gao2015_suppl_table.xlsx',
                         sheet_name='PCT raw data',
                         engine='openpyxl', dtype=str)

    dat = pd.read_excel(f'{EXPERIMENTAL_DATA_DIR}/Gao2015_suppl_table.xlsx',
                        sheet_name='PCT curve metrics',
                        engine='openpyxl', dtype=str)

    info = info[['Model', 'Tumor Type', 'Treatment']].drop_duplicates()
    info = info.sort_values('Tumor Type')
    tumor_types = info[['Model', 'Tumor Type']
                       ].drop_duplicates().set_index('Model').iloc[:-1, :]
    return (dat, tumor_types)


def get_distribution_report(all_pairs: np.array, cyto_pairs: np.array, 
                            targ_pairs: np.array, cyto_targ_pairs: np.array):
    df = pd.DataFrame(np.nan, index=['all_pairs', 'cytotoxic_pairs', 'targeted_pairs', 'cyto+targeted_pairs'],
                      columns=['N', 'mean', 'sd', '95% lower range', '95% lower range'])
    pair_list = [all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs]
    for i in range(len(pair_list)):
        pair = pair_list[i]
        df.iat[i, 0] = len(pair)
        df.iat[i, 1] = np.mean(pair)
        df.iat[i, 2] = np.std(pair)
        df.iat[i, 3] = np.quantile(pair, 0.025)
        df.iat[i, 4] = np.quantile(pair, 0.975)
    return df.round(3)

    
def main():
    pdx_df, tumor_types = import_pdx_data()
    drug1, drug2 = 'cetuximab', '5FU'
    fig = draw_corr_pdx(pdx_df, tumor_types, drug1, drug2)
    fig.savefig(f'{FIG_DIR}/CRC_{drug1}_{drug2}_BestAvgResponse_corr.pdf',
                bbox_inches='tight', pad_inches=0.1)

    # use cell line data (CTRPv2)
    cell_info, drug_info, cancer_type, ctrp = import_ctrp_data()
    drug_pairs = [('5-Fluorouracil', 'Docetaxel'), ('5-Fluorouracil', 'Lapatinib'),
                  ('Gemcitabine', 'Oxaliplatin'), ('Methotrexate', 'Oxaliplatin'),
                  ('5-Fluorouracil', 'Topotecan'), ('5-Fluorouracil', 'Oxaliplatin'),
                  ('nintedanib', 'Docetaxel'), ('Ifosfamide', 'Doxorubicin'),
                  ('Lapatinib', 'Paclitaxel') , ('Selumetinib', 'Dacarbazine')]
    for drug1, drug2 in drug_pairs:
        fig = draw_corr_cell(ctrp, cancer_type, drug1, drug2)
        fig.savefig(f'{FIG_DIR}/{drug1}_{drug2}_AUC_corr.pdf')

    for drug1, drug2 in [('Dabrafenib', 'Trametinib')]:
        fig = draw_corr_cell(ctrp, cancer_type, drug1, drug2, only_cancer_type='Melanoma')
        fig.savefig(f'{FIG_DIR}/{drug1}_{drug2}_AUC_corr.pdf')
    
    # cell line drug pair spearmanr distribution
    all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs = prepare_ctrp_agg_data(drug_info)
    fig = draw_ctrp_spearmanr_distribution(all_pairs, cyto_pairs, 
                                           targ_pairs, cyto_targ_pairs)
    fig.savefig(f'{FIG_DIR}/CTRPv2_corr_distributions.pdf',
                bbox_inches='tight', pad_inches=0.1)
    
    dist_report = get_distribution_report(
        all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs)
    dist_report.to_csv(f'{TABLE_DIR}/experimental_correlation_report.csv')


if __name__ == '__main__':
    main()
