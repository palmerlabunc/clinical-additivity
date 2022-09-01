import pandas as pd
import numpy as np
from itertools import combinations


def import_ctrp_data():
    """Imports all CTRP data

    Returns:
        pd.DataFrame: cell line information
        pd.DataFrame: drug information
        pd.Series: cancer type information
        pd.DataFrame: viablility data
        pd.DataFrame: pairwise correlation data
    """
    indir = '../data/experimental/'
    # metadata
    cell_info = pd.read_csv(indir + 'CTRPv2_CCL.csv', index_col=0)
    drug_info = pd.read_csv(indir + 'CTRPv2_drug.csv')
    # drop non-cancer cell types
    noncancer = cell_info[cell_info['Cancer_Type_HH'] == 'Non-cancer'].index
    cell_info = cell_info.drop(noncancer)
    cancer_type = cell_info['Cancer_Type_HH'].squeeze()

    # viability data
    ctrp = pd.read_csv(indir + 'Recalculated_CTRP_12_21_2018.txt',
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


def get_ctrp_corr_data(df, cancer_type, drug_a, drug_b, metric='CTRP_AUC'):
    """Filters and returns data frame ready to calculate correlation

    Args:
        df (pd.DataFrame): CTRPv2 viability data
        cancer_type (pd.Series): cancer type information
        drug_a (str): name of drug A
        drug_b (str): naem of drug B
        metric (str): viability metric to use (Default: CTRP_AUC)

    Returns:
        pd.DataFrame: ready to calculate correlation.
    """
    a = df[df['Harmonized_Compound_Name'] ==
           drug_a][['Harmonized_Cell_Line_ID', metric]]
    b = df[df['Harmonized_Compound_Name'] ==
           drug_b][['Harmonized_Cell_Line_ID', metric]]
    # take mean if duplicated cell lines
    a = a.groupby('Harmonized_Cell_Line_ID').mean()
    b = b.groupby('Harmonized_Cell_Line_ID').mean()
    merged = pd.concat([a, b, cancer_type], axis=1, join='inner')
    merged.columns = [drug_a, drug_b, 'Cancer_Type_HH']

    # drop cancer type if 1st quantile is less then 0.8 both drugs
    by_type = merged.groupby('Cancer_Type_HH').quantile(0.25)
    valid_types = by_type[(by_type < 0.8).sum(axis=1) != 0].index
    merged = merged[merged['Cancer_Type_HH'].isin(valid_types)]
    return merged


def prepare_ctrp_agg_data(drug_info):
    """Prepare data frame to plot distribution of correlations for drug pairs.

    Args:
        drug_info (pd.DataFrame): drug info data

    Returns:
        np.array: array of correlation values between all drug pairs
        np.array: array of correlation values between cytotoxic drug pairs
        np.array: array of correlation values between targeted drug pairs
        np.array: array of correlation values between cytotoxic-targeted drug pairs
    """
    indir = '../data/experimental/'
    df = pd.read_csv(
        indir + 'CTRPv2_clincal_active_drug_pairwise_corr.csv', index_col=0)
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
    indir = '../data/experimental/'
    info = pd.read_excel(indir + 'Gao2015_suppl_table.xlsx',
                         sheet_name='PCT raw data',
                         engine='openpyxl', dtype=str)

    dat = pd.read_excel(indir + 'Gao2015_suppl_table.xlsx',
                        sheet_name='PCT curve metrics',
                        engine='openpyxl', dtype=str)

    info = info[['Model', 'Tumor Type', 'Treatment']].drop_duplicates()
    info = info.sort_values('Tumor Type')
    tumor_types = info[['Model', 'Tumor Type']
                       ].drop_duplicates().set_index('Model').iloc[:-1, :]
    return (dat, tumor_types)


def get_pdx_corr_data(df, tumor_types, drug1, drug2, metric='BestAvgResponse'):
    """Prepare data to calculate correlation between drug response to drug1 and 2.

    Args:
        df (pd.DataFrame): drug response data
        tumor_types (pd.DataFrame): tumor type dataframe (model, tumor type info)
        drug1 (str): name of drug 1
        drug2 (str): name of drug 2
        metric (str, optional): drug response metric. Defaults to 'BestAvgResponse'.

    Returns:
        pd.DataFrame: 
    """    
    a = df[df['Treatment'] == drug1].set_index('Model')[metric].astype(float)
    b = df[df['Treatment'] == drug2].set_index('Model')[metric].astype(float)
    merged = pd.concat([a, b, tumor_types], axis=1, join='inner')
    merged.columns = [drug1, drug2, 'Tumor Type']
    return merged



