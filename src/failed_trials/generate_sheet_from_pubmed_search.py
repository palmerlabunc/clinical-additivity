import yaml
from webscrape_from_pubmed import find_title_abstract
import pandas as pd

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

INDIR = CONFIG['dir']['raw_all_phase3_data'] + '/PubMed_search'
SHEET_DICT = {'breast': 'csv-breastcanc-set.csv', 'colorectal': 'csv-colorectal-set.csv',
              'lung': 'csv-lungcancer-set.csv', 'prostate': 'csv-prostateca-set.csv'}
OUTDIR = CONFIG['dir']['all_phase3_data'] + '/2017-2020'

def main():
    for cancer_type in SHEET_DICT.keys():
        df = pd.read_csv(f'{INDIR}/{SHEET_DICT[cancer_type]}', header=0, index_col=None)
        tmp = pd.Series(df['PMID']).apply(find_title_abstract)
        new_df = pd.DataFrame(tmp.to_list(), index=df['PMID'].values,
                              columns=['Title', 'Abstract'])
        new_df.to_excel(f'{OUTDIR}/{cancer_type}_cancer_search.xlsx')


if __name__ == '__main__':
    main()