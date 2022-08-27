from plotting.plot_placebo import plot_placebo
from plotting.plot_experimental_correlation import draw_corr_cell, draw_corr_pdx, draw_ctrp_spearmanr_distribution
from experimental_correlation import import_ctrp_data, import_pdx_data, prepare_ctrp_agg_data


def suppl_fig1(outdir):
    fig = plot_placebo()
    fig.savefig(outdir + 'suppl_fig1_placebo_curves.pdf')


def suppl_fig2(outdir):
    # use PDX data (PDXE)
    pdx_df, tumor_types = import_pdx_data()
    drug1, drug2 = 'cetuximab', '5FU'
    fig = draw_corr_pdx(pdx_df, tumor_types, drug1, drug2)
    fig.savefig(outdir + f'suppl_fig2_CRC_{drug1}_{drug2}_BestAvgResponse.pdf',
                bbox_inches='tight', pad_inches=0.1)

    # use cell line data (CTRPv2)
    cell_info, drug_info, cancer_type, ctrp = import_ctrp_data()
    drug_pairs = [('5-Fluorouracil', 'Docetaxel'), ('5-Fluorouracil', 'Lapatinib'),
                ('Gemcitabine', 'Oxaliplatin'), ('Methotrexate', 'Oxaliplatin'),
                ('5-Fluorouracil', 'Topotecan')]
    for drug1, drug2 in drug_pairs:
        fig = draw_corr_cell(ctrp, cancer_type, drug1, drug2)
        fig.savefig(outdir + f'suppl_fig2_{drug1}_{drug2}_correlation.pdf')

    for drug1, drug2 in [('Dabrafenib', 'Trametinib')]:
        fig = draw_corr_cell(ctrp, cancer_type, drug1, drug2, only_cancer_type='Melanoma')
        fig.savefig(outdir + f'suppl_fig2_{drug1}_{drug2}_correlation.pdf')
    
    # cell line drug pair spearmanr distribution
    all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs = prepare_ctrp_agg_data(drug_info)
    fig = draw_ctrp_spearmanr_distribution(all_pairs, cyto_pairs, 
                                          targ_pairs, cyto_targ_pairs)
    fig.savefig(outdir + 'suppl_fig2_CTRPv2_corr_distributions.pdf',
                bbox_inches='tight', pad_inches=0.1)


def main():
    outdir = '../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)

    figure2(outdir)
    figure3(outdir)
    figure4(outdir)


