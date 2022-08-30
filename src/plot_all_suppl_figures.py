from pathlib import Path
from experimental_correlation import import_ctrp_data, import_pdx_data, prepare_ctrp_agg_data
from hsa_add_diff import hsa_add_diff
from lognormal_fitting import fit_lognormal
from plotting.plot_placebo import plot_placebo
from plotting.plot_experimental_correlation import draw_corr_cell, draw_corr_pdx, draw_ctrp_spearmanr_distribution
from plotting.plot_concept_figure import plot_concept_figure
from plotting.plot_hsa_add_diff import plot_hsa_add_diff_vs_lognormal, reg_hsa_add_diff_vs_lognormal

#FIXME figure numbers are subject to change

def suppl_fig1(outdir):
    """Generate and export all placebo/BSC survival distributions (suppl fig.1)

    Args:
        outdir (str): filepath to output directory
    """
    fig = plot_placebo()
    fig.savefig(outdir + 'suppl_fig1_placebo_curves.pdf')


def suppl_fig2(outdir):
    """Generate and export all experimental correlations (suppl fig.2).
    Each panel will be exported to a separate figure.

    Args:
        outdir (str): filepath to output directory
    """    
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


def suppl_fig3(outdir):
    """Generate and export additivity concept figure (suppl fig.3)
    This figure explains the connection between Bliss Independence and
    our model of clinical drug additivity.

    Args:
        outdir (str): filepath to output directory
    """    
    fig = plot_concept_figure()
    fig.savefig(outdir + 'suppl_fig3_concept_figure.pdf')


def suppl_fig4(outdir):
    """Generate and export figure about explaining why lognormal fit sigma vs. 
    difference between HSA and additivity. (suppl fig.3)

    Args:
        outdir (str): filepath to output directory
    """
    lognorm_df = fit_lognormal()
    diff_df = hsa_add_diff()
    results = reg_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    print(f'pearsonr={results.rvalue}\nslope={results.slope}\npvalue={results.pvalue}')
    fig = plot_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    fig.savefig(outdir + 'suppl_fig4_hsa_additivity_sigma.pdf')


def main():
    outdir = '../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)

    #suppl_fig1(outdir)
    #suppl_fig2(outdir)
    #suppl_fig3(outdir)
    suppl_fig4(outdir)


if __name__ == '__main__':
    main()

