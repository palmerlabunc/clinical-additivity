from pathlib import Path
from experimental_correlation import import_ctrp_data, import_pdx_data, prepare_ctrp_agg_data
from hsa_add_diff import hsa_add_diff
from lognormal_fitting import fit_lognormal
from lognormal_examples import get_lognormal_examples
from predict_success import predict_success, calc_correlation
from plotting.plot_placebo import plot_placebo
from plotting.plot_experimental_correlation import draw_corr_cell, draw_corr_pdx, draw_ctrp_spearmanr_distribution
from plotting.plot_concept_figure import plot_concept_figure
from plotting.plot_hsa_add_diff import plot_hsa_add_diff_vs_lognormal, corr_hsa_add_diff_vs_lognormal
from plotting.plot_survival_curves_suppl import plot_additivity_suppl, plot_between_hsa_suppl
from plotting.plot_predict_success import plot_predict_success
from plotting.plot_lognormal_examples import plot_lognormal_examples
from plotting.plot_CDK46_ER import plot_abema_vs_palbo
#FIXME figure numbers are subject to change

def suppl_fig1(outdir):
    """Generate and export all placebo/BSC survival distributions (suppl fig.1)

    Args:
        outdir (str): file path to output directory
    """
    fig = plot_placebo()
    fig.savefig(outdir + 'suppl_fig1_placebo_curves.pdf')


def suppl_fig2(outdir):
    """Generate and export all experimental correlations (suppl fig.2).
    Each panel will be exported to a separate figure.

    Args:
        outdir (str): file path to output directory
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
        outdir (str): file path to output directory
    """    
    fig = plot_concept_figure()
    fig.savefig(outdir + 'suppl_fig3_concept_figure.pdf')


def suppl_fig4(outdir):
    """Generate and export figure about explaining why lognormal fit sigma vs. 
    difference between HSA and additivity. (suppl fig.4)

    Args:
        outdir (str): filepath to output directory
    """
    lognorm_df = fit_lognormal()
    diff_df = hsa_add_diff()
    r, p = corr_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    print(f'pearsonr={r}\npvalue={p}')
    fig = plot_hsa_add_diff_vs_lognormal(lognorm_df, diff_df)
    fig.savefig(outdir + 'suppl_fig4_hsa_additivity_sigma.pdf')


def suppl_fig5(outdir):
    """Generate and export extended survival curves of both monotherapies, 
    combination, predicted additivity, and HSA. (suppl fig.5)

    Args:
        outdir (str): file path to output directory
    """    
    fig_add = plot_additivity_suppl()
    fig_add.savefig(outdir +  'suppl_fig5_additive_survival.pdf',
                    bbox_inches='tight', pad_inches=0.1)
    fig_ind = plot_between_hsa_suppl()
    fig_ind.savefig(outdir + 'suppl_fig5_between_hsa_survival.pdf',
                    bbox_inches='tight', pad_inches=0.1)


def suppl_fig6(outdir):
    """Generate and export plots showing how variability in monotherapy responses 
    affect the HSA and additivity predictions. (suppl fig.6)
    A) lognormal survival curve examples
    B) Generate and export scatterplot of HR(obs combo vs. exp combo) vs.
    HR(exp combo vs. exp combo). Calcuate pearsonr between the two values.

    Args:
        outdir (str): file path to output directory
    """    
    # Panel A
    less_variable = get_lognormal_examples(20, 500, 2, 2.2, 0.5)
    more_variable = get_lognormal_examples(20, 500, 1, 1.5, 2)
    fig = plot_lognormal_examples(less_variable, more_variable)
    fig.savefig(outdir + 'suppl_fig6_explain_HSA_additive_difference.pdf')
    # Panel B
    results = predict_success()
    r_hsa, p_hsa = calc_correlation('HSA', results)
    r_add, p_add = calc_correlation('additivity', results)
    print("r_hsa={0:.02f}, p_hsa={1:.03f}, r_add={2:.02f}, p_add={3:.03f}".format(r_hsa, p_hsa, r_add, p_add))
    fig = plot_predict_success(results)
    fig.savefig(outdir + 'suppl_fig6_HR_combo_control_scatterplot.pdf', 
                bbox_inches='tight')


def suppl_fig7(outdir):
    """Generate and export plot comparing abemaciclib vs. palbociclib.

    Args:
        outdir (str): file path to output directory
    """    
    fig = plot_abema_vs_palbo()
    fig.savefig(outdir + 'suppl_fig7_Abemaciclib_vs_Palbociclib.pdf',
                bbox_inches='tight')


def main():
    outdir = '../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)
    #print("1")
    #suppl_fig1(outdir)
    #print("2")
    #suppl_fig2(outdir)
    #print("3")
    #suppl_fig3(outdir)
    #print("4")
    suppl_fig4(outdir)
    #print("5")
    #suppl_fig5(outdir)
    #print("6")
    #suppl_fig6(outdir)
    #suppl_fig7(outdir)



if __name__ == '__main__':
    main()

