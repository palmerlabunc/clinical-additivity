from pathlib import Path
from plotting.plot_doses_difference import plot_doses_difference
from plotting.all_survival_qqplot import plot_all_survival_qqplot, qqplot_legends
from plotting.forest_plot import forest_plot
from plotting.r2_msd_histogram import plot_r2_histogram, plot_msd_histogram
from plotting.subgroup_boxplots import plot_ici_boxplot, plot_angio_boxplot, plot_mono_approval_boxplot, plot_HR_boxplot
from plotting.plot_survival_curves import plot_additive_survival, plot_between_survival, plot_hsa_survival
from plotting.forest_plot import forest_plot

def figure1(outdir):
    """Generate and export doses difference between monotherapy
    and combination trials.

    Args:
        outdir (str): output directory
    """    
    # Figure 1B
    fig = plot_doses_difference()
    fig.savefig(outdir + 'fig1b_doses_diffence.pdf')


def figure2(outdir):
    """ Generate and export forest plot of all combinations 
    including supplments (figure 2).

    Args:
        outdir (str): output directory
    """
    fig = forest_plot()
    fig.savefig(outdir + 'fig3_forest_plot.pdf')


def figure3(outdir):
    """ Generate and export survival curves of all combinations 
    (figure 3).

    Args:
        outdir (str): output directory
    """
    fig_part1 = plot_additive_survival()
    fig_part1.savefig(outdir + 'fig3_additive_survival_plots.pdf')
    
    fig_part2 = plot_between_survival()
    fig_part2.savefig(outdir + 'fig3_between_survival_plots.pdf')
    
    fig_part3 = plot_hsa_survival()
    fig_part3.savefig(outdir + 'fig3_hsa_survival_plots.pdf')


def figure4(outdir):
    """ Generate and export goodness of fit and subgroup analysis
    (figure 4).

    Args:
        outdir (str): output directory
    """
    # Figure 4A
    figa = plot_all_survival_qqplot()
    figa.savefig(outdir + 'fig4a_all_combo_qqplot.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    figa_legend = qqplot_legends()
    figa_legend.savefig(outdir + 'fig4a_legend.pdf')
    
    # Figure 4B
    figb = plot_r2_histogram()
    figb.savefig(outdir + 'fig4b_r2_histogram.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    # Figure 4C
    figc = plot_msd_histogram()
    figc.savefig(outdir + 'fig4c_msd_histogram.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    # Figure 4D
    # ICI combination
    figd_ici = plot_ici_boxplot()
    figd_ici.savefig(outdir + 'fig4d_ici_boxplot.pdf',
                      bbox_inches='tight', pad_inches=0.1)
    # Anti-angiogenesis combination
    figd_angio = plot_angio_boxplot()
    figd_angio.savefig(outdir + 'fig4d_angio_boxplot.pdf',
                        bbox_inches='tight', pad_inches=0.1)
    # Monotherapy Approval
    figd_mono = plot_mono_approval_boxplot()
    figd_mono.savefig(outdir + 'fig4d_mono_boxplot.pdf',
                       bbox_inches='tight', pad_inches=0.1)
    # HR < 0.6?
    figd_hr = plot_HR_boxplot()
    figd_hr.savefig(outdir + 'fig4d_hr_boxplot.pdf',
                     bbox_inches='tight', pad_inches=0.1)


def main():
    outdir = '../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)

    #figure1(outdir)
    #figure2(outdir)
    #figure3(outdir)
    figure4(outdir)


if __name__ == '__main__':
    main()
