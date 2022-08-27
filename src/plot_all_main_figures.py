from plotting.all_survival_qqplot import plot_all_survival_qqplot, qqplot_legends
from plotting.forest_plot import forest_plot
from plotting.r2_msd_histogram import plot_r2_histogram, plot_msd_histogram
from plotting.subgroup_boxplots import plot_ici_boxplot, plot_angio_boxplot, plot_mono_approval_boxplot, plot_HR_boxplot
from plotting.plot_survival_curves import plot_additive_survival, plot_between_survival, plot_hsa_survival
from plotting.forest_plot import forest_plot
from pathlib import Path

def figure2(outdir):
    """ Generate and export figure 2.

    Args:
        outdir (str): output directory
    """
    fig2 = forest_plot()
    fig2.savefig(outdir + 'fig3_forest_plot.pdf')


def figure3(outdir):
    """ Generate and export figure 3.

    Args:
        outdir (str): output directory
    """
    fig3_part1 = plot_additive_survival()
    fig3_part1.savefig(outdir + 'fig3_additive_survival_plots.pdf')
    
    fig3_part2 = plot_between_survival()
    fig3_part2.savefig(outdir + 'fig3_between_survival_plots.pdf')
    
    fig3_part3 = plot_hsa_survival()
    fig3_part3.savefig(outdir + 'fig3_hsa_survival_plots.pdf')


def figure4(outdir):
    """ Generate and export figure 4.

    Args:
        outdir (str): output directory
    """
    # Figure 4A
    fig4a = plot_all_survival_qqplot()
    fig4a.savefig(outdir + 'fig4a_all_combo_qqplot.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    fig4a_legend = qqplot_legends()
    fig4a_legend.savefig(outdir + 'fig4a_legend.pdf')
    
    # Figure 4B
    fig4b = plot_r2_histogram()
    fig4b.savefig(outdir + 'fig4b_r2_histogram.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    # Figure 4C
    fig4c = plot_msd_histogram()
    fig4c.savefig(outdir + 'fig4c_msd_histogram.pdf',
                  bbox_inches='tight', pad_inches=0.1)

    # Figure 4D
    # ICI combination
    fig4d_ici = plot_ici_boxplot()
    fig4d_ici.savefig(outdir + 'fig4d_ici_boxplot.pdf',
                      bbox_inches='tight', pad_inches=0.1)
    # Anti-angiogenesis combination
    fig4d_angio = plot_angio_boxplot()
    fig4d_angio.savefig(outdir + 'fig4d_angio_boxplot.pdf',
                        bbox_inches='tight', pad_inches=0.1)
    # Monotherapy Approval
    fig4d_mono = plot_mono_approval_boxplot()
    fig4d_mono.savefig(outdir + 'fig4d_mono_boxplot.pdf',
                       bbox_inches='tight', pad_inches=0.1)
    # HR < 0.6?
    fig4d_hr = plot_HR_boxplot()
    fig4d_hr.savefig(outdir + 'fig4d_hr_boxplot.pdf',
                     bbox_inches='tight', pad_inches=0.1)


def main():
    outdir = '../figures/'
    new_directory = Path(outdir)
    new_directory.mkdir(parents=True, exist_ok=True)

    figure2(outdir)
    figure3(outdir)
    figure4(outdir)


if __name__ == '__main__':
    main()
