import matplotlib.pyplot as plt
import numpy as np
import popsummary
import plot_funcs_bbh_mass as pf

pf.setup()

bptp_o4 = popsummary.popresult.PopulationResult('../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')

bs_o4 = popsummary.popresult.PopulationResult('../data_release/BBHMassSpinRedshift_BSplineIID.h5')


bptp_m1, bptp_m1_pdfs = pf.get_params(bptp_o4, 'mass_1')

bs_m1, bs_m1_pdfs = pf.get_params(bs_o4, 'rate_vs_mass_1_at_z0-2', rate = False)

plp_m1, plplow, plppd, plphi = pf.get_03b_plp_ppds('../gwtc3_data/analyses/PowerLawPeak/')

color1 = '#FE6100'
color2 = '#648FFF'
colors = [0, color1, color2]

bptp_pdfs = [bptp_m1_pdfs]
bptp_ms = [bptp_m1]

bs_pdfs = [bs_m1_pdfs]
bs_ms = [bs_m1]

for i in range(1):

    fig = plt.figure(figsize=(11,4.5), tight_layout=True)
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=0.2)
    pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))
    ax.plot(plp_m1, plppd, color = 'k', lw = 1.5, alpha = 0.5, ls = '-')
    ax.plot(plp_m1, plplow, color = 'k', lw = 0.75, alpha = 0.7, ls = '--', label = r'\textsc{Power Law + Peak}, \textsc{GWTC-3.0}')
    ax.plot(plp_m1, plphi, color = 'k', lw = 0.75, alpha = 0.7, ls = '--')
    pf.plot_90CI(ax, bs_ms[i], bs_pdfs[i], color = colors[2], label = r'\textsc{B-Spline}, \textsc{GWTC-4.0}', fill_alpha = 0.4)
    pf.plot_90CI(ax, bptp_ms[i], bptp_pdfs[i], color = colors[1], label = r'\textsc{Broken Power Law + 2 Peaks}, \textsc{GWTC-4.0}', fill_alpha = 0.4)
    plt.xticks(np.arange(0,100,10))

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 2, 0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.grid(ls = ':', alpha = 0.2)
    plt.savefig('../figures/figure_3.pdf')
    plt.close()
