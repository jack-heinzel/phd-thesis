import matplotlib.pyplot as plt
import popsummary
import plot_funcs_bbh_mass as pf

pf.setup()

bptp_o4 = popsummary.popresult.PopulationResult('../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
bs_o4 = popsummary.popresult.PopulationResult('../data_release/BBHMassSpinRedshift_BSplineIID.h5')

pp_q, pp_lo, pp_ppd, pp_hi = pf.get_03b_plp_ppds('../gwtc3_data/analyses/PowerLawPeak/', mass_1 = False, mass_ratio=True)

bs_q, bs_q_pdfs = pf.get_params(bs_o4, 'rate_vs_mass_ratio_at_z0-2', rate = False)

bptp_q, bptp_q_pdfs = pf.get_params(bptp_o4, 'mass_ratio')

color1 = '#FE6100'
color2 = '#648FFF'
colors = [0, color1, color2]

fig = plt.figure(figsize=(6,4), tight_layout=True)
ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.2)
ax.plot(pp_q, pp_ppd, color = 'k', lw = 1.5, alpha = 0.5, ls = '-')
ax.plot(pp_q, pp_lo, color = 'k', lw = 0.75, alpha = 0.7, ls = '--', label = r'\textsc{Power Law + Peak}, \textsc{GWTC-3.0}')
ax.plot(pp_q, pp_hi, color = 'k', lw = 0.75, alpha = 0.7, ls = '--')
pf.setup_mass_ratio_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0.3))
ax.set_ylim(1e-1,1e3)
pf.plot_90CI(ax, bs_q, bs_q_pdfs, color = colors[2], label = r'\textsc{B-Spline}, \textsc{GWTC-4.0}')
pf.plot_90CI(ax, bptp_q, bptp_q_pdfs, color = colors[1], label = r'\textsc{Broken Power Law + 2 Peaks}, \textsc{GWTC-4.0}')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = 10)
plt.grid(ls = ':', alpha = 0.2)
plt.savefig('../figures/figure_5.pdf')