import matplotlib.pyplot as plt
import numpy as np
import popsummary
import h5py
import plot_funcs_jaxen as pf
import dill

pf.setup()

bptp_o4 = popsummary.popresult.PopulationResult('/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk/init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5')
bs_o4 = popsummary.popresult.PopulationResult('/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/jaxen/november2025_pe_update/result_files/dec4_bspline_iid.h5')
# bs_n =  popsummary.popresult.PopulationResult(fname = '/home/jaxen.godfrey/o4a-astro-dist/o4a-astrodist/analyses/jaxen/may5_bspline_iid_noMinM.h5')

pp_q, pp_lo, pp_ppd, pp_hi = pf.get_03b_plp_ppds('/home/jaxen.godfrey/o4a-astro-dist/o4a-astrodist/analyses/jaxen', mass_1 = False, mass_ratio=True)

bs_q, bs_q_pdfs = pf.get_params(bs_o4, 'rate_vs_mass_ratio_at_z0-2', rate = False)

bptp_q, bptp_q_pdfs = pf.get_params(bptp_o4, 'mass_ratio')
# bptp_z, bptp_z_pdfs = pf.get_params(bptp_o4, 'redshift', rate = False)
# bptp_rate = bptp_o4.get_hyperparameter_samples(hyperparameters='rate')
# z02 = np.sum(bptp_z <= 0.2)
# bptp_q_pdfs = (bptp_q_pdfs / bptp_rate.reshape(-1,1)) * bptp_z_pdfs[:,z02].reshape(-1,1)

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
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = 10, loc = 'upper left')
plt.grid(ls = ':', alpha = 0.2)
plt.savefig('/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/figures/mass_ratio_default.pdf')