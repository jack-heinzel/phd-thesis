import matplotlib.pyplot as plt
import numpy as np
import popsummary
import h5py
import plot_funcs_jaxen as pf

pf.setup()

plpeak_o3 = popsummary.popresult.PopulationResult('/Users/jaxengodfrey/o4a_paper/o4a-astrodist/data/popsummary_baseline_O3.h5')
extended_o4 = popsummary.popresult.PopulationResult('/Users/jaxengodfrey/o4a_paper/o4a-astrodist/data/popsummary_extended_O4.h5')
cyb_o4 = popsummary.popresult.PopulationResult('/Users/jaxengodfrey/o4a_paper/o4a-astrodist/data/cyb_jan7_trunczprior.h5')

with h5py.File('/Users/jaxengodfrey/o4a_paper/o4a-astrodist/data/spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data_best_measured.h5', 'r') as f:
    spline_ppd = np.asarray(f['ppd'])
    spline_m1_pdfs = np.asarray(f['lines']['mass_1'])
    mass_1 = np.linspace(2, 100, 1000)

plpeak_m1, plpeak_m1_pdfs = pf.get_params(plpeak_o3, 'rate_vs_primary_mass_at_z_0.2')
extended_m1, extended_m1_pdfs = pf.get_params(extended_o4, 'rate_vs_primary_mass_at_z_0.2')
cyb_m1, cyb_m1_pdfs = pf.get_params(cyb_o4, 'rate_vs_mass_1_at_z0-2')

plt.style.use('seaborn-v0_8-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1,2,figsize=(12,3.5), tight_layout=True)
ax1 = ax[0]
ax2 = ax[1]
pf.setup_mass_plot(ax1, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0.3), label_kwargs=dict(fontsize=12))
pf.plot_90CI(ax1, extended_m1, extended_m1_pdfs,  color = colors[1], label = r'\textsc{GWTC-4}')
pf.plot_90CI(ax1, plpeak_m1, plpeak_m1_pdfs,  color = 'k', label = r'\textsc{GWTC-3}', fill = False, lw = 0.8)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.legend(fontsize=12)

pf.setup_mass_plot(ax2, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0.3), label_kwargs=dict(fontsize=12))
pf.plot_90CI(ax2, cyb_m1, cyb_m1_pdfs, color = colors[2], label = r'\textsc{GWTC-4}')
pf.plot_90CI(ax2, mass_1, spline_m1_pdfs,  color = 'k', label = r'\textsc{GWTC-3}', fill = False, lw = 0.8)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(fontsize=12)
plt.savefig('/Users/jaxengodfrey/o4a_paper/o4a-astrodist/figures/primary_mass_gwtc3_vs_gwtc4.pdf')

