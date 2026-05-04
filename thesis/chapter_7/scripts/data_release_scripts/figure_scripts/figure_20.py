import matplotlib.pyplot as plt
import numpy as np
from popsummary.popresult import PopulationResult
import plot_funcs_bbh_mass as pf
from plot_funcs_bbh_mass import darken, plot_90CI


ar = PopulationResult(fname = '../data_release/BBHMass_Autoregressive.h5')
bgp = PopulationResult(fname = '../data_release/BBHCorr_MassRedshiftCorrelatedBGPModel.h5')
bs = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BSplineIID.h5')
vm = PopulationResult(fname = '../data_release/BBHMass_FlexibleMixtures.h5')

default = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')

m, mpdfs = pf.get_params(default, 'mass_1')

keys = ['dR_dlnm1_at_z_0.2_enforcedMinM', 'primary_mass', 'rate_vs_mass_1_at_z0-2', 'primary_mass']
files = [ar, bgp, bs, vm]
names = ['ar', 'bgp', 'bs', 'vm']

pdfs = {}

for i in range(len(keys)):
    
    x, px = files[i].get_rates_on_grids(grid_key = keys[i])
    x = x.T[0] if names[i] == 'bgp' else x[0]

    if 'dR_dlnm1' in keys[i]:
        px = px / x.reshape(-1,1)
        px = px.T
    
    if names[i] == 'vm':
        z, pz = files[i].get_rates_on_grids(grid_key = 'rate_z_evol')
        z = z[0]
        zo2 = np.sum(z <= 0.2)
        rate_02 = pz[:,zo2]
        px = px * rate_02.reshape(-1,1)

    pdfs[names[i]] = {'x': x, 'px': px}


pf.setup()

color1 = '#FE6100'
color2 = '#648FFF'

fig = plt.figure(figsize=(11,4.5), tight_layout=True)
ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.2)
pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))

plot_90CI(ax, pdfs['bgp']['x'], pdfs['bgp']['px'][:,1], color = '#929591', median = False, label = r'$\textsc{BGP}$')
plot_90CI(ax, pdfs['bgp']['x'], pdfs['bgp']['px'][:,1], color = darken('#929591'), median = False, fill = False, secondary_ls = '-', lw = 1)

plot_90CI(ax, pdfs['ar']['x'], pdfs['ar']['px'], color = '#15B01A', median = False, label = r'$\textsc{AR}$')
plot_90CI(ax, pdfs['ar']['x'], pdfs['ar']['px'], color = darken('#15B01A'), median = False, fill = False, secondary_ls = '-', lw = 1)

plot_90CI(ax, pdfs['bs']['x'], pdfs['bs']['px'], color = color2, median = False, label = r'$\textsc{B-Spline}$', fill_alpha = 0.5)
plot_90CI(ax, pdfs['bs']['x'], pdfs['bs']['px'], color = darken(color2), median = False, fill = False, secondary_ls = '-', lw = 1)

plot_90CI(ax, pdfs['vm']['x'], pdfs['vm']['px'], color = color1, label = r'$\textsc{Flexible Mixtures}$', median = False, fill_alpha = 0.5)
plot_90CI(ax, pdfs['vm']['x'], pdfs['vm']['px'], color = darken(color1), median = False, fill = False, secondary_ls = '-', lw = 1)

plot_90CI(ax, m, mpdfs, color = 'k', median = False, fill = False, secondary_ls = '-', lw = 2)


ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(5,100)
ax.legend()
plt.grid(ls = ':', alpha = 0.2)

plt.savefig('../figures/figure_20a.pdf')
plt.close()

bs_n = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BSplineIID_NoMinM.h5')

keys = ['dR_dq_at_z_0.2', 'dR_dq_at_z_0.2_enforcedMinM', 'rate_vs_mass_ratio_at_z0-2', 'rate_vs_mass_ratio_at_z0-2_noMinM', 'mass_ratio']
files = [ar, ar, bs, bs_n, vm]
names = ['ar_n', 'ar', 'bs', 'bs_n', 'vm']

pdfs = {}

for i in range(len(keys)):
    
    x, px = files[i].get_rates_on_grids(grid_key = keys[i])
    x = x[0]

    if (keys[i] == 'dR_dq_at_z_0.2_enforcedMinM') | (keys[i] == 'dR_dq_at_z_0.2'):
        px = px.T
    
    if names[i] == 'vm':
        z, pz = files[i].get_rates_on_grids(grid_key = 'rate_z_evol')
        z = z[0]
        zo2 = np.sum(z <= 0.2)
        rate_02 = pz[:,zo2]
        px = px * rate_02.reshape(-1,1)
    pdfs[names[i]] = {'x': x, 'px': px}

q, qpdfs = pf.get_params(default, 'mass_ratio')


fig = plt.figure(tight_layout=True)
ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.2)

pf.setup_mass_ratio_plot(ax)

plot_90CI(ax, pdfs['ar']['x'], pdfs['ar']['px'], color = '#15B01A', median = False, label = r'$\textsc{AR}$')
plot_90CI(ax, pdfs['ar']['x'], pdfs['ar']['px'], color = darken('#15B01A'), median = False, fill = False, secondary_ls = '-', lw = 1)
plot_90CI(ax, pdfs['ar_n']['x'], pdfs['ar_n']['px'], color = darken('#15B01A'), median = False, fill = False, secondary_ls = '--', lw = 2, plot_kwargs={'alpha':0.4})

plot_90CI(ax, pdfs['bs']['x'], pdfs['bs']['px'], color = color2, median = False, label = r'$\textsc{B-Spline}$', fill_alpha = 0.5)
plot_90CI(ax, pdfs['bs']['x'], pdfs['bs']['px'], color = darken(color2), median = False, fill = False, secondary_ls = '-', lw = 1)
plot_90CI(ax, pdfs['bs_n']['x'], pdfs['bs_n']['px'], color = darken(color2), median = False, fill = False, secondary_ls = '--', lw = 2, plot_kwargs={'alpha':0.4})

plot_90CI(ax, pdfs['vm']['x'], pdfs['vm']['px'], color = color1, label = r'$\textsc{Flexible Mixtures}$', median = False, fill_alpha = 0.5)
plot_90CI(ax, pdfs['vm']['x'], pdfs['vm']['px'], color = darken(color1), median = False, fill = False, secondary_ls = '-', lw = 1)

plot_90CI(ax, q, qpdfs, color = 'k', median = False, fill = False, secondary_ls = '-', lw = 2)
ax.legend()
plt.grid(ls = ':', alpha = 0.2)
plt.savefig('../figures/figure_20b.pdf')
plt.close()