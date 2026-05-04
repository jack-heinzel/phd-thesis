from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt

import plot_funcs_bbh_mass as pf


default = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
bp1p = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawOnePeak_GaussianComponentSpins_PowerLawRedshift.h5')
bp3p = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawThreePeaks_GaussianComponentSpins_PowerLawRedshift.h5')
plp = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_PowerLawPeak_GaussianComponentSpins_PowerLawRedshift.h5')

bs = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BSplineIID.h5')

default_x, default_y = pf.get_params(default, 'mass_1')
bp1p_x, bp1p_y = pf.get_params(bp1p, 'mass_1')
bp3p_x, bp3p_y = pf.get_params(bp3p, 'mass_1')
plp_x, plp_y = pf.get_params(plp, 'mass_1')
bs_x, bs_y = pf.get_params(bs, 'rate_vs_mass_1_at_z0-2', rate = False)

colors = ['#000000', '#FE6100', '#648FFF', '#DC267F']

pf.setup()

fig, axes = plt.subplots(nrows = 3, figsize = (8,9.5))

ax = axes[0]
pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))

pf.plot_90CI(ax, bs_x, bs_y, color = 'k', median = False, fill = False, secondary_ls ='--', lw = 0.8, label = r'$\textsc{BS}$')
pf.plot_90CI(ax, default_x, default_y, color = colors[0], median = False, label = r'$\textsc{BP2P}$', fill_alpha = 0.3)
pf.plot_90CI(ax, default_x, default_y, color = pf.darken(colors[0]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})

pf.plot_90CI(ax, plp_x, plp_y, color = colors[1], median = False, label = r'$\textsc{PLP}$', fill_alpha = 0.7)
pf.plot_90CI(ax, plp_x, plp_y, color = pf.darken(colors[1]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})
ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
ax.legend()

ax = axes[1]
pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))

pf.plot_90CI(ax, bs_x, bs_y, color = 'k', median = False, fill = False, secondary_ls ='--', lw = 0.8, label = r'$\textsc{BS}$')
pf.plot_90CI(ax, default_x, default_y, color = colors[0], median = False, label = r'$\textsc{BP2P}$', fill_alpha = 0.3)
pf.plot_90CI(ax, default_x, default_y, color = pf.darken(colors[0]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})

pf.plot_90CI(ax, bp1p_x, bp1p_y, color = colors[2], median = False, label = r'$\textsc{BP1P}$', fill_alpha = 0.7)
pf.plot_90CI(ax, bp1p_x, bp1p_y, color = pf.darken(colors[2]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})
ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
ax.legend()

ax = axes[2]
pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))

pf.plot_90CI(ax, bs_x, bs_y, color = 'k', median = False, fill = False, secondary_ls ='--', lw = 0.8, label = r'$\textsc{BS}$')
pf.plot_90CI(ax, default_x, default_y, color = colors[0], median = False, label = r'$\textsc{BP2P}$', fill_alpha = 0.3)
pf.plot_90CI(ax, default_x, default_y, color = pf.darken(colors[0]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})

pf.plot_90CI(ax, bp3p_x, bp3p_y, color = colors[3], median = False, label = r'$\textsc{BP3P}$', fill_alpha = 0.7)
pf.plot_90CI(ax, bp3p_x, bp3p_y, color = pf.darken(colors[3]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})
ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
ax.legend()

plt.tight_layout()
plt.savefig('../figures/figure_15.pdf')
