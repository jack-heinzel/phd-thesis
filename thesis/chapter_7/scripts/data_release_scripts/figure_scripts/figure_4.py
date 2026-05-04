import matplotlib.pyplot as plt
import numpy as np
from popsummary.popresult import PopulationResult
import plot_funcs_bbh_mass as pf
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

color1 = '#648FFF'
color2 = '#FE6100'
cols = [color1, color2]

bp2p = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
o3b = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_GWTC3.h5')

bp2p_m, bp2p_pdfs = pf.get_params(bp2p, 'mass_1', rate = True)
o3b_m, o3b_pdfs = pf.get_params(o3b, 'mass_1', rate = True)

a1 = bp2p.get_hyperparameter_samples(hyperparameters = 'alpha_1').T[0]
a2 = bp2p.get_hyperparameter_samples(hyperparameters = 'alpha_2').T[0]
a1_o3 = o3b.get_hyperparameter_samples(hyperparameters = 'alpha_1').T[0]
a2_o3 = o3b.get_hyperparameter_samples(hyperparameters = 'alpha_2').T[0]

# Main plot
pf.setup()
fig, ax = plt.subplots(figsize = (9,4))

pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(3,100), yrange=(1e-3,30), xscale = 'log')
pf.plot_90CI(ax, bp2p_m, bp2p_pdfs, color = cols[1], median = True, label = r'$\textsc{GWTC-4.0}$')
pf.plot_90CI(ax, o3b_m, o3b_pdfs, color = 'k', fill = False, lw = 0.8, label = r'\textsc{GWTC-3.0}', plot_kwargs={'alpha': 0.8})
logticks = np.array([4, 6, 10, 20, 40, 60, 100])
ax.set_xticks(logticks)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xlim(8,100)
ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')

# Create an inset axes: [left, bottom, width, height] in axes coords
inset_ax = ax.inset_axes([0.78, 0.56, 0.2, 0.4])
sns.kdeplot(x=a1_o3, y=a2_o3, ax=inset_ax, levels=[0.05, 0.5, 0.95, 1], fill=True, color='k', alpha = 0.8, common_grid=True, cut = True)
sns.kdeplot(x=a1_o3, y=a2_o3, ax=inset_ax, levels=[0.05, 0.5, 0.95], fill=False, color='k', linewidths=1.5, alpha = 0.5, common_grid=True, cut = True)
sns.kdeplot(x=a1, y=a2, ax=inset_ax, levels=[0.05, 0.5, 0.95, 1], fill=True, color=cols[1], common_grid=True, cut = True)
sns.kdeplot(x=a1, y=a2, ax=inset_ax, levels=[0.05, 0.5, 0.95], fill=False, color=pf.darken(cols[1], factor = 0.5), linewidths=1.5, alpha = 0.8, common_grid=True, cut = True)
inset_ax.plot(np.linspace(-10,12), np.linspace(-10,10), color = 'k', ls = '--')
inset_ax.set_xlim(-4,12)
inset_ax.set_ylim(-4,12)
inset_ax.set_xticks([-3,0,3,6,9])
inset_ax.set_yticks([-3,0,3,6,9])
inset_ax.tick_params(labelsize=9)
inset_ax.set_xlabel(r'$\alpha_1$', fontdict = {'fontsize': 15})
inset_ax.set_ylabel(r'$\alpha_2$', fontdict = {'fontsize': 15})
plt.legend(bbox_to_anchor=(0.72, 0.98))
plt.tight_layout()
plt.savefig('../figures/figure_4.pdf')