from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt
import numpy as np

import plot_funcs_bbh_mass as pf

import seaborn as sns

dom = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_DominantMode.h5')
sub = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_SubdominantMode.h5')
bp1p = PopulationResult(fname = '../data_release/BBHMassSpinRedshift_BrokenPowerLawOnePeak_GaussianComponentSpins_PowerLawRedshift.h5')

dom_m, dom_pdfs = pf.get_params(dom, 'mass_1', rate = True)
sub_m, sub_pdfs = pf.get_params(sub, 'mass_1', rate = True)
bp1p_m, bp1p_pdfs = pf.get_params(bp1p, 'mass_1', rate = True)

colors = ['#FE6100', '#785EF0']

# Main plot
pf.setup()
fig, ax = plt.subplots(figsize = (5.25,3.75))

pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(10,100), yrange=(2e-3,1e0))
pf.plot_90CI(ax, dom_m, dom_pdfs, color = colors[0], median = False, label = r'$\mathrm{Mode}\, 1$', fill_alpha = 0.6)
pf.plot_90CI(ax, dom_m, dom_pdfs, color = pf.darken(colors[0]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})
pf.plot_90CI(ax, sub_m, sub_pdfs, color = colors[1], median = False, lw = 0.8, label = r'$\mathrm{Mode}\, 2$', plot_kwargs={'alpha': 0.8}, fill_alpha = 0.6)
pf.plot_90CI(ax, sub_m, sub_pdfs, color = pf.darken(colors[1]), median = False, fill = False, secondary_ls ='-', plot_kwargs = {'alpha': 0.5})
pf.plot_90CI(ax, bp1p_m, bp1p_pdfs, color = 'k', median = False, fill = False, label = r'$\textsc{BP1P}$', plot_kwargs = {'alpha': 0.75})

# ax.axvline(np.median(data[:,0] + data[:,1]))
dom_mu2 = dom.get_hyperparameter_samples(hyperparameters = 'mpp_2').T[0]
dom_sig2 = dom.get_hyperparameter_samples(hyperparameters = 'sigpp_2').T[0]

sub_mu2 = sub.get_hyperparameter_samples(hyperparameters = 'mpp_2').T[0]
sub_sig2 = sub.get_hyperparameter_samples(hyperparameters = 'sigpp_2').T[0]
ax.axvspan(np.percentile(dom_mu2 + dom_sig2, 5), np.percentile(dom_mu2+dom_sig2, 95), color = 'k', alpha = 0.1)

ax.set_xlim(11,70)
ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
ax.legend(bbox_to_anchor=(0,0.4))

# Create an inset axes: [left, bottom, width, height] in axes coords
inset_ax = ax.inset_axes([0.78, 0.62, 0.2, 0.34])
sns.kdeplot(x=dom_mu2, y=dom_sig2, ax=inset_ax, levels=4, fill=True, color=colors[0], alpha = 0.5, common_grid=True, cut = True)
sns.kdeplot(x=dom_mu2, y=dom_sig2, ax=inset_ax, levels=4, fill=False, color=pf.darken(colors[0]), linewidths=1.5, alpha = 0.5, common_grid=True, cut = True)
sns.kdeplot(x=sub_mu2, y=sub_sig2, ax=inset_ax, levels=4, fill=True, color=colors[1], common_grid=True, cut = True, alpha = 0.5)
sns.kdeplot(x=sub_mu2, y=sub_sig2, ax=inset_ax, levels=4, fill=False, color=pf.darken(colors[1]), linewidths=1.5, alpha = 0.5, common_grid=True, cut = True)
inset_ax.tick_params(labelsize=9)
inset_ax.set_xlim(25,40)
inset_ax.set_xlabel(r'$\mu_2$', fontdict = {'fontsize': 15})
inset_ax.set_ylabel(r'$\sigma_2$', fontdict = {'fontsize': 15})
plt.tight_layout()
plt.savefig('../figures/figure_16.pdf')
