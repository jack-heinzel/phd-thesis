import numpy as np
from popsummary.popresult import PopulationResult
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import plot_funcs_bbh_spin as pf
import json

pf.setup()

'''
Setup
'''

# load the data
skewnormal = PopulationResult(fname=pf.DATADIR + "BBHSpin_EpsSkewNormalChiEff.h5")

# chi-eff and chi-p rates on grids
chi_eff_pos, chi_eff_rates = skewnormal.get_rates_on_grids('Effective inspiral spin')
chi_p_pos, chi_p_rates = skewnormal.get_rates_on_grids('Effective precessing spin')

# samples for the skew parameter, epsilon
eps_samples = skewnormal.get_hyperparameter_samples(hyperparameters=['eps_chi_eff']).flatten()

print('Fraction with eps < 0:', np.sum(eps_samples < 0)/len(eps_samples))

# load in gwtc-3 data for comparison 
with open(pf.gwtc3_folder+'GaussianSpin/gaussian-spin-xeff-xp-ppd-data.json', 'r') as f:
    gwtc3_data = json.load(f)

# more plotting macros
figsize = (1.3*pf.FIG_WIDTH, 0.8*pf.FIG_WIDTH/2)

'''
Make chi-eff marginal plot
'''

print('Making chi-eff marginal traces plot')

fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1,5)
i_split = 3

# plot chi-eff
ax = fig.add_subplot(gs[:, :i_split])
pf.plot_traces(ax, chi_eff_pos, chi_eff_rates, color=pf.color_skewnorm_eff, fill_between=True)
pf.plot_traces(ax, gwtc3_data['chi_eff_grid'], gwtc3_data['chi_eff_pdfs'], just_quants=True, color=pf.color_gaussian_eff)

# line at chieff=0 to guide the eye
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)

ax.set_xlabel(r'$\chi_\mathrm{eff}$')
ax.set_ylabel(r'$p(\chi_\mathrm{eff})$')
ax.set_ylim(0, )
ax.set_xlim(-0.35, 0.65)
ax.axvline(0, ls='--', color='gray')

# legend 
handles = [
    Line2D([], [], color=pf.color_skewnorm_eff, label='Skewnormal Effective Spin,\nGWTC-4.0'),
    Line2D([], [], color=pf.color_gaussian_eff, ls='--', label='Gaussian Effective Spins,\nGWTC-3.0')
]

ax.legend(handles=handles, loc='upper right', fontsize=7)

# make inset axes for histogram the skew parameter, epsilon
hist_kws = dict(density=True, bins=20, lw=1)
w, h, dx = 0.35, 0.24, 0.02
axin = ax.inset_axes([1-w, 0.45, w-dx, h]) # args: [x0, y0, width, height]
axin.hist(eps_samples, **hist_kws, color=pf.color_skewnorm_eff)
axin.set_xlabel(r'$\epsilon$ (skew)', fontsize=10)
axin.set_xlim(-1, )
axin.set_yticks([])
axin.grid(color='silver', alpha=0.5, ls=':')

'''
Make chi-p marginal plot
'''

ax = fig.add_subplot(gs[:, i_split:])
pf.plot_traces(ax, chi_p_pos, chi_p_rates, color=pf.color_skewnorm_eff, fill_between=True)
pf.plot_traces(ax, gwtc3_data['chi_p_grid'], gwtc3_data['chi_p_pdfs'], just_quants=True, color=pf.color_gaussian_eff)
ax.set_xlabel(r'$\chi_\mathrm{p}$')
ax.set_ylabel(r'$p(\chi_\mathrm{p})$')
ax.set_ylim(0, )
ax.set_xlim(0, 1)

ax.grid(color='silver', alpha=0.5, ls=':')

plt.subplots_adjust(wspace=1)
plt.savefig(pf.figs_folder+'figure_9.pdf', bbox_inches='tight')

plt.close()