import numpy as np
import scipy as sp
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
bgp_chieff = PopulationResult(fname="../data_release/BBHCorr_MassSpinCorrelatedBGPModel.h5")
skewnormal_chieff = PopulationResult(fname="../data_release/BBHSpin_EpsSkewNormalChiEff.h5")
bspline_chieff =  PopulationResult(fname='../data_release/BBHSpin_BSplineChiEff.h5')
chieff_chip = PopulationResult(fname='../data_release/BBHSpin_GaussianChiEffChiP.h5')
chieff_q_corr = PopulationResult(fname='../data_release/BBHCorr_qchieffSplineCorrelationModel.h5')

# chi-eff rates on grids
chi_eff_pos, chi_eff_rates = bgp_chieff.get_rates_on_grids('effective_inspiral_spin_norm')
chi_eff_pos2, chi_eff_rates2 = bspline_chieff.get_rates_on_grids('p(chi_eff)')
chi_eff_pos3, chi_eff_rates3 = skewnormal_chieff.get_rates_on_grids('Effective inspiral spin')
chi_eff_pos4, chi_eff_rates4 = chieff_chip.get_rates_on_grids('chi_eff')
chi_eff_pos5, chi_eff_rates5 = chieff_q_corr.get_rates_on_grids('chi_eff')

# set up dict with everything
plotting_dict = {
    'BGP':{'x':chi_eff_pos, 'p(x)':chi_eff_rates[0], 'color':pf.color_bgp, 'label':'Binned Gaussian Process', 'zorder':0},
    'BSpline':{'x':chi_eff_pos2, 'p(x)':chi_eff_rates2, 'color':pf.color_bspline, 'label':'B-Spline', 'zorder':4},
    'Skewnormal':{'x':chi_eff_pos3, 'p(x)':chi_eff_rates3, 'color':pf.color_skewnorm_eff, 'label':'Skewnormal Effective Spin', 'zorder':3},
    'Gaussian':{'x':chi_eff_pos4, 'p(x)':chi_eff_rates4, 'color':pf.color_gaussian_eff,'label':'Gaussian Effective Spins', 'zorder':2}, 
    'Correlation':{'x':chi_eff_pos5, 'p(x)':chi_eff_rates5, 'color':pf.color_corr, 'label':r'$(q,\chi_\mathrm{eff})$ Spline Correlation', 'zorder':1},
}
# see that palette is colorblind friendly here: https://davidmathlogic.com/colorblind/#%23A1C935-%23FF5733-%236A6BAD-%230099FF-%23582500

# more plotting macros
figsize = (pf.FIG_WIDTH, pf.FIG_WIDTH)
handles = [Line2D([], [], color=v['color'], label=v['label']) for v in plotting_dict.values()]

# Function to plot the traces
def plot_pdf(ax, _x_pos, _x_rates, color='C0', label='', zorder=0): 

    x_pos, x_rates = pf.format_inputs(_x_pos, _x_rates)

    quantiles = np.quantile(x_rates, (0.05, 0.95), axis=0)
    ppd = np.average(x_rates, axis=0)

    ppd_color = pf.darken_color(color, factor=0.8)

    ax.plot(x_pos, ppd, color=color, ls='-', rasterized=True, label=label, zorder=zorder)
    ax.plot(x_pos, ppd, color=ppd_color, ls='-', rasterized=True, zorder=zorder)
    ax.fill_between(x_pos, quantiles[0], y2=quantiles[1], color=color, alpha=0.2, rasterized=True, zorder=zorder)

'''
Make figure
'''
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(7,2)

## plot traces
ax = fig.add_subplot(gs[:3, :])
for v in plotting_dict.values(): 
    plot_pdf(ax, v['x'], v['p(x)'], color=v['color'],zorder=v['zorder'])
ax.set_xlabel(r'$\chi_\mathrm{eff}$')
ax.set_ylabel(r'$p(\chi_\mathrm{eff})$')
ax.set_xlim(-0.7, 0.7)
ax.axvline(0, ls='--', color='gray')
ax.set_yscale('log')
ax.set_ylim(1e-2, 2e1)
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)
ax.legend(handles=handles, fontsize=8, loc='upper left')

# load in gwtc-3 data for comparison 
with open(pf.gwtc3_folder+'GaussianSpin/gaussian-spin-xeff-xp-ppd-data.json', 'r') as f:
    gwtc3_data = json.load(f)
plotting_dict['GWTC-3'] = {'x':gwtc3_data['chi_eff_grid'], 'p(x)':gwtc3_data['chi_eff_pdfs'], 'color':'#6a6bad', 'ls':'--'}

def plot_hist(ax, samples, v, **hist_kws): 
    if 'ls' in v.keys(): 
        ax.hist(samples, **hist_kws, color=v['color'], ls=v['ls'], histtype='step', lw=1.5) 
    else: 
        ax.hist(samples, **hist_kws, color=v['color'], alpha=0.2) 
        ax.hist(samples, **hist_kws, color=v['color'], histtype='step', lw=1.5) 

## plot histograms of negative chi-eff probes

# 1st percentile
def get_1st_percentile(x, px): 
    x_pos, x_rates = pf.format_inputs(x, px)
    f_total = np.trapezoid(x_rates, x=x_pos)
    cum_sum = sp.integrate.cumulative_trapezoid(x_rates, x=x_pos) / np.expand_dims(f_total, -1)    
    first_percentile = np.array([np.interp(0.01, c, x_pos[:-1]) for c in cum_sum])
    return first_percentile

ax = fig.add_subplot(gs[3:5, 0])
bounds = (-0.7,0)
hist_kws = dict(density=True, bins=np.linspace(*bounds,50), rasterized=True)
for v in plotting_dict.values(): 
    samples = get_1st_percentile(v['x'], v['p(x)'])
    plot_hist(ax, samples, v, **hist_kws) 
        
ax.set_xlim(*bounds)
ax.set_xlabel(r'$\chi_{\mathrm{eff},1\%}$')
ax.set_ylabel('Probability density')
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)

# fraction neg:
def get_f_negative(x, px, x0=0): 
    x_pos, x_rates = pf.format_inputs(x, px)
    mask = x_pos < x0
    f_neg = np.trapezoid(x_rates[:,mask], x=x_pos[mask])
    f_total = np.trapezoid(x_rates, x=x_pos)
    return f_neg/f_total

ax = fig.add_subplot(gs[3:5, 1])
bounds = (0, 0.75)
hist_kws = dict(density=True, bins=np.linspace(*bounds,50), rasterized=True)
for v in plotting_dict.values(): 
    samples = get_f_negative(v['x'], v['p(x)'])
    plot_hist(ax, samples, v, **hist_kws) 
ax.set_xlim(*bounds)
ax.set_xlabel(r'Fraction $\chi_\mathrm{eff} < 0$')
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)


ax = fig.add_subplot(gs[5:, 0])
bounds = (-5, 0)
hist_kws = dict(density=True, bins=np.linspace(*bounds,50), rasterized=True)
for k,v in plotting_dict.items(): 
    samples = np.log10(get_f_negative(v['x'], v['p(x)'], x0=-0.3) / 0.16 )
    mask = np.isnan(samples) | np.isinf(samples)
    
    plot_hist(ax, samples[~mask], v, **hist_kws) 
    
    print(k, np.quantile(10**samples[~mask], 0.90))
    
ax.set_xlim(*bounds)
ax.set_xlabel(r'$\log_{10}$(HM Fraction)')
ax.set_ylabel('Probability density')
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)


# asymmetry:
def get_asymmetry(x, px): 
    x_pos, x_rates = pf.format_inputs(x, px)
    asym=np.zeros(len(x_rates))
    for i,xr in enumerate(x_rates): 
        x0 = x_pos[np.argmax(xr)]
        mask = x_pos < x0
        f_neg = np.trapezoid(xr[mask], x=x_pos[mask])
        f_pos = np.trapezoid(xr[~mask], x=x_pos[~mask])
        f_total = np.trapezoid(xr, x=x_pos)
        asym[i] = (f_pos-f_neg)/f_total
    return asym


plotting_dict.pop('GWTC-3')

ax = fig.add_subplot(gs[5:, 1])
bounds = (-0.2, 1)
hist_kws = dict(density=True, bins=np.linspace(*bounds,50), rasterized=True)
for k,v in plotting_dict.items(): 
    if k=='Gaussian':
        ax.axvline(0, color=v['color']) ## gaussian is not skewed about its peak
    else:
        samples = get_asymmetry(v['x'], v['p(x)'])
        plot_hist(ax, samples, v, **hist_kws) 
        
ax.set_xlim(*bounds)
ax.set_xlabel(r'Skew about peak')
ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)

plt.subplots_adjust(hspace=1.5, wspace=0.2)
plt.savefig(pf.figs_folder+'figure_21.pdf', bbox_inches='tight')
plt.close()