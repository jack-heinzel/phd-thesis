import matplotlib.pyplot as plt
import numpy as np
from popsummary.popresult import PopulationResult

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

plot_bins = [0,1,2,3,4]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

mass_redshift_result = PopulationResult('/home/anarya.ray/public_html/gppop_redshift_popsummary_rerun_ulogm.hdf5')

redshift_edges = [0.01, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2.0]
redshift_bins = [[redshift_edges[i], redshift_edges[i+1]] for i in range(len(redshift_edges)-1)]

fig, axs = plt.subplots(2, 1, figsize=(10,10))
for i, mass in enumerate(['primary', 'secondary']):
    pos, rates = mass_redshift_result.get_rates_on_grids(f'{mass}_mass')
    pos = pos[:,0]
    low_rates, med_rates, hi_rates = np.quantile(rates, q=(0.05,0.5,0.95), axis=0)
    for ii, bin in enumerate(plot_bins):
        axs[i].fill_between(pos, low_rates[bin], hi_rates[bin], color=colors[ii], alpha=0.1)
        axs[i].plot(pos, low_rates[bin], color=colors[ii], alpha=1, linestyle='-', label=f'$z \\in ({redshift_bins[bin][0]}, {redshift_bins[bin][1]})$')
        axs[i].plot(pos, hi_rates[bin], color=colors[ii], alpha=1, linestyle='-')
    axs[i].set(
        xlabel=f'$m_{i+1} \ [M_\\odot]$',
        ylabel=f'$p(m_{i+1}|z)$',
        xlim=[5, 100],
        ylim=[1e-4, 1e2],
        yscale='log'
    )
    axs[i].grid(linestyle=':')
axs[0].legend(frameon=False)
plt.savefig('../../figures/mass_redshift_correlated_bgp.pdf', bbox_inches='tight')
