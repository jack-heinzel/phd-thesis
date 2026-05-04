import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from popsummary.popresult import PopulationResult

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

plt.rcParams['font.size']  = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 20

mass_spin_result = PopulationResult('../data_release/BBHCorr_MassSpinCorrelatedBGPModel.h5')

fig = plt.figure(figsize=(8,12))
gs = GridSpec(3, 1, top=0.98, bottom=0.02, hspace=0.25)
axs = []

axs.append(fig.add_subplot(gs[0]))
pos, rates = mass_spin_result.get_rates_on_grids('effective_inspiral_spin_norm')
pos = pos[:,0]
low_rates, med_rates, hi_rates = np.quantile(rates, q=(0.05,0.5,0.95), axis=1)
axs[0].fill_between(pos, low_rates[0], hi_rates[0], color='tab:blue', alpha=0.3)
axs[0].fill_between(pos, low_rates[1], hi_rates[1], color='tab:orange', alpha=0.3)
axs[0].plot(pos, med_rates[0], color='tab:blue', label='$m \\in (30 M_\\odot, 40 M_\\odot)$')
axs[0].plot(pos, med_rates[1], color='tab:orange', label='$m \\notin (30 M_\\odot, 40 M_\\odot)$')
axs[0].set(
    xlabel='$\\chi_\\mathrm{eff}$',
    ylabel='$p(\\chi_\\mathrm{eff}|m_1,m_2)$',
    xlim=[-0.3, 0.4],
    ylim=[0, 16],
)
axs[0].grid(linestyle=':')
axs[0].legend(frameon=False)

for i, mass in enumerate(['primary', 'secondary']):
    i += 1
    axs.append(fig.add_subplot(gs[i]))
    pos, rates = mass_spin_result.get_rates_on_grids(f'{mass}_mass_norm')
    pos = pos[:,0]
    low_rates, med_rates, hi_rates = np.quantile(rates, q=(0.05,0.5,0.95), axis=1)
    axs[i].fill_between(pos, low_rates[0], hi_rates[0], color='tab:blue', alpha=0.3)
    axs[i].fill_between(pos, low_rates[1], hi_rates[1], color='tab:orange', alpha=0.3)
    axs[i].plot(pos, med_rates[0], color='tab:blue', label='$\\chi_\\mathrm{eff} \\in (-0.05, 0.05)$')
    axs[i].plot(pos, med_rates[1], color='tab:orange', label='$\\chi_\\mathrm{eff} \\in (0.1, 0.2)$')
    axs[i].set(
        xlabel=f'$m_{i} \ [M_\\odot]$',
        ylabel=f'$p(m_{i}|\\chi_\\mathrm{{eff}})$',
        xlim=[5, 100],
        ylim=[1e-6, 1e0],
        yscale='log'
    )
    axs[i].set_yticks([1e-5,1e-3,1e-1])
    axs[i].grid(linestyle=':')
    axs[i].legend(frameon=False)
plt.savefig('../figures/figure_13.pdf', bbox_inches='tight')