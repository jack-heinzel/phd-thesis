import numpy as np
from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from scipy.signal import savgol_filter

matplotlib.rc('text', **{'usetex':True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 20

fig = plt.figure(figsize=(14,14))
gs0 = GridSpec(1, 1, top=0.97, bottom=2/3+0.035, left=0.32, right=0.68)
gs1 = GridSpec(1, 2, top=2/3-0.035, bottom=1/3+0.035, wspace=0.25)
gs2 = GridSpec(1, 2, top=1/3-0.035, bottom=0.03, wspace=0.25)

### fig 1

axs0 = [fig.add_subplot(gs0[i]) for i in range(1)]

res = PopulationResult('../data_release/BBHCorr_zchieffCopulaCorrelationModel.h5')
kappa = res.get_hyperparameter_samples(hyperparameters='kappa')

bins = 20
color = 'tab:blue'

axs0[0].hist(
    kappa, bins=bins, density=True, histtype='stepfilled', color=color, alpha=0.2
)
axs0[0].hist(
    kappa, bins=bins, density=True, histtype='step', color=color, alpha=1
)
axs0[0].axvline(0, color='black', linestyle=':')
axs0[0].set(xlabel='$\kappa_{\chi_\mathrm{eff}, z}$')
axs0[0].grid(linestyle=':')

quants = np.quantile(kappa, q=(0.05,0.5,0.95))
med = round(quants[1], 2)
minus = round(quants[1]-quants[0], 2)
plus = round(quants[2]-quants[1], 2)

### fig 2

axs1 = [fig.add_subplot(gs1[i]) for i in range(2)]

plot_bins = [0,1,2,3,4]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

mass_redshift_result = PopulationResult('../data_release/BBHCorr_MassRedshiftCorrelatedBGPModel.h5')

redshift_edges = [0.01, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2.0]
redshift_bins = [[redshift_edges[i], redshift_edges[i+1]] for i in range(len(redshift_edges)-1)]

for i, mass in enumerate(['primary', 'secondary']):
    pos, rates = mass_redshift_result.get_rates_on_grids(f'{mass}_mass')
    pos = pos[:,0]
    low_rates, med_rates, hi_rates = np.quantile(rates, q=(0.05,0.5,0.95), axis=0)
    for ii, bin in enumerate(plot_bins):
        axs1[i].fill_between(pos, low_rates[bin], hi_rates[bin], color=colors[ii], alpha=0.1)
        axs1[i].plot(pos, low_rates[bin], color=colors[ii], alpha=1, linestyle='-', label=f'$z \\in ({redshift_bins[bin][0]}, {redshift_bins[bin][1]})$')
        axs1[i].plot(pos, hi_rates[bin], color=colors[ii], alpha=1, linestyle='-')
    axs1[i].set(
        xlabel=f'$m_{i+1} \ [M_\\odot]$',
        ylabel=f'$p(m_{i+1}|z)$',
        xlim=[5, 100],
        ylim=[1e-4, 1e2],
        yscale='log'
    )
    axs1[i].grid(linestyle=':')
axs1[1].legend(frameon=False)

### fig 3

axs2 = [fig.add_subplot(gs2[i]) for i in range(2)]

result = PopulationResult('../data_release/BBHMass_FlexibleMixtures.h5')

pos_grid_2D, rates_grid_2D = result.get_rates_on_grids('chirp_mass_aligned_spin_joint')

mch_ax = np.unique(pos_grid_2D[0])
sz_ax = np.unique(pos_grid_2D[1])
mean_rate_mchsz = rates_grid_2D.reshape(len(sz_ax), len(mch_ax))
mean_rate_mchsz = np.transpose(mean_rate_mchsz)

p50_mchsz, p90_mchsz = [], []
for ii, _ in enumerate(mch_ax):
    pdf = mean_rate_mchsz[ii]
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    p50_mchsz.append(sz_ax[np.where(cdf > 0.5)[0][0]])
    p90_mchsz.append(sz_ax[np.where(cdf > 0.9)[0][0]])

pos_grid_2D, rates_grid_2D = result.get_rates_on_grids('mass_ratio_aligned_spin_joint')

q_ax = np.unique(pos_grid_2D[0])
sz_ax = np.unique(pos_grid_2D[1])
mean_rate_qsz = rates_grid_2D.reshape(len(sz_ax), len(q_ax))
mean_rate_qsz = np.transpose(mean_rate_qsz)

p5_qsz, p50_qsz, p95_qsz = [], [], []
for ii in range(1, len(q_ax) - 1):
    pdf = mean_rate_qsz[ii]
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    p5_qsz.append(sz_ax[np.where(cdf > 0.05)[0][0]])
    p50_qsz.append(sz_ax[np.where(cdf > 0.5)[0][0]])
    p95_qsz.append(sz_ax[np.where(cdf > 0.95)[0][0]])

axs2[0].plot(q_ax[1:-1], savgol_filter(p50_qsz, 20, 2), color=color)
axs2[0].fill_between(q_ax[1:-1], savgol_filter(p5_qsz, 20, 2), savgol_filter(p95_qsz, 20, 2), color=color, alpha=0.2)
axs2[0].set(
    xlabel='$q$',
    ylabel='$p(\\chi_z|q)$',
    xlim=[q_ax[1], q_ax[-2]],
)
axs2[1].plot(mch_ax, savgol_filter(p50_mchsz, 50, 2), color=color)
axs2[1].fill_between(mch_ax, np.zeros_like(mch_ax), savgol_filter(p90_mchsz, 50, 2), color=color, alpha=0.2)
axs2[1].set(
    xlabel='$\\mathcal{M} \ [M_\\odot]$',
    ylabel='$p(|\\chi_z| | \\mathcal{M}$)',
    xlim=[mch_ax[0], mch_ax[-1]],
    ylim=[-0.006, None]
)
for ax in axs2:
    ax.grid(linestyle=':')

plt.savefig('../figures/figure_22.pdf', bbox_inches='tight')