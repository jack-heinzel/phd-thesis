import matplotlib.pyplot as plt
import numpy as np
from popsummary.popresult import PopulationResult
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
color = 'tab:blue'

result = PopulationResult('/home/christian.adamcewicz/projects/o4/vamana_gwtc4/vamana_gwtc4.h5')

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

fig, axs = plt.subplots(2, 1, figsize=(10,10))
axs[0].plot(q_ax[1:-1], savgol_filter(p50_qsz, 20, 2), color=color)
axs[0].fill_between(q_ax[1:-1], savgol_filter(p5_qsz, 20, 2), savgol_filter(p95_qsz, 20, 2), color=color, alpha=0.2)
axs[0].set(
    xlabel='$q$',
    ylabel='$S_z$',
    xlim=[q_ax[1], q_ax[-2]],
)
axs[1].plot(mch_ax, savgol_filter(p50_mchsz, 50, 2), color=color)
axs[1].fill_between(mch_ax, np.zeros_like(mch_ax), savgol_filter(p90_mchsz, 50, 2), color=color, alpha=0.2)
axs[1].set(
    xlabel='$\\mathcal{M} [M_\\odot]$',
    ylabel='$|S_z|$',
    xlim=[mch_ax[0], mch_ax[-1]],
    ylim=[-0.006, None]
)
for ax in axs:
    ax.grid(linestyle=':')
plt.savefig('../../figures/mass_spin_correlation_vamana.pdf', bbox_inches='tight')