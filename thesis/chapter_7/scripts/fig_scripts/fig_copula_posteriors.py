import numpy as np
from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

analysis = 'chieff_z'

labels = dict(
    q_chieff = '$\\kappa_{(q, \\chi_\\mathrm{eff})}$',
    m_chieff = '$\\kappa_{(m_1, \\chi_\\mathrm{eff})}$',
    m_z = '$\\kappa_{(m_1, z)}$',
    chieff_z = '$\\kappa_{(\\chi_\\mathrm{eff}, z)}$',
)

res_files = dict(
    q_chieff = '/home/christian.adamcewicz/public_html/o4a_astrodist/qchieffCopulaCorrelationModelPopsummaryV2.h5',
    m_chieff = '/home/christian.adamcewicz/public_html/o4a_astrodist/mchieffCopulaCorrelationModelPopsummaryV2.h5',
    m_z = '/home/christian.adamcewicz/public_html/o4a_astrodist/mzCopulaCorrelationModelPopsummaryV2.h5',
    chieff_z = '/home/christian.adamcewicz/public_html/o4a_astrodist/zchieffCopulaCorrelationModelPopsummaryV2.h5'
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 14
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
color = 'tab:blue'
bins = 20

fig, axs = plt.subplots()

res = PopulationResult(res_files[analysis])
kappa = res.get_hyperparameter_samples(hyperparameters='kappa')
axs.hist(
    kappa, bins=bins, density=True, histtype='stepfilled', color=color, alpha=0.2
)
axs.hist(
    kappa, bins=bins, density=True, histtype='step', color=color, alpha=1
)
axs.axvline(0, color='black', linestyle=':')
axs.set(xlabel=labels[analysis])
axs.grid(linestyle=':')

quants = np.quantile(kappa, q=(0.05,0.5,0.95))
med = round(quants[1], 2)
minus = round(quants[1]-quants[0], 2)
plus = round(quants[2]-quants[1], 2)
axs.set_title(f"${med}_{{-{minus}}}^{{+{plus}}}$")

perc_correlated = 0.5
_kappa = med
if _kappa < 0:
    while _kappa < 0:
        perc_correlated += 0.001
        if perc_correlated > 1:
            break
        _kappa = np.quantile(
            kappa,
            perc_correlated
        )
else:
    while _kappa > 0:
        perc_correlated -= 0.001
        if perc_correlated < 0:
            break
        _kappa = np.quantile(
            kappa,
            perc_correlated
        )
    perc_correlated = 1 - perc_correlated
        
plt.savefig('../../figures/copula_posteriors.pdf', bbox_inches='tight')