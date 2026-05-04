import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as plt_lines
from corner import hist2d
from popsummary.popresult import PopulationResult

copula_result = PopulationResult(
    '../data_release/BBHCorr_qchieffCopulaCorrelationModel.h5'
)

linear_result = PopulationResult(
    '../data_release/BBHCorr_qchieffLinearCorrelationModel.h5'
)

plot_kwargs = dict(
    bins=50,
    smooth=0.9,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=28),
    levels=(0.99,0.90,0.50),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    tick_kwags=dict(fontsize=12),
    max_n_ticks=5,
)

copula_q_draws = copula_result.get_fair_population_draws(parameters=['mass_ratio']).flatten()
copula_chi_eff_draws = copula_result.get_fair_population_draws(parameters=['chi_eff']).flatten()

linear_q_draws = linear_result.get_fair_population_draws(parameters=['mass_ratio']).flatten()
linear_chi_eff_draws = linear_result.get_fair_population_draws(parameters=['chi_eff']).flatten()

matplotlib.rc('text', **{'usetex':True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

labels = ['Copula', 'Linear']
colors = dict(
    Copula = 'tab:blue',
    Linear = 'tab:orange',
)
cmaps = dict(
    Copula = 'Blues',
    Linear = 'Oranges',
)

fig = hist2d(
    copula_q_draws,
    copula_chi_eff_draws,
    color='tab:blue',
    **plot_kwargs
)
fig = hist2d(
    linear_q_draws,
    linear_chi_eff_draws,
    fig=fig,
    color='tab:orange',
    alpha=0.5,
    **plot_kwargs
)
plt.xlabel('$q$')
plt.ylabel('$\\chi_\mathrm{eff}$')
plt.grid(linestyle=':')
plt.legend(
    handles=[plt_lines.Line2D([], [], linestyle='-' , color=colors[result_type], label=fr'\textsc{{{result_type}}}') for result_type in labels],
    frameon=False)
plt.savefig('../figures/figure_12.pdf', bbox_inches='tight')