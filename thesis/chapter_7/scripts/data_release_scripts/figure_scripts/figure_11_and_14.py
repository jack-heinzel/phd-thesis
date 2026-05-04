import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as plt_lines
from corner import hist2d
import numpy as np
from popsummary.popresult import PopulationResult

matplotlib.rc('text', **{'usetex':True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size']  = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 20

plot_kwargs = dict(
    bins=30,
    smooth=0.05,
    levels=(0.99,0.9,0.5),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=5,
)

labels = ['Linear', 'Spline']
param_types = ['mu', 'sigma']
colors = dict(
    Linear='tab:blue',
    Spline='tab:orange',
)
params = ['q', 'z']
result_files = dict(
    Linear_q = '../data_release/BBHCorr_qchieffLinearCorrelationModel.h5',
    Linear_z = '../data_release/BBHCorr_zchieffLinearCorrelationModel.h5',
    Spline_q = '../data_release/BBHCorr_qchieffSplineCorrelationModel.h5',
    Spline_z = '../data_release/BBHCorr_zchieffSplineCorrelationModel.h5'
)
lims = dict(q=[0,1], z=[0,1.5])
upper_redshift = 0.79

###
spline_point = 0.6
tol = 0.025

linear_result = PopulationResult(
    result_files['Linear_q']
)
spline_result = PopulationResult(
    result_files['Spline_q']
)

linear_mu = linear_result.get_hyperparameter_samples(hyperparameters='mu_chieff_1')
linear_sigma = linear_result.get_hyperparameter_samples(hyperparameters='ln_sigma_chieff_1')

q, spline_mu_all = spline_result.get_rates_on_grids(f'mu_chieff')
_, spline_sigma_all = spline_result.get_rates_on_grids(f'sigma_chieff')
spline_sigma_all = np.log(spline_sigma_all)

q = q[0]

spline_mu = np.array([])
spline_sigma = np.array([])
q_1 = spline_point-tol
q_2 = spline_point+tol

for i in range(spline_mu_all.shape[0]):
    mu_1 = np.interp(q_1, q, spline_mu_all[i])
    mu_2 = np.interp(q_2, q, spline_mu_all[i])
    sigma_1 = np.interp(q_1, q, spline_sigma_all[i])
    sigma_2 = np.interp(q_2, q, spline_sigma_all[i])

    spline_mu = np.append(spline_mu, (mu_2-mu_1)/(2*tol))
    spline_sigma = np.append(spline_sigma, (sigma_2-sigma_1)/(2*tol))
###

for param in params:
    plot_points = dict()

    for result_type in labels:
        result = PopulationResult(result_files[f'{result_type}_{param}'])

        for param_type in param_types:
            positions, _res = result.get_rates_on_grids(f'{param_type}_chieff')
            res = np.quantile(_res, q=(0.05, 0.5, 0.95), axis=0)
            for ii in range(3):
                plot_points[f'{result_type}_{param_type}_{ii}'] = res[ii]

    if param == 'z':
        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2, 1, top=0.98, bottom=0.02, hspace=0.05)
    else:
        fig = plt.figure(figsize=(8,10.5))
        gs = GridSpec(2, 1, top=0.98, bottom=0.45, hspace=0.05)
        gs2 = GridSpec(1, 1, top=0.38, bottom=0.02)
    axs = []
    for ii, param_type in enumerate(param_types):
        axs.append(fig.add_subplot(gs[ii]))
        for result_type in labels:
            axs[ii].plot(
                positions[0], plot_points[f'{result_type}_{param_type}_1'],
                color=colors[result_type], label=fr'\textsc{{{result_type}}}'
            )
            axs[ii].fill_between(
                positions[0],
                plot_points[f'{result_type}_{param_type}_0'], plot_points[f'{result_type}_{param_type}_2'],
                color=colors[result_type], alpha=0.2
            )
        axs[ii].set(
        xlabel=f'${param}$', ylabel=f'$\{param_type}_\mathrm{{eff}}({param})$',
        xlim=lims[param]
        )
        axs[ii].grid(linestyle=':')
        axs[ii].label_outer()
        if param == 'z':
            axs[ii].axvline(upper_redshift, color='gray', linestyle='--') 
    axs[1].set(yscale='log')
    axs[0].legend(frameon=False)
    if param == 'q':
        post_labels = [r'\textsc{Linear}', rf'\textsc{{Spline}} $(q={spline_point})$']
        ax2 = fig.add_subplot(gs2[0])
        hist2d(
            spline_mu,
            spline_sigma,
            color=colors['Spline'],
            ax=ax2,
            new_fig=False,
            **plot_kwargs
        )
        hist2d(
            linear_mu,
            linear_sigma,
            color=colors['Linear'],
            ax=ax2,
            new_fig=False,
            **plot_kwargs
        )
        ax2.grid(linestyle=':')
        ax2.set(xlabel='$\delta \mu_{\mathrm{eff}|q}$', ylabel='$\delta \ln \sigma_{\mathrm{eff}|q}$')
        ax2.legend(
            handles=[plt_lines.Line2D([], [], linestyle='-' , color=colors[lab], label=post_labels[i])
                     for i, lab in enumerate(labels)],
        frameon=False,  loc='upper right')
        ax2.axvline(0, color='gray', linestyle='--')
        ax2.axhline(0, color='gray', linestyle='--') 

    if param == "q":
        fname = "../figures/figure_11.pdf"
    elif param == "z":
        fname = "../figures/figure_14.pdf"
    plt.savefig(fname, bbox_inches='tight')