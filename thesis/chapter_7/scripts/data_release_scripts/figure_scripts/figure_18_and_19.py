import numpy as np
from popsummary.popresult import PopulationResult
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plot_funcs_bbh_spin as pf
import json 

pf.setup()

'''
Setup
'''

# Load popsummary files
popsummary_results = {
    'variance_cut' : PopulationResult(fname="../data_release/BBHSpin_GaussianChiEffChiP.h5"),
    'Neff_cut' : PopulationResult(fname="../data_release/BBHSpin_GaussianChiEffChiP_NeffCut.h5")
}   
    
# Load in gwtc-3 data for comparison 
with open(pf.gwtc3_folder+'GaussianSpin/gaussian-spin-xeff-xp-ppd-data.json', 'r') as f:
    gwtc3_data = json.load(f)
    gwtc3_hyperposterior = gwtc3_data['pop_samples']
    gwtc3_hyperposterior['cov'] = gwtc3_hyperposterior['rho']
    gwtc3_hyperposterior.pop('rho')

# Runs
run_labels = {
    'GWTC-3':r'GWTC-3, $N_{\rm eff}$ cut',
    'Neff_cut':r'GWTC-4, $N_{\rm eff}$ cut',  
    'variance_cut':r'GWTC-4, ${\sigma^2}_{\ln \hat{\cal L}}$ cut'
}

# Colors for the different runs
colors_dict = {
    'GWTC-3':pf.color_gwtc3,
    'Neff_cut':pf.color_gaussian_eff_2,
    'variance_cut':pf.color_gaussian_eff
}


'''
Cornerplot
'''

# Get hyper-parameter names and labels
hyper_params = popsummary_results['variance_cut'].get_metadata(field = 'hyperparameters')[:-2]
hyper_params_labels = popsummary_results['variance_cut'].get_metadata(field = 'hyperparameter_latex_labels')[:-2]

# Plotting limits
limits_dict = dict(mu_eff=[-0.05,0.15], sig_eff=[0.05,0.2], mu_p=[0.05,0.5], sig_p=[0.05,0.5], cov=[-0.75, 0.75])

# Put everything in one dict
data_dict = {k:{} for k in popsummary_results.keys()}
data_dict['GWTC-3'] = {} 
for param,lbl in zip(hyper_params, hyper_params_labels): 
    for k in ['GWTC-3', 'Neff_cut', 'variance_cut']: 
        if k=='GWTC-3': 
            samples = np.asarray(gwtc3_hyperposterior[param])
        else: 
            samples = popsummary_results[k].get_hyperparameter_samples(hyperparameters=[param])
        data_dict[k][param] = samples.flatten()

# Make plot
fig, axes = plt.subplots(5,5, figsize=(9,9))

for run in ['GWTC-3', 'Neff_cut', 'variance_cut']: 

    run_label = run_labels[run]
    data = data_dict[run]
    if run=='GWTC-3': 
        hist_kws = dict(density=True, color=colors_dict[run], histtype='step', lw=1.5)
    else:
        hist_kws = dict(density=True, color=colors_dict[run], alpha=0.8)
        
    pf.cornerplot(axes, data, hyper_params, run_label, limits_dict, hist_kws, nbins=20)
    
pf.cornerplot_axes_setup(axes, hyper_params_labels)
axes[0][0].legend(bbox_to_anchor=(2.5,1), frameon=False)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(pf.figs_folder+'figure_18.pdf', bbox_inches='tight')
plt.close()


'''
PDFs
'''

scaling = 1.2
figsize = (scaling*pf.FIG_WIDTH, scaling*0.7*pf.FIG_WIDTH)

fig, axes = plt.subplots(2,2,figsize=figsize)

### 1D dists

handles = []

for run in ['Neff_cut', 'variance_cut']: 

    run_label = run_labels[run]
    color = colors_dict[run]
    
    chi_eff_grid, chi_eff_rates = popsummary_results[run].get_rates_on_grids('chi_eff')
    chi_p_grid, chi_p_rates = popsummary_results[run].get_rates_on_grids('chi_p')
    kws = dict(color=color, fill_between=True)

    pf.plot_traces(axes[0][0], chi_eff_grid, chi_eff_rates, **kws)
    pf.plot_traces(axes[1][1], chi_p_grid, chi_p_rates, **kws)

    # for legend
    handles.append(Line2D([], [], color=color, label=run_label))

# format axes
axes[0][0].set_xlabel(r'$\chi_\mathrm{eff}$')
axes[0][0].set_ylabel(r'$p(\chi_\mathrm{eff})$')
axes[0][0].set_ylim(0, )
axes[0][0].set_xlim(-0.5, 0.5)
axes[0][0].grid(color='silver', ls=':')
axes[0][0].legend(handles=handles, bbox_to_anchor=(2,1), frameon=False)

axes[1][1].set_xlabel(r'$\chi_\mathrm{p}$')
axes[1][1].set_ylabel(r'$p(\chi_\mathrm{p})$')
axes[1][1].set_ylim(0, )
axes[1][1].set_xlim(0,1)
axes[1][1].grid(color='silver', ls=':')

axes[0][1].set_visible(False)

### 2D dists

for run in ['variance_cut', 'Neff_cut']: 

    joint_pos, joint_ppd = popsummary_results[run].get_rates_on_grids('joint')
    
    cmap = pf.make_colormap(colors_dict[run])
    
    axes[1][0].imshow(
        joint_ppd, 
        extent=[min(joint_pos[0]), max(joint_pos[0]), min(joint_pos[1]), max(joint_pos[1])], 
        aspect='auto', 
        cmap=cmap, alpha=0.3
    )

    # Compute and plot the 90% and 50% credible region
    color = pf.darken_color(colors_dict[run], factor=0.8)
    levels = pf.compute_contour_levels(joint_ppd, levels=[0.9, 0.5])
    X = np.linspace(min(joint_pos[0]), max(joint_pos[0]), joint_ppd.shape[1])  # Match imshow extent
    Y = np.linspace(min(joint_pos[1]), max(joint_pos[1]), joint_ppd.shape[0])  
    axes[1][0].contour(X, Y, np.flip(np.flip(joint_ppd), axis=1), levels=levels, colors=color, linewidths=2)

axes[1][0].set_xlabel(r'$\chi_\mathrm{eff}$')
axes[1][0].set_xlim(-0.5, 0.5)
axes[1][0].set_ylabel(r'$\chi_\mathrm{p}$')
axes[1][0].set_ylim(0, 1)
axes[1][0].grid(color='silver', ls=':')

plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.savefig(pf.figs_folder+'figure_19.pdf', bbox_inches='tight')
plt.close()