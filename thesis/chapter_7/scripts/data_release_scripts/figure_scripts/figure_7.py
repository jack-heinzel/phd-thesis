import numpy as np
from popsummary.popresult import PopulationResult
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plot_funcs_bbh_spin as pf
import h5py

pf.setup()

'''
Setup
'''

# load the bspline data
result_bspline = PopulationResult(fname=pf.DATADIR + "BBHMassSpinRedshift_BSplineIID.h5")

# load the default data
result_default = PopulationResult(fname=pf.DATADIR + "BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5")

# rates on grids
chi_grid_bspline, chi_rates_bspline   = result_bspline.get_rates_on_grids('p(a)')
cost_grid_bspline, cost_rates_bspline = result_bspline.get_rates_on_grids('p(cos_tilt)')
chi_grid_default, chi_rates_default   = result_default.get_rates_on_grids('a')
cost_grid_default, cost_rates_default = result_default.get_rates_on_grids('cos_tilt')

# load gwtc-3 data 
with h5py.File(pf.gwtc3_folder+\
               'PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', 'r') as f:
    lines_chi = f['lines']['a_1'][:]
    
with h5py.File(pf.gwtc3_folder+\
               'PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', 'r') as f:
    lines_cost = f['lines']['cos_tilt_1'][:]
    
chi_grid = np.linspace(0,1,1000)
cost_grid = np.linspace(-1,1,1000)

# more plotting macros
figsize = (1.3*pf.FIG_WIDTH, 0.8*pf.FIG_WIDTH/2)

# legend 
handles = [
    Line2D([], [], color=pf.color_gwtc3, ls='--', label='Default, GWTC-3.0'),
    Line2D([], [], color=pf.color_default, label='Gaussian Component\nSpins, GWTC-4.0'),
    Line2D([], [], color=pf.color_bspline, label='B-Spline, GWTC-4.0'), 
]

'''
Make Figure
'''

fig, axes = plt.subplots(1,2,figsize=figsize)

## spin magnitudes

pf.plot_traces(axes[0], chi_grid, lines_chi, color=pf.color_gwtc3, normed=False, just_quants=True)
pf.plot_traces(axes[0], chi_grid_bspline, chi_rates_bspline, color=pf.color_bspline, normed=False, fill_between=True)
pf.plot_traces(axes[0], chi_grid_default, chi_rates_default, color=pf.color_default, normed=False, fill_between=True)

axes[0].set_xlabel(r'$\chi$')
axes[0].set_ylabel(r'$p(\chi)$')
axes[0].set_ylim(0, )
axes[0].set_xlim(0, 1)

axes[0].grid(color='silver', alpha=0.5, ls=':')
axes[0].legend(handles=handles, loc='upper right')

## spin tilts

pf.plot_traces(axes[1], cost_grid, lines_cost, color=pf.color_gwtc3, normed=False, just_quants=True)
pf.plot_traces(axes[1], cost_grid_bspline, cost_rates_bspline, color=pf.color_bspline, normed=False, fill_between=True)
pf.plot_traces(axes[1], cost_grid_default, cost_rates_default, color=pf.color_default, normed=False, fill_between=True)

axes[1].set_xlabel(r'$\cos\theta$')
axes[1].set_ylabel(r'$p(\cos\theta)$')
axes[1].set_ylim(0, )
axes[1].set_xlim(-1, 1)

axes[1].grid(color='silver', alpha=0.5, ls=':')

plt.subplots_adjust(wspace=0.3)

# save figure 
plt.savefig(pf.figs_folder+'figure_7.pdf', bbox_inches='tight')

plt.close()
