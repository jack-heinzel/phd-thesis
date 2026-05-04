import numpy as np
import popsummary
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

pdb_file = '/home/jack.heinzel/public_html/o4a_population_paper/plotting/PDB-in-O4/analysis_results/multi_pdb_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_popsummary.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)

pdb_m1, pdb_Rm1 = pdb_result.get_rates_on_grids('primary_mass')
pdb_m2, pdb_Rm2 = pdb_result.get_rates_on_grids('secondary_mass') # same m grid
pdb_Rm = pdb_Rm1 + pdb_Rm2

bgp_file = '/home/anarya.ray/public_html/gppop_m1m2_popsummary_ifar4.hdf5'
bgp_result = popsummary.popresult.PopulationResult(fname=bgp_file)
bgp_m1, bgp_Rm1 = bgp_result.get_rates_on_grids('primary_mass')
bgp_m2, bgp_Rm2 = bgp_result.get_rates_on_grids('secondary_mass') # same m grid
bgp_Rm = bgp_Rm1 + bgp_Rm2

mlows = np.array([1, 2.5, 8])
mhis = np.array([2.5, 5, 10])

pdb_Rs = np.array([np.trapz(pdb_Rm[:,np.digitize(mlows[ii], pdb_m1[0]): np.digitize(mhis[ii], pdb_m1[0])-1], x=pdb_m1[0,np.digitize(mlows[ii], pdb_m1[0]): np.digitize(mhis[ii], pdb_m1[0])-1]) / (pdb_m1[0,np.digitize(mhis[ii], pdb_m1[0])-1] - pdb_m1[0,np.digitize(mlows[ii], pdb_m1[0])]) for ii in range(len(mlows))]).T
bgp_Rs = np.array([np.trapz(bgp_Rm[:,np.digitize(mlows[ii], bgp_m1[0]): np.digitize(mhis[ii], bgp_m1[0])-1], x=bgp_m1[0,np.digitize(mlows[ii], bgp_m1[0]): np.digitize(mhis[ii], bgp_m1[0])-1]) / (bgp_m1[0,np.digitize(mhis[ii], bgp_m1[0])-1] - bgp_m1[0,np.digitize(mlows[ii], bgp_m1[0])]) for ii in range(len(mlows))]).T

from corner import corner
labels = [f'$\langle \mathcal{{R}}\\rangle_{{m\in[{mlows[ii]},{mhis[ii]}]M_\odot}}$' for ii in range(len(mlows))]

fig = corner(np.log10(pdb_Rs), color='b', bins=30, smooth=1.5, smooth1d=None, plot_datapoints=False, plot_density=False, fill_contours=True, levels=(1-np.exp(-np.arange(1,3)**2 / 2)), axes_scale='linear', hist_kwargs={'density': True})
fig = corner(np.log10(bgp_Rs), color='r', bins=30, smooth=1.5, smooth1d=None, plot_datapoints=False, plot_density=False, fill_contours=True, levels=(1-np.exp(-np.arange(1,3)**2 / 2)), labels=labels, fig=fig, axes_scale='linear', hist_kwargs={'density': True})

axes = fig.get_axes()
axes[3].plot(np.linspace(-3,3,100), np.linspace(-3,3,100), 'k--')
axes[6].plot(np.linspace(-3,3,100), np.linspace(-3,3,100), 'k--')
axes[7].plot(np.linspace(-3,3,100), np.linspace(-3,3,100), 'k--')

axes[3].fill_betweenx(np.linspace(-3,3,100), np.full(100, -3), np.linspace(-3,3,100), color='k', alpha=0.3)
axes[6].fill_betweenx(np.linspace(-3,3,100), np.full(100, -3), np.linspace(-3,3,100), color='k', alpha=0.3)
axes[7].fill_betweenx(np.linspace(-3,3,100), np.full(100, -3), np.linspace(-3,3,100), color='k', alpha=0.3)

from matplotlib.lines import Line2D 
from matplotlib.patches import Patch
z = np.ones(3)
axes[2].legend([Line2D(z,z,color='r'), Line2D(z,z,color='b'), Patch(linewidth=1, linestyle='dashed', color='k', alpha=0.3)], ['BGP', 'PDB', '$R_x <R_y$'])

log_ticks = np.log10(np.concatenate([
    np.linspace(1e-3,9e-3,10),
    np.linspace(1e-2,9e-2,10),
    np.linspace(1e-1,9e-1,10),
    np.linspace(1e0,9e0,10),
    np.linspace(1e1,9e1,10),
    np.linspace(1e2,9e2,10)
]))
major_ticks = np.arange(-3,4)
names = [f'$10^{{{x}}}$' for x in major_ticks]
blank_names = ['' for x in major_ticks]
# set x axes correctly
for a in [0,3,4,6,7,8]:
    xlim = axes[a].get_xlim()
    ylim = axes[a].get_ylim()
    axes[a].set_xticks(log_ticks, minor=True)
    if a in [0,3,4]:
        axes[a].set_xticks(major_ticks, blank_names)
    else:
        axes[a].set_xticks(major_ticks, names)
    axes[a].set_xlim(xlim)
    axes[a].set_ylim(ylim)
    

# set y axes correctly
for a in [3,6,7]:
    xlim = axes[a].get_xlim()
    ylim = axes[a].get_ylim()
    axes[a].set_yticks(log_ticks, minor=True)
    if a in [7]:
        axes[a].set_yticks(major_ticks, blank_names)
    else:
        axes[a].set_yticks(major_ticks, names)
    axes[a].set_xlim(xlim)
    axes[a].set_ylim(ylim)
    
fig.savefig('../../figures/pdb_lowgap_rates.pdf')