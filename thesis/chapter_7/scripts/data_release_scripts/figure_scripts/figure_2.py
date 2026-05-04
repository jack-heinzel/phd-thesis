import numpy as np
import popsummary
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

fig, axes = plt.subplots(nrows=2, figsize=(10,8))

DATADIR = "../data_release/"
pdb_file = DATADIR + 'AllCBC_FullPop.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)

bgp_file = DATADIR + 'AllCBC_FullPopBGP.h5'
bgp_result = popsummary.popresult.PopulationResult(fname=bgp_file)

mass_key = ['primary_mass', 'secondary_mass']
color2 = '#FE6100'
color1 = '#648FFF'
for ii in range(1,3):
    ax = axes[ii-1]
    pdb_m, pdb_Rm = pdb_result.get_rates_on_grids(mass_key[ii-1])
    bgp_m, bgp_Rm = bgp_result.get_rates_on_grids(mass_key[ii-1])
    ax.fill_between(pdb_m[0], np.percentile(pdb_Rm, 5, axis=0), np.percentile(pdb_Rm, 95, axis=0), color=color1, alpha=0.3, rasterized=True)
    ax.fill_between(bgp_m[0], np.percentile(bgp_Rm, 5, axis=0), np.percentile(bgp_Rm, 95, axis=0), color=color2, alpha=0.3, rasterized=True)

    ax.plot(pdb_m[0], np.mean(pdb_Rm, axis=0), color=color1, label=r'\textsc{FullPop}-4.0')
    ax.plot(bgp_m[0], np.mean(bgp_Rm, axis=0), color=color2, label='BGP')

    ax.set_yscale('log')
    
    ax.set_ylim(1e-2, 2e3)
    ax.set_xlim(1,15)

    ax.set_ylabel(f'$\\textrm{{d}}\mathcal{{R}}/\\textrm{{d}}m_{ii}$ [Gpc${{}}^{{-3}}$ yr${{}}^{{-1}}M_\odot^{{-1}}$]')
    ax.set_xlabel(f'$m_{ii}$ [$M_\odot$]')
    ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')

axes[0].legend(frameon=True, loc='upper right')
inset = axes[1].inset_axes([0.78, 0.56, 0.2, 0.4])

a = pdb_result.get_hyperparameter_samples(hyperparameters=['A'])
inset.hist(a, bins=30, density=True, color=color1)
inset.set_xlabel('$A$ (gap depth)')
inset.set_ylabel('$p(A)$')
inset.set_xlim(0,1)
inset.set_yticks([])
inset.grid(color='silver', alpha=0.5, ls=':')

plt.tight_layout()
plt.savefig('../figures/figure_2.pdf')
