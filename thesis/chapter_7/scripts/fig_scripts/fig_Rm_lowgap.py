import numpy as np
import popsummary
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

fig, axes = plt.subplots(ncols=2, figsize=(10,4))

pdb_file = '/home/jack.heinzel/public_html/o4a_population_paper/plotting/PDB-in-O4/analysis_results/multi_pdb_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_popsummary.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)

bgp_file = '/home/anarya.ray/public_html/gppop_m1m2_popsummary_ifar4.hdf5'
bgp_result = popsummary.popresult.PopulationResult(fname=bgp_file)

mass_key = ['primary_mass', 'secondary_mass']

for ii in range(1,3):
    ax = axes[ii-1]
    pdb_m, pdb_Rm = pdb_result.get_rates_on_grids(mass_key[ii-1])
    bgp_m, bgp_Rm = bgp_result.get_rates_on_grids(mass_key[ii-1])
    ax.fill_between(pdb_m[0], np.percentile(pdb_Rm, 5, axis=0), np.percentile(pdb_Rm, 95, axis=0), color='C0', alpha=0.3)
    ax.fill_between(bgp_m[0], np.percentile(bgp_Rm, 5, axis=0), np.percentile(bgp_Rm, 95, axis=0), color='C1', alpha=0.3)

    ax.plot(pdb_m[0], np.mean(pdb_Rm, axis=0), color='C0', label='PDB')
    ax.plot(bgp_m[0], np.mean(bgp_Rm, axis=0), color='C1', label='BGP')

    ax.set_yscale('log')
    ax.set_xlim(1, 10)
    ax.set_ylim(1e-2, 1e3)
    ax.set_xticks(np.arange(1,11))
    ax.legend(frameon=False, loc='upper right')

    ax.set_ylabel(f'$\\textrm{{d}}\mathcal{{R}}/\\textrm{{d}}m_{ii}$ [Gpc${{}}^{{-3}}$yr${{}}^{{-1}}M_\odot^{{-1}}$]')
    ax.set_xlabel(f'$m_{ii}$ [$M_\odot$]')

plt.tight_layout()
plt.savefig('../../figures/full_mass_spectrum_lowgap.pdf')