import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import popsummary
from scipy.interpolate import LinearNDInterpolator
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

pdb_file = '/home/amanda.farah/projects/O4/PDB-in-O4/analysis_results/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)

h = pdb_result.get_metadata('hyperparameters')
dipnames = ['A', 'NSmax', 'BHmin']
fancynames = ['$A$', '$m_{\\rm low}$ [$M_\odot$]', '$m_{\\rm high}$ [$M_\odot$]']
dip_pars = pdb_result.get_hyperparameter_samples(hyperparameters=dipnames)
from corner import corner
print(dip_pars.shape) # we will want more posterior samples...
fig = corner(
    dip_pars, 
    labels=fancynames, 
    bins=30,
    smooth=1,
    plot_density=False, 
    plot_contours=True, 
    fill_contours=True,
    no_fill_contours=False, 
    plot_datapoints=False,
    levels=1-np.exp(-np.arange(1,4)**2 / 2),
    color='cornflowerblue',
)
fig.savefig('../../figures/pdb_lowgap_corner.pdf')