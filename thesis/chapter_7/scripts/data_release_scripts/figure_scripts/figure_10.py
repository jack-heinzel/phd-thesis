import numpy as np
import popsummary
from matplotlib import pyplot as plt
import bilby
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

powerlaw_file = '../data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5'
# plot parametric
result = popsummary.popresult.PopulationResult(fname=powerlaw_file)
# print(result.get_metadata('hyperparameters'))

maxz = 1.5

color3 = '#FE6100'
color2 = '#289e49'
color1 = '#648FFF'

z_list, R_z = result.get_rates_on_grids('redshift')
z_list = z_list[0]
samples = result.get_hyperparameter_samples(hyperparameters=['lamb', 'rate'])
kappa, R_0 = samples[:,0], samples[:,1]

gwtc3_result = bilby.core.result.read_in_result('../gwtc3_data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
gwtc3_posterior = gwtc3_result.posterior
kappa_gwtc3 = np.array(gwtc3_posterior['lamb'])
R_0_gwtc3 = np.array(gwtc3_posterior['rate'])

R_z_gwtc3 = R_0_gwtc3[:,None] * (1+z_list)**kappa_gwtc3[:,None]
fig, ax = plt.subplots(nrows=2, figsize=(6,8))

ax[0].hist(kappa_gwtc3, bins=np.linspace(-2,6,51), density=True, histtype='step', linestyle='dashed', color=color1, label='GWTC-3.0')
ax[0].hist(kappa, bins=np.linspace(-2,6,51), density=True, color=color1, label='GWTC-4.0')

ax[0].set_xlabel('$\kappa$')
ax[0].set_ylabel('$p(\kappa)$')
ax[0].legend(frameon=False, loc='upper left')

madau_dickinson = (1 + z_list)**2.7 / (1 + ((1+z_list)/2.9)**5.6)
madau_dickinson /= madau_dickinson[0]

norm = np.percentile(R_z, 50, axis=0)[0]
ax[1].plot(z_list, madau_dickinson*norm, color='k', label='Star Formation (Arbitrary Norm)')
ax[1].plot(z_list, np.percentile(R_z_gwtc3, 5, axis=0), linestyle='dashed', color=color1, lw=0.8)
ax[1].plot(z_list, np.percentile(R_z_gwtc3, 95, axis=0), linestyle='dashed', color=color1, lw=0.8, label='\\textsc{Power Law Redshift}, GWTC-3.0')
ax[1].fill_between(z_list, np.percentile(R_z, 5, axis=0), np.percentile(R_z, 95, axis=0), color=color1, alpha=0.3)
ax[1].plot(z_list, np.percentile(R_z, 50, axis=0), color=color1, label='\\textsc{Power Law Redshift}, GWTC-4.0')


ax[1].set_xlabel('$z$')
ax[1].set_ylabel('$\\mathcal{R}(z)$ [Gpc${}^{-3}$ yr${}^{-1}$]')

bs_file = '../data_release/BBHMassSpinRedshift_BSplineIID.h5'
# plot parametric
pl_result = popsummary.popresult.PopulationResult(fname=powerlaw_file)
bs_result = popsummary.popresult.PopulationResult(fname=bs_file)
z_0 = 0.

try:
    pl_z_list, pl_R_z = pl_result.get_rates_on_grids('redshift')
    pl_z_list = pl_z_list[0]

except KeyError:
    samples = pl_result.get_hyperparameter_samples(hyperparameters=['lamb', 'rate'])
    kappa, R_0 = samples[:,0], samples[:,1]
    
    pl_z_list = np.linspace(0, maxz, 1000)
    pl_R_z = R_0[:,None] * ((1+pl_z_list)/(1.0 + z_0))**kappa[:,None]

bs_z_list, bs_R_z = bs_result.get_rates_on_grids('rate_vs_redshift')

bs_z_list = bs_z_list[0]

ax[1].plot(bs_z_list, np.percentile(bs_R_z, 50, axis=0), color=color3, linestyle='solid', label='\\textsc{B-Spline}, GWTC-4.0', zorder=1)
ax[1].fill_between(bs_z_list, np.percentile(bs_R_z, 5, axis=0), np.percentile(bs_R_z, 95, axis=0), color=color3, alpha=0.3, zorder=2)#, label='BS')

ax[1].set_yscale('log')
ax[1].set_ylim([8, 3e3])
ax[1].set_xlim([0,maxz])

ax[1].set_xlabel('$z$')
ax[1].set_ylabel('$\\mathcal{R}(z)$ [Gpc${}^{-3}$ yr${}^{-1}$]')

ax[1].grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
ax[1].legend(frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig('../figures/figure_10.pdf')