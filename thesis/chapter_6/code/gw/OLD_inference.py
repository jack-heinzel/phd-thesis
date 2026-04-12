# script for running hyper PE with huge MC integrals
import numpy as np
import os
import gwpopulation
gwpopulation.set_backend('jax')
xp = gwpopulation.utils.xp
print(f'xp.__name__ = {xp.__name__}')
import bilby
import pandas as pd
import pickle as pkl
import sys
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import optparse
import glob
import astropy
Planck15 = astropy.cosmology.Planck15
z_at_value = astropy.cosmology.z_at_value
import h5py
import json

import optparse

class DagNabbitCustomLikelihood(gwpopulation.hyperpe.HyperparameterLikelihood):
    def __init__(
        self, 
        posteriors, 
        hyper_prior, 
        ln_evidences=None, 
        max_samples=1e100, 
        selection_function=lambda args: 1, 
        conversion_function=lambda args: (args, None), 
        cupy=False, 
        maximum_uncertainty=xp.inf, 
        posterior_length_corrections=None,
    ):
        super(DagNabbitCustomLikelihood, self).__init__(posteriors, hyper_prior, ln_evidences=ln_evidences, max_samples=max_samples, selection_function=selection_function, conversion_function=conversion_function, cupy=cupy, maximum_uncertainty=maximum_uncertainty)
        if posterior_length_corrections is None:
            self.posterior_length_corrections = xp.ones(self.n_posteriors)
        else:
            self.posterior_length_corrections = xp.array(posterior_length_corrections)
        print(f'Using posterior length corrections of mean {xp.mean(self.posterior_length_corrections)} +/- {xp.std(self.posterior_length_corrections)}, minimum = {xp.min(self.posterior_length_corrections)}')
        # self.posterior_length_corrections should be fraction of length of posterior to number of samples
    
    def _compute_per_event_ln_bayes_factors(self, return_uncertainty=True):
        weights = self.hyper_prior.prob(self.data) / self.sampling_prior
        expectation = xp.mean(weights, axis=-1)
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            variance /= self.posterior_length_corrections
            return xp.log(expectation), variance
        else:
            return xp.log(expectation)

class NoSecondarySmoothingMass(gwpopulation.models.mass.SinglePeakSmoothedMassDistribution):
    def p_q(self, dataset, beta, mmin, delta_m):
        # include delta_m because the parent class wants to give it, but don't use it
        p_q = gwpopulation.utils.powerlaw(dataset["mass_ratio"], beta, 1, 1e-3)
        return xp.nan_to_num(p_q)
    
def iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, mu_spin, sigma_spin):
    prior = (1 - xi_spin) / 4 + xi_spin * gwpopulation.utils.truncnorm(
        dataset["cos_tilt_1"], mu_spin, sigma_spin, 1, -1
    ) * gwpopulation.utils.truncnorm(dataset["cos_tilt_2"], mu_spin, sigma_spin, 1, -1)
    return prior

def g_q(q, n):
    return xp.exp(xp.abs(q - 0.1)**n - 0.9**n)

def salvo_spin_orientation_gaussian_isotropic(dataset, f_spin, n_spin, mu_spin, sigma_spin):
    q = dataset['mass_ratio']
    g = g_q(q, n_spin)

    xi_spin = f_spin * (g - g_q(0.1, n_spin)) / (g_q(1, n_spin) - g_q(0.1, n_spin))
    prior = (1 - xi_spin) / 4 + xi_spin * gwpopulation.utils.truncnorm(
        dataset["cos_tilt_1"], mu_spin, sigma_spin, 1, -1
    ) * gwpopulation.utils.truncnorm(dataset["cos_tilt_2"], mu_spin, sigma_spin, 1, -1)
    return prior
    
def salvos_spin_model(dataset, f_spin, n_spin, mu_spin, sigma_spin, amax, alpha_chi, beta_chi):

    prior = salvo_spin_orientation_gaussian_isotropic(
        dataset, f_spin, n_spin, mu_spin, sigma_spin
    ) * gwpopulation.models.spin.iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi)
    return prior

def mu_variable_spin_model(dataset, xi_spin, mu_spin, sigma_spin, amax, alpha_chi, beta_chi):

    prior = iid_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, mu_spin, sigma_spin
    ) * gwpopulation.models.spin.iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi)
    return prior

def ParseCommandLine():
    parser = optparse.OptionParser()

    parser.add_option('--NPE', type=int, default=16_000, help='size of PE MC integrals')
    parser.add_option('--NVT', type=int, default=10_000, help='size in millions of VT MC integral')
    parser.add_option('--Nevents', type=int, default=400, help='number of events to include in the inference')
    parser.add_option('--homedir', default='../../', help='relative path to the mc-bias home directory')
    parser.add_option('--datadir', default='data/gw/pe-and-vt', help='path from home to data')
    parser.add_option('--resultdir', default='data/gw/results', help='path from home to result')
    parser.add_option('--nlive', type=int, default=1000)
    parser.add_option('--gpu', default='0')
    parser.add_option('--dagnabbit', action='store_true', default=False) #TODO 
    parser.add_option('--UncertaintyCut', action='store_true', default=False)
    parser.add_option('--delta_function_test', action='store_true', default=False)
    parser.add_option('--use_old_events', action='store_true', default=False)
    parser.add_option('--use_small_events', action='store_true', default=False)
    parser.add_option('--no_RB_events', action='store_true', default=False)
    parser.add_option('--defaultSampleEvents', action='store_true', default=False)
    parser.add_option('--remove_207', action='store_true', default=False)
    parser.add_option('--remove_result', action='store_true', default=False)
    parser.add_option('--salvos_PE', action='store_false', default=True)
    parser.add_option('--PreallocateFrac', default='0.4')

    parser.add_option('--pe_seed', type=int, default=0)
    parser.add_option('--vt_seed', type=int, default=0)
    parser.add_option('--turn_off_secondary_smoothing', action='store_true', default=False, 
                      help='Smoothing the q distribution causes a sharp peak at q=1 for low primary mass binaries. This causes ridiculously poor reweighting efficiency for these events. Turning it off can help the likelihood variance dramatically.')
    parser.add_option('--print_single_event_variances', action='store_true', default=False)

    parser.add_option('--keep_bad_events', action='store_true', default=False, help='Whether to keep the 31 wrong events when m1 is a UNsmoothed powerlaw')
    
    opts, args = parser.parse_args()

    return opts

true_hyperpar = {
    'alpha': 3., 
    'beta': 1.,
    'mmax': 85.,
    'mmin': 6.,
    'lam': 0.05,
    'mpp': 35.,
    'sigpp': 4.,
    'delta_m': 5.,
    'lamb': 2.,
    'amax': 1.0,
    'mu_chi': 0.27272727272, # mean of the beta distribution for spins
    'sigma_chi': 0.0305149396, # var of the beta distribution for spins
    'xi_spin': 0.5,
    'mu_spin': 0.5,
    'sigma_spin': 0.5,
    }


opts = ParseCommandLine()
if opts.salvos_PE:
    true_hyperpar = {
        'alpha': 3.4, 
        'beta': 1.1,
        'mmax': 87.,
        'mmin': 5.,
        'lam': 0.04,
        'mpp': 34.,
        'sigpp': 3.6,
        'delta_m': 4.8,
        'lamb': 2.73,
        'amax': 1.0,
        'mu_chi': 0.2737704918, # mean of the beta distribution for spins
        'sigma_chi': 0.0280028464255, # var of the beta distribution for spins
        'f_spin': 1,
        'n_spin': 2,
        'mu_spin': 1,
        'sigma_spin': 1.15,
        }

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = opts.PreallocateFrac

datadir = os.path.join(opts.homedir, opts.datadir)
resultdir = os.path.join(opts.homedir, opts.resultdir)

run_label = f'nevents_{opts.Nevents}_nvt_{opts.NVT}M_npe_{opts.NPE}'
if opts.UncertaintyCut:
    run_label = 'varcut1_' + run_label
if opts.turn_off_secondary_smoothing:
    run_label = 'qpowerlaw_' + run_label
if opts.remove_207:
    run_label = 'bad_pe_gone_' + run_label 
if opts.dagnabbit:
    run_label = 'dagnabbit_' + run_label
if opts.use_old_events:
    run_label = 'old_events_' + run_label
if opts.use_small_events:
    run_label = 'small_events_' + run_label
if opts.no_RB_events:
    run_label = 'no_RB_' + run_label
if opts.defaultSampleEvents:
    run_label = 'bilbySample_' + run_label
if opts.keep_bad_events:
    run_label = 'keep_bad_events_' + run_label
if opts.delta_function_test:
    run_label = 'TEST_DELTA_FN_POSTERIORS_' + run_label 

if opts.salvos_PE:
    run_label = 'salvo_pe/' + run_label
else:
    print("WARNING, USING OLD PE")
    run_label = 'old_results_not_salvo_pe/' + run_label

sub_run_label = f'peseed_{opts.pe_seed}_vtseed_{opts.vt_seed}'
run_directory = os.path.join(resultdir, run_label, sub_run_label)
os.makedirs(run_directory, exist_ok=True)

priors = { # should be GWTC 3 priors
    'alpha': bilby.prior.Uniform(0, 6,name='alpha', latex_label='$\\alpha$'),
    'beta': bilby.prior.Uniform(-2, 10,name='beta', latex_label='$\\beta_q$'),
    'mmax': bilby.prior.Uniform(80, 100,name='mmax', latex_label='$m_{\\rm max}$'),
    'mmin': bilby.prior.Uniform(2, 7,name='mmin', latex_label='$m_{\\rm min}$'),
    'lam': bilby.prior.Uniform(0., 1.,name='lam', latex_label='$\\lambda_m$'),
    'mpp': bilby.prior.Uniform(20, 50,name='mpp', latex_label='$\\mu_m$'),
    'sigpp': bilby.prior.Uniform(1, 10,name='sigpp', latex_label='$\\sigma_m$'),
    'delta_m': bilby.prior.Uniform(0, 10,name='delta_m', latex_label='$\\delta_m$'),
    'lamb': bilby.prior.Uniform(-4, 8,name='lamb', latex_label='$\\lambda_z$'),
    'amax': 1.0, # begin spin
    'f_spin': bilby.prior.Uniform(0, 1,name='f_spin', latex_label='$f_{\\rm spin}$'), 
    'n_spin': bilby.prior.Uniform(-5, 5,name='n_spin', latex_label='$n_{\\rm spin}$'), 
    'mu_spin': bilby.prior.Uniform(-1, 1,name='mu_spin', latex_label='$\\mu_{\\rm spin}$'),
    'sigma_spin': bilby.prior.Uniform(0.1, 4,name='sigma_spin', latex_label='$\\sigma_{\\rm spin}$'), 
    'mu_chi': bilby.prior.Uniform(0, 1,name='mu_chi', latex_label='$\\mu_\\chi$'),
    'sigma_chi': bilby.prior.Uniform(0.005, 0.25,name='sigma_chi', latex_label='$\\sigma_{\\chi}^2$'),
}

maxlike = true_hyperpar.copy()

if opts.turn_off_secondary_smoothing:
    mass_model = NoSecondarySmoothingMass()
    vt_mass_model = NoSecondarySmoothingMass()
else:
    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    vt_mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)
vt_redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)

spin_model = salvos_spin_model
vt_spin_model = salvos_spin_model

model = gwpopulation.experimental.jax.NonCachingModel([mass_model, spin_model, redshift_model])
vt_model = gwpopulation.experimental.jax.NonCachingModel([vt_mass_model, vt_spin_model, vt_redshift_model])

# load VT of correct size
gwpop_data = {
    'total_generated': 0, 'analysis_time': 0, 'mass_ratio': np.array([]), 'redshift': np.array([]), 
    'a_1': np.array([]), 'a_2': np.array([]), 'mass_1': np.array([]), 'cos_tilt_1': np.array([]), 
    'cos_tilt_2': np.array([]), 'prior': np.array([]), 'snr': np.array([])
    }

lower_read_number = int(np.floor(opts.NVT*opts.vt_seed / 1000))
upper_read_number = int(np.ceil(opts.NVT*(opts.vt_seed + 1) / 1000))
for ii in range(lower_read_number, upper_read_number):
    print(f'loading {ii} set of 1e9 injections')
    with open(os.path.join(datadir, f'copy_{ii}_o4_custom_vt_1e9.pkl'), 'rb') as ff:
        ii_gwpop_data = pkl.load(ff)
    for key in gwpop_data.keys():
        lower_add = np.maximum(opts.vt_seed * opts.NVT / 1000 - ii, 0)
        upper_add = np.minimum((opts.vt_seed + 1) * opts.NVT / 1000 - ii, 1)
        if isinstance(ii_gwpop_data[key], np.ndarray) or isinstance(ii_gwpop_data[key], pd.core.series.Series):
            if isinstance(ii_gwpop_data[key], np.ndarray):
                if ii_gwpop_data[key].ndim == 0:
                    continue  #accidentally 
            # print(key, ii_gwpop_data[key])
            lower = int(len(ii_gwpop_data[key])*lower_add)
            upper = int(len(ii_gwpop_data[key])*upper_add)
            gwpop_data[key] = np.concatenate(
                (gwpop_data[key], 
                 np.array(ii_gwpop_data[key])[lower:upper])
                )
        elif key == 'total_generated':
            gwpop_data[key] = gwpop_data[key] + int(ii_gwpop_data[key]*(upper_add - lower_add))
        elif key == 'analysis_time':
            gwpop_data[key] = ii_gwpop_data[key]

for key in gwpop_data.keys():
    if key in ['total_generated', 'analysis_time']:
        continue
    elif key in ['mass_ratio', 'redshift', 'a_1', 'a_2', 'mass_1', 'cos_tilt_1', 'cos_tilt_2', 'prior']:
        if opts.salvos_PE:
            keep = xp.array(gwpop_data['snr']) > 11.
            gwpop_data[key] = xp.array(gwpop_data[key])[keep]
        else:
            gwpop_data[key] = xp.array(gwpop_data[key])
if opts.salvos_PE:
    print(f"Downselecting {int(np.mean(keep)*1000)/10}% of samples from snr > 9 to snr > 11")

print(f"Using {len(gwpop_data['mass_1'])} found events from {gwpop_data['total_generated']}")
selection_effects = gwpopulation.vt.ResamplingVT(vt_model, gwpop_data, n_events=opts.Nevents)#, unbias_factor=opts.UnbiasFactor)

# maxlike = bilby.core.prior.PriorDict(priors).sample()
# maxlike['mmin'] = 2.1
# print(maxlike)

# converted, _ = gwpopulation.conversions.convert_to_beta_parameters(maxlike)
# m, v = selection_effects(converted.copy())
# print(m, v, opts.Nevents**2 * v / m**2 ) 
# x = selection_effects.model.prob(selection_effects.data)
# print(np.all(~np.isnan(x)))
# print(x)
# bad = np.arange(len(x))[np.isnan(x)][:10]
# for b in bad:
#     print({k: float(selection_effects.data[k][b]) for k in ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']})

# del selection_effects
# selection_effects = gwpopulation.vt.ResamplingVT(vt_model, gwpop_data, n_events=opts.Nevents)#, unbias_factor=opts.UnbiasFactor)

if opts.dagnabbit:
    if opts.salvos_PE:
        with open(os.path.join(datadir, 'salvos_posteriors.pkl'), 'rb') as ff:
            posteriors = pkl.load(ff)
    else:
        if opts.use_old_events:
            with open(os.path.join(datadir, 'OLD_dagnabbit_large_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff) # THESE STILL INCLUDE SAMPLES BELOW THRESHOLD, THESE NEED TO BE CLEARED
        elif opts.use_small_events:
            sys.exit('These posteriors don\'t exist yet')
            with open(os.path.join(datadir, 'dagnabbit_small_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff)
        else:
            sys.exit('These posteriors don\'t exist yet')
            with open(os.path.join(datadir, 'dagnabbit_large_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff) # THESE STILL INCLUDE SAMPLES BELOW THRESHOLD, THESE NEED TO BE CLEARED
elif opts.delta_function_test:
    if opts.use_old_events:
        with open(os.path.join(datadir, 'OLD_true_parameters.pkl'), 'rb') as ff:
            true_pars = pkl.load(ff)
    else:
        with open(os.path.join(datadir, 'true_parameters.pkl'), 'rb') as ff:
            true_pars = pkl.load(ff)
    posteriors = []
    # print('minimum mass', np.min(true_pars['mass_1'].iloc[np.array([i for i in np.arange(550) if i not in [548, 534, 474, 470, 469, 464, 463, 429, 405, 383, 380, 376, 370, 348, 341, 320, 258, 228, 227, 211, 200, 191, 172, 127, 113, 100, 69, 61, 46, 32, 11]])]))
    
    for ii, par in true_pars.iterrows():
        p = pd.DataFrame({key: par[key]*np.ones(43_000) for key in ['mass_ratio', 'redshift', 'a_1', 'a_2', 'mass_1', 'tilt_1', 'tilt_2']})
        p['cos_tilt_1'] = np.cos(p.pop('tilt_1'))
        p['cos_tilt_2'] = np.cos(p.pop('tilt_2'))
        p['prior'] = np.ones(43_000) # make them this large just for easyness' sake
        posteriors.append(p)
else:
    if opts.salvos_PE:
        with open(os.path.join(datadir, 'salvos_posteriors.pkl'), 'rb') as ff:
            posteriors = pkl.load(ff)
    else:    
        if opts.use_old_events:
            with open(os.path.join(datadir, 'OLD_large_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff)
        elif opts.use_small_events:
            with open(os.path.join(datadir, 'small_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff) 
        elif opts.no_RB_events:
            with open(os.path.join(datadir, 'xp_noRB_small_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff) 
        elif opts.defaultSampleEvents:
            with open(os.path.join(datadir, 'xp_defaultSample_small_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff)
        else:
            sys.exit('These posteriors don\'t exist yet')
            with open(os.path.join(datadir, 'large_posteriors.pkl'), 'rb') as ff:
                posteriors = pkl.load(ff)

# mmin_thing = np.min([np.max(p['mass_1']*p['mass_ratio']) for p in posteriors])
# print(f'mmin thing = {mmin_thing}')

# mmax_thing = np.max([np.min(p['mass_1']*p['mass_ratio']) for p in posteriors])
# print(f'mmax thing = {mmax_thing}')

np.random.seed(1)
NPE = opts.NPE
for ii in range(len(posteriors)):
    prand = posteriors[ii].sample(frac=1) # randomize order, otherwise ordered in LL
    if NPE*(opts.pe_seed + 1) > len(posteriors[ii]['mass_1']):
        print(f'Seed {opts.pe_seed} is too large, we cannot draw an independent set of {NPE} PE samples for the {ii}th posterior with a total of {len(posteriors[ii]["mass_1"])} samples')
        print('Use different settings')
        quit()
    prand = prand.iloc[NPE*opts.pe_seed:NPE*(opts.pe_seed + 1)]
    prand = prand.reset_index(drop=True)
    posteriors[ii] = prand

if opts.use_old_events:
    if not opts.keep_bad_events:
        # bad_events = [11, 32, 46, 61, 69, 100, 113, 127, 172, 191, 200, 211, 227, 228, 258, 320, 341, 348, 370, 376, 380, 383, 405, 429, 463, 464, 469, 470, 474, 534, 548]
        print('Removing 60 bad events from the set of 550')
        # bad_events = [548, 534, 474, 470, 469, 464, 463, 429, 405, 383, 380, 376, 370, 348, 341, 320, 258, 228, 227, 211, 200, 191, 172, 127, 113, 100, 69, 61, 46, 32, 11]
        bad_events = [
            548, 534, 493, 490, 481, 474, 470, 469, 464, 463, 429, 405, 388, 387, 
            383, 380, 378, 376, 370, 361, 348, 343, 341, 334, 324, 320, 298, 292, 
            278, 274, 266, 264, 258, 251, 228, 227, 211, 206, 200, 191, 190, 187, 
            172, 157, 151, 137, 127, 113, 100, 80, 73, 69, 62, 61, 46, 32, 11, 
            7, 3, 2
            ]
        if opts.remove_207:
            bad_events = [
                548, 534, 493, 490, 481, 474, 470, 469, 464, 463, 429, 405, 388, 387, 
                383, 380, 378, 376, 370, 361, 348, 343, 341, 334, 324, 320, 298, 292, 
                278, 274, 266, 264, 258, 251, 228, 227, 211, 207, 206, 200, 191, 190,
                187, 172, 157, 151, 137, 127, 113, 100, 80, 73, 69, 62, 61, 46, 32, 11, 
                7, 3, 2
            ]
        for a in bad_events: # can also add 470, 69, 469, 100, 348, 463
            posteriors.pop(a)

posteriors = posteriors[:opts.Nevents]

if opts.dagnabbit:
    posterior_length_corrections = []
    for ii in range(len(posteriors)):
        p = posteriors[ii]
        if opts.salvos_PE:
            network = np.array(p['mf_net_snr'])
            keep = np.real(network) > 11    
        else:
            network = np.sqrt(np.array(p['H1_matched_filter_snr']).real**2 + np.array(p['L1_matched_filter_snr']).real**2)
            keep = np.real(network) > 9
        posterior_length_corrections.append(np.mean(keep))
        newprior = np.where(keep, posteriors[ii]['prior'], np.inf)
        posteriors[ii]['prior'] = newprior
else:
    posterior_length_corrections = np.ones(len(posteriors))

if opts.salvos_PE:
    if len(posteriors) >= 1067:
        posterior_length_corrections[1066] *= 2589 / 16118

print(f'Finished loading clean posteriors, using {len(posteriors)} events with {len(posteriors[0]["mass_1"])} PE samples')

if opts.UncertaintyCut:
    maximum_uncertainty = 1
else:
    maximum_uncertainty = np.inf

# like = DagNabbitCustomLikelihood(posteriors, model, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively

# jit_likelihood = gwpopulation.experimental.jax.JittedLikelihood(like)
# jit_likelihood.parameters.update(converted.copy())
# jit_likelihood.hyper_prior.parameters.update(converted.copy())

# if opts.print_single_event_variances:
#     event_bfs, event_vars = jit_likelihood._compute_per_event_ln_bayes_factors()
#     a = np.argsort(-event_vars)
#     b= event_vars[np.argsort(-event_vars)]
#     for ii in range(10):
#         print(a[ii], b[ii])
# print(jit_likelihood.ln_likelihood_and_variance())

# del like
# del jit_likelihood

like = DagNabbitCustomLikelihood(posteriors, model, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively
jit_likelihood = gwpopulation.experimental.jax.JittedLikelihood(like)

nlivestr = ''
if opts.nlive != 1000:
    nlivestr = f"_nlive{opts.nlive}"
RESULT_FILE = os.path.join(run_directory, 'sample_posterior' + nlivestr + '_result.json')
if os.path.exists(RESULT_FILE) and opts.remove_result:
    question = input(f'REMOVE RESULT FILE {RESULT_FILE}? (y/n) \n')
    if question.lower() in ['yes', 'y']:
        os.remove(RESULT_FILE)
        print('Result file removed.')
    else:
        print('Exiting.')
        sys.exit()
if not os.path.exists(RESULT_FILE):
    result = bilby.run_sampler(
        likelihood=jit_likelihood, priors=priors, sampler='dynesty', label='sample_posterior'+nlivestr,
        outdir=run_directory, nlive=opts.nlive, sample='acceptance-walk', naccept=10, resume=not opts.remove_result, 
        check_point_delta_t=180, checkpoint_every=3,
    )
else:
    print('Loading saved result')
    result = bilby.core.result.read_in_result(RESULT_FILE)

import jax
posterior = result.posterior
func = jax.jit(jit_likelihood.generate_extra_statistics)
print('computing extra statistics for posterior')
full_posterior = pd.DataFrame(
    [func(parameters) for parameters in result.posterior.to_dict(orient="records")]
).astype(float)
full_posterior['worst_neff'] = np.min(np.array([(full_posterior[f"var_{ii}"] + 1 / NPE / jit_likelihood.posterior_length_corrections[ii])**(-1) for ii in range(opts.Nevents)]), axis=0)

for ii in range(opts.Nevents):
    full_posterior.pop(f'ln_bf_{ii}')
    full_posterior.pop(f'var_{ii}')

result.posterior = pd.DataFrame(full_posterior)
truths = maxlike.copy()

truths.pop('amax')
if not opts.delta_function_test:
    truths['worst_neff'] = None
truths['selection'] = None
truths['variance'] = None

result.plot_corner(parameters=truths)