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
from models_and_utils import *

def ParseCommandLine():
    parser = optparse.OptionParser()

    parser.add_option('--NPE', type=int, default=16_000, help='size of PE MC integrals')
    parser.add_option('--NVT', type=int, default=10_000, help='size in millions of VT MC integral')
    parser.add_option('--Nevents', type=int, default=1000, help='number of events to include in the inference')
    parser.add_option('--homedir', default='../../', help='relative path to the mc-bias home directory')
    parser.add_option('--datadir', default='data/gw/pe-and-vt', help='path from home to data')
    parser.add_option('--resultdir', default='data/gw/results', help='path from home to result')
    parser.add_option('--nlive', type=int, default=1000)
    parser.add_option('--gpu', default='0')
    parser.add_option('--dagnabbit', action='store_true', default=False) #TODO 
    parser.add_option('--UncertaintyCut', action='store_true', default=False)
    parser.add_option('--delta_function_test', action='store_true', default=False)
    parser.add_option('--remove_result', action='store_true', default=False)
    parser.add_option('--PreallocateFrac', default='0.4')
    parser.add_option('--only_spin_inference', action='store_true', default=False)
    parser.add_option('--pe_seed', type=int, default=0)
    parser.add_option('--vt_seed', type=int, default=0)
    parser.add_option('--print_single_event_variances', action='store_true', default=False)
    parser.add_option('--compute_truth_likelihood', action='store_true', default=False)
    opts, args = parser.parse_args()

    return opts

opts = ParseCommandLine()
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
if opts.only_spin_inference:
    run_label = 'OnlySpin_' + run_label

sub_run_label = f'peseed_{opts.pe_seed}_vtseed_{opts.vt_seed}'
run_directory = os.path.join(resultdir, run_label, sub_run_label)
os.makedirs(run_directory, exist_ok=True)

non_spin_keys = ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m', 'lamb', 'f_spin', 'n_spin', 'mu_spin', 'sigma_spin']

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

if opts.only_spin_inference:
    for k in non_spin_keys:
        priors[k] = true_hyperpar[k]

maxlike = true_hyperpar.copy()

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
        gwpop_data[key] = float(gwpop_data[key])
    elif key in ['mass_ratio', 'redshift', 'a_1', 'a_2', 'mass_1', 'cos_tilt_1', 'cos_tilt_2', 'prior']:
        keep = xp.array(gwpop_data['snr']) > 11.
        gwpop_data[key] = xp.array(gwpop_data[key])[keep]

print(f"Downselecting {int(np.mean(keep)*1000)/10}% of samples from snr > 9 to snr > 11")

print(f"Using {len(gwpop_data['mass_1'])} found events from {gwpop_data['total_generated']}")
selection_effects = gwpopulation.vt.ResamplingVT(vt_model, gwpop_data, n_events=opts.Nevents)#, unbias_factor=opts.UnbiasFactor)

with open(os.path.join(datadir, 'salvos_posteriors.pkl'), 'rb') as ff:
    posteriors = pkl.load(ff)

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

posteriors = posteriors[:opts.Nevents]

if opts.dagnabbit:
    posterior_length_corrections = []
    for ii in range(len(posteriors)):
        p = posteriors[ii]
        network = np.array(p['mf_net_snr'])
        keep = np.real(network) > 11
        posterior_length_corrections.append(np.mean(keep))
        newprior = np.where(keep, posteriors[ii]['prior'], np.inf)
        posteriors[ii]['prior'] = newprior
else:
    posterior_length_corrections = np.ones(len(posteriors))

if len(posteriors) >= 1067:
    posterior_length_corrections[1066] *= 2589 / 16118

print(f'Finished loading clean posteriors, using {len(posteriors)} events with {len(posteriors[0]["mass_1"])} PE samples')

if opts.UncertaintyCut:
    maximum_uncertainty = 1
else:
    maximum_uncertainty = np.inf

like = DagNabbitCustomLikelihood(posteriors, model, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively
jit_likelihood = gwpopulation.experimental.jax.JittedLikelihood(like)

if opts.compute_truth_likelihood:
    jit_likelihood.parameters.update(true_hyperpar)
    jit_likelihood.hyper_prior.parameters.update(true_hyperpar)
    print('m, v', jit_likelihood.ln_likelihood_and_variance())    
    sys.exit()

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

print('Computing accuracy and precision statistics')

import jax
old_posterior = result.posterior

mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
vt_mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)
vt_redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)

spin_model = salvos_spin_model
vt_spin_model = salvos_spin_model

model = gwpopulation.experimental.jax.NonCachingModel([mass_model, spin_model, redshift_model])
vt_model = gwpopulation.experimental.jax.NonCachingModel([vt_mass_model, vt_spin_model, vt_redshift_model])


selection_effects = CorrectedResamplingVT(vt_model, gwpop_data, old_posterior, n_events=opts.Nevents, conversion_function=gwpopulation.conversions.convert_to_beta_parameters)#, unbias_factor=opts.UnbiasFactor)
like = CorrectedPlusDagNabbitCustomLikelihood(posteriors, model, old_posterior, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively
like.parameters = {}
jit_likelihood = gwpopulation.experimental.jax.JittedLikelihood(like)

loop_post = [{par: p[par] for par in priors} for p in result.posterior.to_dict(orient="records")]
# print(loop_post)
jit_likelihood.parameters.update(loop_post[0])
print('like:', jit_likelihood.log_likelihood_ratio())
func = jax.jit(jit_likelihood.generate_extra_statistics)
print('computing extra statistics for posterior')
full_posterior = pd.DataFrame(
    [func(loop_post[ii]) for ii in tqdm(range(len(loop_post)))]
).astype(float)
full_posterior['worst_neff'] = np.min(np.array([(full_posterior[f"var_{ii}"] + 1 / NPE / jit_likelihood.posterior_length_corrections[ii])**(-1) for ii in range(opts.Nevents)]), axis=0)

for ii in range(opts.Nevents):
    full_posterior.pop(f'ln_bf_{ii}')
    full_posterior.pop(f'var_{ii}')

result.posterior = pd.DataFrame(full_posterior)
truths = maxlike.copy()
if opts.only_spin_inference:
    for k in non_spin_keys:
        truths.pop(k)

truths.pop('amax')
if not opts.delta_function_test:
    truths['worst_neff'] = None

for k in ["variance", "total_correction", "likelihood_correction", "selection", "selection_variance"]:
    truths[k] = None

result.plot_corner(parameters=truths)

precision = np.mean(full_posterior['variance']) - np.mean(full_posterior['total_correction'] - full_posterior['likelihood_correction'])
precision /= 2*np.log(2)

accuracy = np.var(full_posterior['total_correction']) / 2 / np.log(2)
meanvar = np.mean(full_posterior["variance"])

print('precision', precision)
print('accuracy', accuracy)
print('mean var', meanvar)

with open(os.path.join(result.outdir, result.label + '_error_statistics.txt'), 'w') as ff:
    ff.write(f'Statistics measured in units of bits \n====================================\n')
    ff.write(f'precision \t {precision} \n')
    ff.write(f'accuracy \t {accuracy} \n')
    ff.write(f'error \t\t {precision+accuracy} \n')
    ff.write(f'mean var \t {meanvar} \n')
