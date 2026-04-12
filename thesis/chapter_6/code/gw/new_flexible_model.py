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
    parser.add_option('--NVT', type=int, default=1_000, help='size in millions of VT MC integral')
    parser.add_option('--Nevents', type=int, default=1000, help='number of events to include in the inference')
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
    parser.add_option('--PreallocateFrac', default='0.5')

    parser.add_option('--MinimumNeff', type=float, default=0.)
    parser.add_option('--pin_extra_params', action='store_true')
    
    parser.add_option('--no_exp', action='store_true', default=False)
    parser.add_option('--expm1', action='store_true', default=False)

    parser.add_option('--nodes', type=int, default=10)
    parser.add_option('--pe_seed', type=int, default=0)
    parser.add_option('--vt_seed', type=int, default=0)
    parser.add_option('--turn_off_secondary_smoothing', action='store_true', default=False, 
                      help='Smoothing the q distribution causes a sharp peak at q=1 for low primary mass binaries. This causes ridiculously poor reweighting efficiency for these events. Turning it off can help the likelihood variance dramatically.')
    parser.add_option('--print_single_event_variances', action='store_true', default=False)
    parser.add_option('--compute_truth_likelihood', action='store_true', default=False)
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
    true_spin_mag_func = lambda x: 14.9331056819 * x**0.67 * (1-x)**3.43
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
        'f_spin': 1.0,
        'n_spin': 2.0,
        'mu_spin': 1.0,
        'sigma_spin': 1.15,
        }
    x_nodes = np.linspace(0,1,opts.nodes)
    for ii, n in enumerate(x_nodes):
        if opts.no_exp:
            true_hyperpar[f'fa{ii}'] = true_spin_mag_func(n)
        elif opts.expm1:
            true_hyperpar[f'fa{ii}'] = np.log1p(true_spin_mag_func(n))
        else:
            true_hyperpar[f'fa{ii}'] = np.maximum(-10, np.log(true_spin_mag_func(n)))


os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = opts.PreallocateFrac

datadir = os.path.join(opts.homedir, opts.datadir)
resultdir = os.path.join(opts.homedir, opts.resultdir)

run_label = f'nnodes_{opts.nodes}_nevents_{opts.Nevents}_nvt_{opts.NVT}M_npe_{opts.NPE}'
if opts.UncertaintyCut:
    run_label = 'varcut1_' + run_label
if opts.MinimumNeff >= 1.:
    run_label = f'MinimumNeff{int(opts.MinimumNeff)}_' + run_label
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
if opts.no_exp:
    run_label = 'no_exp_spin_model_' + run_label
elif opts.expm1:
    run_label = 'expm1_' + run_label
if opts.pin_extra_params:
    run_label = 'pin_extra_params_' + run_label 


if opts.salvos_PE:
    run_label = 'new_flexible_models_salvo_pe/' + run_label
else:
    print("WARNING, USING OLD PE")
    run_label = 'old_results_flexible_models/' + run_label

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
    'f_spin': bilby.prior.Uniform(0, 1, name='f_spin', latex_label='$f_{\\rm spin}$'), 
    'n_spin': bilby.prior.Uniform(-5, 5, name='n_spin', latex_label='$n_{\\rm spin}$'), 
    'mu_spin': bilby.prior.Uniform(-1, 1, name='mu_spin', latex_label='$\\mu_{\\rm spin}$'),
    'sigma_spin': bilby.prior.Uniform(0.1, 4, name='sigma_spin', latex_label='$\\sigma_{\\rm spin}$'), 
}

if opts.pin_extra_params:
    for p in true_hyperpar:
        priors[p] = true_hyperpar[p]

for n in range(opts.nodes):
    p = f'fa{n}'
    if opts.UncertaintyCut:
        priors[p] = bilby.core.prior.Uniform(-5, 5, name=p, latex_label=f"$f_{{a:{n}}}$")
    else:
        priors[p] = bilby.core.prior.Uniform(-10, 10, name=p, latex_label=f"$f_{{a:{n}}}$")
# for n in range(opts.nodes-1):
#     priors[f'fa{n}'] = bilby.prior.Uniform(-10, 10, opts.nodes, label='fa')
# priors.pop(f'fa{opts.nodes - 1}')

maxlike = true_hyperpar.copy()

truths = maxlike.copy()

truths.pop('amax')
if not opts.delta_function_test:
    truths['worst_neff'] = None
truths['selection'] = None
truths['variance'] = None
if opts.pin_extra_params:
    for p in priors:
        if isinstance(priors[p], float) and p in truths:
            truths.pop(p)
# print(truths)

if opts.turn_off_secondary_smoothing:
    mass_model = NoSecondarySmoothingMass()
    vt_mass_model = NoSecondarySmoothingMass()
else:
    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    vt_mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)
vt_redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)

spin_model = OldLinearInterpolateModel(opts.nodes)
vt_spin_model = OldLinearInterpolateModel(opts.nodes)

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
        if opts.salvos_PE:
            keep = xp.array(gwpop_data['snr']) > 11.
            gwpop_data[key] = xp.array(gwpop_data[key])[keep]
        else:
            gwpop_data[key] = xp.array(gwpop_data[key])
if opts.salvos_PE:
    print(f"Downselecting {int(np.mean(keep)*1000)/10}% of samples from snr > 9 to snr > 11")

print(f"Using {len(gwpop_data['mass_1'])} found events from {gwpop_data['total_generated']}")
selection_effects = gwpopulation.vt.ResamplingVT(vt_model, gwpop_data, n_events=opts.Nevents)#, unbias_factor=opts.UnbiasFactor)

# random = bilby.core.prior.ConditionalPriorDict(priors).sample(10)
# print(random)
# print(np.sum([random[f'fa{ii}'] for ii in range(opts.nodes-1)], axis=0))
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
        print('loading dagnabbit salvo PE')
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
            print('loading salvo PE')
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
    if len(posteriors) >= 1600:
        posterior_length_corrections[1599] *= 2589 / 16118

print(f'Finished loading clean posteriors, using {len(posteriors)} events with {len(posteriors[0]["mass_1"])} PE samples')

if opts.UncertaintyCut:
    maximum_uncertainty = 1
else:
    maximum_uncertainty = np.inf

like = DagNabbitCustomLikelihood(posteriors, model, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively
like = gwpopulation.experimental.jax.JittedLikelihood(like)

like_par = true_hyperpar.copy()
like.parameters.update(like_par)
like.hyper_prior.parameters.update(like_par)
print('m, v=',like.ln_likelihood_and_variance())
if opts.compute_truth_likelihood:
    sys.exit()

like = DagNabbitCustomLikelihood(posteriors, model, selection_function=selection_effects, maximum_uncertainty=maximum_uncertainty, max_samples=NPE, conversion_function=gwpopulation.conversions.convert_to_beta_parameters, posterior_length_corrections=posterior_length_corrections, minimum_neff=opts.MinimumNeff)#, fid_parameters=maxlike, reject_zscore=10) # reject more aggressively
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
        likelihood=jit_likelihood, priors=bilby.core.prior.ConditionalPriorDict(priors), sampler='dynesty', label='sample_posterior'+nlivestr,
        outdir=run_directory, nlive=opts.nlive, sample='acceptance-walk', naccept=10, resume=not opts.remove_result, 
        check_point_delta_t=200,
    )
    
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
    result.save_to_file()

else:
    print('Loading saved result')
    result = bilby.core.result.read_in_result(RESULT_FILE)
    import jax

# result.plot_corner(parameters=truths)

# normalize to remove degeneracy in posterior
posterior = result.posterior
fs_keys = spin_model.variable_names

truths_temp = truths.copy()
truths = {}
for k in sorted(list(truths_temp.keys())):
    truths[k] = truths_temp[k]

fs = xp.array([posterior[v] for v in fs_keys])
log_norms = xp.log(xp.array([spin_model.norm(fs[:,ii]) for ii in range(len(result.posterior))]))
for v in fs_keys:
    if opts.no_exp:
        posterior[v] = posterior[v]/xp.exp(log_norms)
    elif opts.expm1:
        posterior[v] = xp.log1p(xp.array(posterior[v]/xp.exp(log_norms)))
    else:
        posterior[v] = posterior[v] - log_norms
result.posterior = posterior
# result.label = result.label + '_normed_heights'
# result.plot_corner(parameters=truths)

a_spins = xp.linspace(0,1,1000)
if opts.no_exp:
    fa_spins = xp.array([xp.interp(a_spins, spin_model.nodes, fs[:,ii]) / xp.exp(log_norms[ii]) for ii in range(len(result.posterior))])
elif opts.expm1:
    fa_spins = xp.expm1(xp.array([xp.interp(a_spins, spin_model.nodes, fs[:,ii]) / xp.exp(log_norms[ii]) for ii in range(len(result.posterior))]))
else:
    fa_spins = xp.exp(xp.array([xp.interp(a_spins, spin_model.nodes, fs[:,ii] - log_norms[ii]) for ii in range(len(result.posterior))]))
fa_true = true_spin_mag_func(a_spins)

plt.plot(a_spins, fa_true, 'k-', label='Truth')
plt.fill_between(a_spins, xp.percentile(fa_spins, 5, 0), xp.percentile(fa_spins, 95, 0), color='b', alpha=0.3, label='inferred')

plt.xlim(0,1)
plt.xlabel('$a$')
plt.ylabel('$p(a)$')
plt.legend()
plt.savefig(os.path.join(run_directory, 'sample_posterior' + nlivestr + '_spin_magnitudes.pdf'))
with open(RESULT_FILE.replace('_result.json', '_spin_mag.pkl'), 'wb') as ff:
    pkl.dump({'a': np.array(a_spins), 'p_a': np.array(fa_spins), 'truth': np.array(fa_true)}, ff)

old_posterior = posterior

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

if 'amax' in truths:
    truths.pop('amax')
if not opts.delta_function_test:
    truths['worst_neff'] = None

for k in ["variance", "total_correction", "likelihood_correction", "selection", "selection_variance"]:
    truths[k] = None

result.plot_corner(parameters=truths)

precision = np.mean(result.posterior['variance']) - np.mean(result.posterior['total_correction'] - result.posterior['likelihood_correction'])
precision /= 2*np.log(2)

accuracy = np.var(result.posterior['total_correction']) / 2 / np.log(2)

print('precision', precision)
print('accuracy', accuracy)

with open(os.path.join(result.outdir, result.label + '_error_statistics.txt'), 'w') as ff:
    ff.write(f'Statistics measured in units of bits \n====================================\n')
    ff.write(f'precision \t {precision} \n')
    ff.write(f'accuracy \t {accuracy} \n')
    ff.write(f'error \t\t {precision+accuracy} \n')
