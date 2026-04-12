import numpy as np
import scipy
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import logsumexp as LSE
from matplotlib import pyplot as plt
import pickle as pkl
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

import argparse

import os
from tqdm import tqdm

def parse_commandline():
    '''
    parse supplied options from command line
    '''
    parser = argparse.ArgumentParser(prog='python run_models.py', description='Run for producing samples from a MC estimated hierarchical posterior', epilog='text at bottom of help')

    parser.add_argument("--gpus", default='')
    parser.add_argument("--ndim", default='3', type=int)
    parser.add_argument("--nobs", default='100', type=int)
    parser.add_argument("--npe", default='100', type=int)
    parser.add_argument("--nrandom", default='100', type=int)
    parser.add_argument("--large_noise", action='store_true', default=False)
    parser.add_argument("--redo_suspicious", action='store_true', default=False)
    parser.add_argument("--load_analytical", action='store_true', default=False)
    opts = parser.parse_args()
    return opts

opts = parse_commandline()


os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpus
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'


ndim = opts.ndim
npe = opts.npe
nrandom = opts.nrandom
Nobs = opts.nobs
true_mean = 0
if opts.large_noise:
    true_sd = 0.25
    noise_dir = 'large_noise'
else:
    true_sd = 1.
    noise_dir = 'standard_noise'

true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,ndim)) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
noise = 1
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,ndim))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42)

print('ndim = ', ndim)

def calculate_contour(log_p_array, contour=np.array([0.9]), flatten=True):
    if flatten:
        log_p_array = log_p_array.flatten()
    ln_p_sort = -np.sort(-log_p_array) # sort from big to small
    cdf = np.cumsum(np.exp(ln_p_sort))
    cdf /= cdf[-1] # normalize

    p_boundary = np.interp(contour, cdf, ln_p_sort) # pdf boundary between inner and outer contour
    return p_boundary
    
def draw_PE_sample(PRNGkey, NPE, observations, noise_sigma):
    scatter = jax.random.normal(PRNGkey, shape=(observations.shape[0], ndim, NPE)) * noise_sigma
    r = jnp.expand_dims(observations, axis=2) + scatter
    # print(r.shape)
    return r

def log_gaussian(x, mu, sigma):
    p = jnp.sum(-(x - mu)**2 / 2 / sigma**2, axis=1) - ndim*0.5*jnp.log(2*jnp.pi) - jnp.sum(jnp.log(sigma), axis=1)
    # print(p.shape)
    return p

@jax.jit
def naive_log_likelihood_estimator(mu, sigma, observations_array):
    # expand dims to right shapes
    
    # num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    # observations_array = jnp.expand_dims(observations_array, axis=tuple([2+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,:,None], sigma[None,:,None])
    NPE = obs_weights.shape[1]

    # runs out of memory, split up computation over mu and sigma subs...
    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    # print(numerator, denominator)

    var_numerator = jnp.exp(LSE(2*obs_weights, axis=1) - 2*jnp.log(NPE) - 2*numerator) - 1/NPE
    num_variance = jnp.sum(var_numerator, axis=0)
    neffs = jnp.exp(2*LSE(obs_weights, axis=1) - LSE(2*obs_weights, axis=1))
    worst_neff = jnp.min(neffs, axis=0)

    return jnp.sum(numerator, axis=0), num_variance, worst_neff

@jax.jit
def numerator_covariance_term(mu, sigma, mup, sigmap, observations_array):

    num_dimension = len(jnp.shape(mu))
    observations_array = jnp.expand_dims(observations_array, axis=tuple([3+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,:,None], sigma[None,:,None])
    obs_weights_p = log_gaussian(observations_array, mup[None,:,None], sigmap[None,:,None]) # shape (Nobs, Npe)
    NPE = obs_weights.shape[1]

    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    numerator_p = LSE(obs_weights_p, axis=1) - jnp.log(NPE) 

    cov_numerator = LSE(obs_weights+obs_weights_p, axis=1)-numerator-numerator_p-jnp.log(NPE)-jnp.log(NPE-1)
    cov_numerator = jnp.exp(cov_numerator) - 1 / (NPE-1)
    cov = jnp.sum(cov_numerator, axis=0)

    return cov

def posterior_bias_correction(mu_samples, sigma_samples, observations, npe, noise_sigma, big=False):
    '''
    idea is that mu_samples, sigma_samples should be from the biased posterior
    we can derive the correction from this
    '''
    assert mu_samples.shape[0] == sigma_samples.shape[0] # ensure same number of samples each
    observations_array = draw_PE_sample(PRNGkey, npe, observations, noise_sigma)

    f = lambda i,j: numerator_covariance_term(mu_samples[i], sigma_samples[i], 
                        mu_samples[j], sigma_samples[j], observations_array)
    
    indexes = jnp.arange(mu_samples.shape[0])
    ii, jj = jnp.meshgrid(indexes, indexes)
    cov = jax.tree.map(f, ii, jj) # Nsamples by Nsamples array
    assert (cov == cov.T).all() # ensure symmetric

    weights = jnp.mean(cov, axis=1)
    return weights

def analytical_likelihood(mu, sigma, observations_centers, noise_sigma):

    Nobs = len(observations_centers)
    log_l_norm = -(Nobs / 2) * jnp.sum(jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2)))
    expo = -jnp.sum((observations_centers - mu[None,:])**2 / 2 / (sigma[None,:]**2 + noise_sigma**2), axis=(0,1))
    ll = expo + log_l_norm
    return ll

def analytical_posterior(mu, sigma_log, observations_centers, noise_sigma, dsigma):

    ll = analytical_likelihood(mu, jnp.exp(10*sigma_log), observations_centers, noise_sigma)
    log_evidence = LSE(ll) + jnp.log(dsigma)
    log_posterior = ll - log_evidence
    return log_posterior, log_evidence

import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
constraints = dist.constraints

def analytical_model(data=None):
    mus = numpyro.sample('mu', dist.ImproperUniform(constraints.real, (), event_shape=(ndim,)))
    sigmas = numpyro.sample('sigma', dist.ImproperUniform(constraints.positive, (), event_shape=(ndim,)))

    ll = analytical_likelihood(mus, sigmas, data, noise)
    numpyro.factor('log_likelihood', ll)

def numerical_model(data=None):

    mus = numpyro.sample('mu', dist.ImproperUniform(constraints.real, (), event_shape=(ndim,)))
    sigmas = numpyro.sample('sigma', dist.ImproperUniform(constraints.positive, (), event_shape=(ndim,)))

    ll, var, wneff = naive_log_likelihood_estimator(mus, sigmas, data)
    numpyro.deterministic('ll_variance', var)
    numpyro.deterministic('worst_neff', wneff)
    
    numpyro.factor('log_likelihood', ll)

def run_analytical_model(load_old=False, pkl_loc=''):
    if load_old:
        try:
            with open(pkl_loc, 'rb') as ff:
                d = pkl.load(ff)    
        except FileNotFoundError: # what kind of exception?
            raise FileNotFoundError('File does not exist, cannot redo suspicious runs')
        old_analytical = d['analytical_samples']
        return old_analytical

    analytical_nuts_kernel = NUTS(analytical_model)
    analytical_mcmc = MCMC(analytical_nuts_kernel, num_warmup=1000, num_samples=10000)
    analytical_mcmc.run(jr.PRNGKey(1), data=obs)

    analytical_mcmc.print_summary()
    samples = analytical_mcmc.get_samples()
    samples = np.concatenate((samples['mu'], samples['sigma']), axis=1)
    return samples

def run_numerical_models(sus=False, pkl_loc=''):
    rhat_thresh = 1.01
    neff_thresh = 500

    numerical_hierarchical_posterior_samples = []
    PRNGkey = jr.PRNGKey(42)
    if sus:
        try:
            with open(pkl_loc, 'rb') as ff:
                d = pkl.load(ff)    
        except FileNotFoundError: # what kind of exception?
            raise FileNotFoundError('File does not exist, cannot redo suspicious runs')
        # old_analytical = d['analytical_samples']
        old_samples = d['samples']
        old_summary = d['meta_data']
    exceptions = ['ll_variance', 'worst_neff']
    summary = []
    iterable = tqdm(range(nrandom))
    for ii in iterable:

        PRNGkey, _ = jr.split(PRNGkey) # want to iterate the PRNG key every time so we make sure sus runs have the same rng key
        redo = True
        mult = 1
        warmup_mult = 1
        if sus:
            oldsumm = old_summary[ii]
            worst_rhat = np.max([oldsumm[key]['r_hat'] for key in oldsumm if key not in exceptions])
            worst_neff = np.min([oldsumm[key]['n_eff'] for key in oldsumm if key not in exceptions])
            if (worst_rhat < rhat_thresh) and (worst_neff > neff_thresh):
                summary.append(oldsumm)
                numerical_hierarchical_posterior_samples.append(old_samples[ii])
                redo = False
            else:
                redo = True
                warmup_mult = 5
                mult = 5*(int(neff_thresh / worst_neff)+1)
        if redo:
            observations_array = draw_PE_sample(PRNGkey, npe, obs, noise)

            numerical_nuts_kernel = NUTS(numerical_model)
            numerical_mcmc = MCMC(numerical_nuts_kernel, num_warmup=warmup_mult*300, num_samples=mult*1000, progress_bar=False, thinning=mult)
            numerical_mcmc.run(jr.PRNGKey(1), data=observations_array)
            
            samples = numerical_mcmc.get_samples()
            summ = numpyro.diagnostics.summary(samples, group_by_chain=False)
            worst_rhat = np.max([summ[key]['r_hat'] for key in summ if key not in exceptions])
            worst_neff = np.min([summ[key]['n_eff'] for key in summ if key not in exceptions])
            if worst_rhat > rhat_thresh:
                print(f'warning: rhat = {worst_rhat} > {rhat_thresh}')
            if worst_neff < neff_thresh:
                print(f'warning: neff = {worst_neff} < {neff_thresh}')
            iterable.set_description(f'Worst Neff = {worst_neff}, Worst Rhat = {worst_rhat}')
            summary.append(summ)
            samples = np.concatenate((samples['mu'], samples['sigma']), axis=1)
            
            numerical_hierarchical_posterior_samples.append(samples)
        
    numerical_hierarchical_posterior_samples = np.array(numerical_hierarchical_posterior_samples)
    return numerical_hierarchical_posterior_samples, summary


dir_location = f'data/{noise_dir}/Ndim{ndim}_Nobs{Nobs}/'
os.makedirs(dir_location, exist_ok=True)
pkl_loc = dir_location + f'npe_{npe}_nrepeat{nrandom}.pkl'
print(pkl_loc)
if opts.load_analytical:
    load_old = True
    print('Loading old analytical samples')
else:
    print('Redoing analytical run')
    load_old = False
analytical_samples = run_analytical_model(load_old=load_old, pkl_loc=pkl_loc)

if opts.redo_suspicious:
    numerical_hierarchical_posterior_samples, summary = run_numerical_models(sus=True, pkl_loc=pkl_loc)
else:
    numerical_hierarchical_posterior_samples, summary = run_numerical_models(sus=False)

with open(pkl_loc, 'wb') as ff:
    pkl.dump({'analytical_samples': analytical_samples, 'samples': numerical_hierarchical_posterior_samples, 'meta_data': summary}, ff)



