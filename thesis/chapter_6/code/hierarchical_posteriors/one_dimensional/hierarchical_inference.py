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
import os
import argparse
from tqdm import tqdm

def parse_commandline():
    '''
    parse supplied options from command line
    '''
    parser = argparse.ArgumentParser(prog='python run_models.py', description='Run for producing samples from a MC estimated hierarchical posterior', epilog='text at bottom of help')

    parser.add_argument("--gpus", default='')
    parser.add_argument("--ndim", default=1, type=int)
    parser.add_argument("--nobs", default=100, type=int)
    parser.add_argument("--npe", default=100, type=int)
    parser.add_argument("--nrandom", default=100, type=int)
    parser.add_argument("--noise_sd", default=1, type=float)
    parser.add_argument("--redo_suspicious", action='store_true', default=False)
    parser.add_argument("--load_analytical", action='store_true', default=False)
    parser.add_argument("--minsigma", default=0.25, type=float)
    parser.add_argument("--maxsigma", default=1.75, type=float)
    parser.add_argument("--density", default=1000, type=int)
    parser.add_argument("--cov_batch_size", default=1, type=int)
    
    opts = parser.parse_args()
    return opts

opts = parse_commandline()
ndim = opts.ndim

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpus
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

def draw_PE_sample(PRNGkey, NPE, observations, noise_sigma):
    scatter = jax.random.normal(PRNGkey, shape=(observations.shape[0], ndim, NPE)) * noise_sigma
    r = jnp.expand_dims(observations, axis=2) + scatter
    # print(r.shape)
    # print(noise_sigma)
    return r

def log_gaussian(x, mu, sigma):
    p = jnp.sum(-(x - mu)**2 / 2, axis=1) / sigma**2 - ndim*0.5*jnp.log(2*jnp.pi) - ndim*jnp.log(sigma)
    # print(p.shape)
    return p

@jax.jit
def naive_log_likelihood_estimator(mu, sigma, observations_array):
    # expand dims to right shapes
    
    num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    observations_array = jnp.expand_dims(observations_array, axis=tuple([3+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,None,None,...], sigma[None,None,...])
    NPE = obs_weights.shape[1]

    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)

    var_numerator = jnp.exp(LSE(2*obs_weights, axis=1) - 2*jnp.log(NPE) - 2*numerator) - 1/NPE
    num_variance = jnp.sum(var_numerator, axis=0)
    neffs = jnp.exp(2*LSE(obs_weights, axis=1) - LSE(2*obs_weights, axis=1))
    worst_neff = jnp.min(neffs, axis=0)

    return jnp.sum(numerator, axis=0), num_variance, worst_neff

@jax.jit # new addition? Does jit speed it up?
def covariance_term(mu, sigma, observations_array):
    
    num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    observations_array = jnp.expand_dims(observations_array, axis=tuple([3+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,None,None,...], sigma[None,None,...])
    NPE = obs_weights.shape[1]

    obs_weights_p = obs_weights[:,:,None,:] # shape (Nobs, Npe, 1, Nsigma)

    numerator_p = LSE(obs_weights_p, axis=1) - jnp.log(NPE) 
    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    
    # old method : 
    f = lambda index: jnp.sum(jnp.exp(
        LSE(obs_weights_p[...,index]+obs_weights, axis=1)-numerator_p[...,index]-numerator-jnp.log(NPE-1)-jnp.log(NPE) 
    ) - 1 / (NPE-1), axis=0)
    # cov_numerator = jax.lax.map(f, jnp.arange(obs_weights.shape[0]))
    cov = jax.lax.map(f, jnp.arange(obs_weights.shape[-1]), batch_size=opts.cov_batch_size)
    # cov_numerator = jnp.exp(cov_numerator) - 1 / (NPE-1)
    
    # cov = jnp.sum(cov_numerator, axis=0)#  - jnp.sum(numerator, axis=0) - jnp.sum(numerator_p, axis=0)

    return cov

def random_posterior(PRNGkey, mu, sigma, observations, npe, noise_sigma, dsigma):

    observations_array = draw_PE_sample(PRNGkey, npe, observations, noise_sigma)
    PRNGkey, _ = jax.random.split(PRNGkey)
    
    lls, vs, wneff = naive_log_likelihood_estimator(mu, sigma, observations_array)
    
    log_evidence = LSE(lls) + jnp.log(dsigma)
    log_posterior = lls - log_evidence

    cov = covariance_term(mu, sigma, observations_array)
    # print(cov)
    lps = [log_posterior]
    evidences = [log_evidence]
    corrected_posterior = log_posterior
    corrections = [0*log_posterior]
    mean_corrections = [0]
    for iii in range(1, 3):
        
        correction = jnp.sum(jnp.exp(corrected_posterior[:,None])*cov, axis=0)*dsigma
        mean_correction = jnp.sum(jnp.exp(corrected_posterior)*correction)*dsigma
        
        correction -= jnp.mean(correction)

        corrected_posterior = log_posterior + correction # + next_order# + 0.5*second_correction #  can do different thing, e.g. + jnp.log1p(correction)
        
        corr_log_evidence = LSE(corrected_posterior) + jnp.log(dsigma)
        corrected_posterior -= corr_log_evidence
        lps.append(corrected_posterior)
        evidences.append(corr_log_evidence)
        corrections.append(correction)
        mean_corrections.append(mean_correction)
        
    return lps, evidences, vs, wneff, corrections, mean_corrections

# random_posterior = jax.jit(random_posterior, static_argnums=(3,)) # array is not hashable

def analytical_posterior(mu, sigma, observations_centers, noise_sigma, dsigma):
    '''
    check how this works for arbitrary ndim
    '''
    Nobs = len(observations_centers)
    log_l_norm = -ndim * (Nobs / 2) * jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2))
    expo = -jnp.sum((jnp.expand_dims(observations_centers, tuple(np.arange(2,len(mu.shape)+2))) - mu[None,None,...])**2, axis=(0,1)) / 2 / (sigma**2 + noise_sigma**2)
    ll = expo + log_l_norm
    log_evidence = LSE(ll) + jnp.log(dsigma)
    log_posterior = ll - log_evidence
    return log_posterior, log_evidence

Nobs = opts.nobs
true_mean = 0
true_sd = 1
true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,ndim)) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
noise = opts.noise_sd
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,ndim))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42)

kl_list = []
ckl_list = []
pest_list = []
cpest_list = []
panal_list = []

npe = opts.npe
PRNGkey, _ = jr.split(PRNGkey)

if npe >= 5000:
    big = True
else:
    big = False
# @jax.jit

sigmas = jnp.linspace(opts.minsigma, opts.maxsigma, opts.density)

mus = jnp.full_like(sigmas, true_mean)
dsigma = sigmas[1] - sigmas[0]

def kl_and_ptrue(PRNGkey):
    # random_posterior(PRNGkey, mu, sigma, observations, npe, noise_sigma, dsigma)
    log_posteriors, log_evidences, vs, wneff, corrections, mean_corrections = random_posterior(PRNGkey, mus, sigmas, obs, npe, noise, dsigma)
    analytic_log_posterior, analytic_log_evidence = analytical_posterior(mus, sigmas, obs, noise, dsigma)

    KL_divs = [jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - lp)) * dsigma / jnp.log(2) for lp in log_posteriors]

    return KL_divs, log_posteriors, corrections, mean_corrections, log_evidences, vs, wneff, analytic_log_posterior, analytic_log_evidence

n_repeat = opts.nrandom
keys = jr.split(PRNGkey, n_repeat)

kl, pest, corrs, mcs, zs, vs, ws  = [], [], [], [], [], [], []
klist = tqdm(keys)
klist.set_description(f'N_PE = {npe}')
for key in klist:
    k, prob, c, mc, z, v, w, panalytic, zanalytic = kl_and_ptrue(key)
    kl.append(k)
    pest.append(prob)
    corrs.append(c)
    mcs.append(mc)
    zs.append(z)
    vs.append(v)
    ws.append(w)

if opts.density != 1000:
    densityStr = f'density{opts.density}_'
else:
    densityStr = ""
os.makedirs(f'../../../data/one_dimensional_data/noise{int(noise)}p{int((noise - int(noise))*10)}_Nobs{Nobs}_nsample{n_repeat}_{densityStr}posteriors/', exist_ok=True)
with open(f'../../../data/one_dimensional_data/noise{int(noise)}p{int((noise - int(noise))*10)}_Nobs{Nobs}_nsample{n_repeat}_{densityStr}posteriors/npe_{npe}.pkl', 'wb') as ff:
    pkl.dump({'KL': np.array(kl), 
              'posteriors': np.array(pest), 
              'corrections': np.array(corrs),
              'mean_covariance': np.array(mcs),
              'mean_variance': np.array(vs),
              'evidence': np.array(zs),
              'minimum_effective_samples': np.array(ws),
              'mus': np.array(mus), 
              'sigmas': np.array(sigmas),
              'analytic_posterior': np.array(panalytic),
              'analytic_evidence': np.array(zanalytic)}, ff)