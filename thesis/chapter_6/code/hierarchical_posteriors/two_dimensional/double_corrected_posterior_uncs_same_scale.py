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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

from tqdm import tqdm

# generate injections from population, without selection effects
# implement the p value test of https://arxiv.org/pdf/2304.06138
ndim = 1
print('ndim = ', ndim)
def draw_PE_sample(PRNGkey, NPE, observations, noise_sigma):
    scatter = jax.random.normal(PRNGkey, shape=(len(observations), ndim, NPE)) * noise_sigma
    r = jnp.expand_dims(observations, axis=2) + scatter
    # print(r.shape)
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

    # runs out of memory, split up computation over mu and sigma subs...
    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    # print(numerator, denominator)

    var_numerator = jnp.exp(LSE(2*obs_weights, axis=1) - 2*jnp.log(NPE) - 2*numerator) - 1/NPE
    num_variance = jnp.sum(var_numerator, axis=0)
    neffs = jnp.exp(2*LSE(obs_weights, axis=1) - LSE(2*obs_weights, axis=1))
    worst_neff = jnp.min(neffs, axis=0)

    return jnp.sum(numerator, axis=0), num_variance, worst_neff

@jax.jit
def covariance_term(mu, sigma, mup, sigmap, observations_array):
    # expand dims to right shapes
    
    num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    # print(jnp.shape(mu))
    observations_array = jnp.expand_dims(observations_array, axis=tuple([3+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,None,None,...], sigma[None,None,...])
    obs_weights_p = log_gaussian(observations_array, mup[None,None,None,...], sigmap[None,None,...]) # shape (Nobs, Npe, mu1, mu2)
    NPE = obs_weights.shape[1]

    obs_weights = obs_weights[:,:,None,None,:,:]
    obs_weights_p = obs_weights_p[...,None,None]

    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    numerator_p = LSE(obs_weights_p, axis=1) - jnp.log(NPE) 
    
    cov_numerator = jnp.exp(LSE(obs_weights+obs_weights_p, axis=1)-numerator-numerator_p-jnp.log(NPE)-jnp.log(NPE-1)) - 1 / (NPE-1)
    # print(numerator, denominator)
    cov = jnp.sum(cov_numerator, axis=0)
    
    return cov

def random_posterior(PRNGkey, mu, sigma, observations, npe, noise_sigma, dmu, dsigma, big=False):

    observations_array = draw_PE_sample(PRNGkey, npe, observations, noise_sigma)

    if big:
        lls, vs, wneff = jax.lax.map(
            lambda x: naive_log_likelihood_estimator(x[0], jnp.exp(jnp.log(10) * x[1]), observations_array),
            jnp.array([mu, sigma]).swapaxes(0,1)
        )
    else:
        lls, vs, wneff = naive_log_likelihood_estimator(mu, jnp.exp(jnp.log(10) * sigma), observations_array)
    
    log_evidence = LSE(lls) + jnp.log(dsigma * dmu)
    log_posterior = lls - log_evidence

    cov = np.array([covariance_term(mu, jnp.exp(jnp.log(10) * sigma), 
                m[:,None], jnp.exp(jnp.log(10) * s[:,None]), observations_array)[:,0,:,:] for m, s in zip(mu, sigma)
                ])
    
    lps = [log_posterior]
    evidences = [log_evidence]
    corrected_posterior = log_posterior
    for iii in range(1, 10):
        correction = np.sum(cov*jnp.exp(corrected_posterior[...,None,None]), axis=(0,1)) * dsigma * dmu
        corrected_posterior = log_posterior + correction
        corr_log_evidence = LSE(corrected_posterior) + jnp.log(dsigma * dmu)
        corrected_posterior -= corr_log_evidence
        lps.append(corrected_posterior)
        evidences.append(corr_log_evidence)
        
    return lps, evidences, vs, wneff

def analytical_posterior(mu, sigma_log, observations_centers, noise_sigma, dmu, dsigma):
    '''
    check how this works for arbitrary ndim
    '''
    Nobs = len(observations_centers)
    sigma = jnp.exp(jnp.log(10) * sigma_log)
    log_l_norm = -ndim * (Nobs / 2) * jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2))
    expo = -jnp.sum((jnp.expand_dims(observations_centers, tuple(np.arange(2,len(mu.shape)+2))) - mu[None,None,...])**2, axis=(0,1)) / 2 / (sigma**2 + noise_sigma**2)
    ll = expo + log_l_norm
    log_evidence = LSE(ll) + jnp.log(dsigma * dmu)
    log_posterior = ll - log_evidence
    return log_posterior, log_evidence

Nobs = 1000
true_mean = 1
true_sd = 1
true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,ndim)) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
noise = 1
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,ndim))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42)

kl_list = []
ckl_list = []
pest_list = []
cpest_list = []
panal_list = []

npelist = np.logspace(1,3,60, dtype=int)
# npelist = [10,20,50,100,200,1000]

print(npelist)
for npe in npelist:
    PRNGkey, _ = jr.split(PRNGkey)

    if npe >= 500:
        big = True
    else:
        big = False
    # @jax.jit

    sigmas = jnp.linspace(-0.3,0.2,101)
    mus = jnp.linspace(0.5,1.5,101)

    dsigma = sigmas[1] - sigmas[0]
    dmu = mus[1] - mus[0]

    mu_mesh, sig_mesh = jnp.meshgrid(mus, sigmas)

    def kl_and_ptrue(PRNGkey):

        log_posteriors, log_evidences, vs, wneff = random_posterior(PRNGkey, mu_mesh, sig_mesh, obs, npe, noise, dmu, dsigma, big=big)
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_mesh, sig_mesh, obs, noise, dmu, dsigma)

        KL_divs = [jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - lp)) * dmu * dsigma / jnp.log(2) for lp in log_posteriors]

        return KL_divs, log_posteriors, analytic_log_posterior

    n_repeat = 100
    keys = jr.split(PRNGkey, n_repeat)
    # kl, p_est, p_anal = jax.lax.map(kl_and_ptrue, keys, batch_size=1)
    # eh, I'd rather have a tqdm progress bar lol
    kl, ckl, c2kl, pest, cpest, c2pest, apest = [], [], [], [], [], [], []
    klist = tqdm(keys)
    klist.set_description(f'N_PE = {npe}')
    for key in klist:
        k, p, pw = kl_and_ptrue(key)
        kl.append(k)
        pest.append(p)
        apest.append(pw)

    # kl_list.append(kl)
    # ckl_list.append(ckl)
    # pest_list.append(pest)
    # cpest_list.append(cpest)
    # panal_list.append(pw)
    os.makedirs(f'double_corrected_data/Ndim{ndim}_N{Nobs}_posteriors/', exist_ok=True)
    with open(f'double_corrected_data/Ndim{ndim}_N{Nobs}_posteriors/npe_{npe}.pkl', 'wb') as ff:
        pkl.dump((np.array(kl), np.array(pest), np.array(pw), np.array(mu_mesh), np.array(sig_mesh)), ff)