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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

from tqdm import tqdm

# generate injections from population, without selection effects


def calculate_contour(log_p_array, contour=0.9):
    ln_p_sort = -jnp.sort(-log_p_array) # sort from big to small
    cdf = jnp.cumsum(jnp.exp(ln_p_sort))
    cdf /= cdf[-1] # normalize

    p_boundary = jnp.interp(contour, cdf, log_p_array) # pdf boundary between inner and outer contour
    return p_boundary

def draw_PE_sample(PRNGkey, NPE, observations, noise_sigma):
    scatter = jax.random.normal(PRNGkey, shape=(len(observations), NPE)) * noise_sigma
    return jnp.expand_dims(observations, axis=1) + scatter

def log_gaussian(x, mu, sigma):
    return -(x - mu)**2 / 2 / sigma**2 - 0.5*jnp.log(2*jnp.pi) - jnp.log(sigma)

@jax.jit
def naive_log_likelihood_estimator(mu, sigma, observations_array):
    # expand dims to right shapes
    
    num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    observations_array = jnp.expand_dims(observations_array, axis=tuple([2+ii for ii in range(num_dimension)]))
    
    obs_weights = log_gaussian(observations_array, mu[None,None,...], sigma[None,None,...])
    NPE = obs_weights.shape[1]

    numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    # print(numerator, denominator)

    var_numerator = jnp.exp(LSE(2*obs_weights, axis=1) - 2*jnp.log(NPE) - 2*numerator) - 1/NPE
    num_variance = jnp.sum(var_numerator, axis=0)
    neffs = jnp.exp(2*LSE(obs_weights, axis=1) - LSE(2*obs_weights, axis=1))
    worst_neff = jnp.min(neffs, axis=0)

    return jnp.sum(numerator, axis=0), num_variance, worst_neff

def random_posterior(PRNGkey, mu, sigma, observations, npe, noise_sigma, dmu, dsigma):

    observations_array = draw_PE_sample(PRNGkey, npe, observations, noise_sigma)

    lls, vs, wneff = naive_log_likelihood_estimator(mu, jnp.exp(jnp.log(10) * sigma), observations_array)

    log_evidence = LSE(lls) + jnp.log(dsigma * dmu)
    log_posterior = lls - log_evidence

    return log_posterior, log_evidence, vs, wneff

def analytical_posterior(mu, sigma_log, observations_centers, noise_sigma, dmu, dsigma):
    Nobs = len(observations_centers)
    sigma = jnp.exp(jnp.log(10) * sigma_log)
    log_l_norm = -(Nobs / 2) * jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2))
    expo = -jnp.sum((jnp.expand_dims(observations_centers, tuple(np.arange(1,len(mu.shape)+1))) - mu[None,...])**2, axis=0) / 2 / (sigma**2 + noise_sigma**2)
    ll = expo + log_l_norm
    log_evidence = LSE(ll) + jnp.log(dsigma * dmu)
    log_posterior = ll - log_evidence
    return log_posterior, log_evidence


Nobs = 100
true_mean = 1
true_sd = 0.1
true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,)) + true_mean
noise = 1
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42)

kl_list = []
pest_list = []
panal_list = []
exp_vlist = []
exp_wlist = []

npelist = np.logspace(2,5,51,dtype=int)

for npe in npelist:
    PRNGkey, _ = jr.split(PRNGkey)

    @jax.jit
    def kl_and_ptrue(PRNGkey):

        sigmas = jnp.linspace(-2,0,201)
        mus = jnp.linspace(0,2,201)

        dsigma = sigmas[1] - sigmas[0]
        dmu = mus[1] - mus[0]

        mu_mesh, sig_mesh = jnp.meshgrid(mus, sigmas)

        log_posterior, log_evidence, vs, wneff = random_posterior(PRNGkey, mu_mesh, sig_mesh, obs, npe, noise, dmu, dsigma)
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_mesh, sig_mesh, obs, noise, dmu, dsigma)

        KL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - log_posterior)) * dmu * dsigma
        # print(KL_div)
        exp_v = jnp.sum(jnp.exp(log_posterior) * jnp.sqrt(vs)) * dmu * dsigma
        exp_wneff = jnp.sum(jnp.exp(log_posterior) * wneff) * dmu * dsigma
        p_at_truth = log_posterior[100,100]
        analytic_p_at_truth = analytic_log_posterior[100,100]
        # print(p_at_truth, analytic_p_at_truth)
        return KL_div / jnp.log(2), p_at_truth, analytic_p_at_truth, exp_v, exp_wneff

    n_repeat = 100
    keys = jr.split(PRNGkey, n_repeat)
    # kl, p_est, p_anal = jax.lax.map(kl_and_ptrue, keys, batch_size=1)
    # eh, I'd rather have a tqdm progress bar lol
    kl, p_est, p_anal, exp_v, exp_w = [], [], [], [], []
    klist = tqdm(keys)
    klist.set_description(f'N_PE = {npe}')
    for key in klist:
        k, e, p, pv, pw = kl_and_ptrue(key)
        kl.append(k)
        p_est.append(e)
        p_anal.append(p)
        exp_v.append(pv)
        exp_w.append(pw)

    kl_list.append(kl)
    pest_list.append(p_est)
    panal_list.append(p_anal)
    exp_vlist.append(exp_v)
    exp_wlist.append(exp_w)

with open(f'data/N{Nobs}_noise_smaller_population_2_to_5_divergences_dense.pkl', 'wb') as ff:
    pkl.dump((np.array(kl_list), np.array(pest_list), np.array(panal_list), np.array(exp_vlist), np.array(exp_wlist), npelist), ff)
    