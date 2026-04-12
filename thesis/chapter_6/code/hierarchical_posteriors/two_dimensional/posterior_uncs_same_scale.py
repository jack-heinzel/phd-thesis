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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

from tqdm import tqdm

# generate injections from population, without selection effects
# implement the p value test of https://arxiv.org/pdf/2304.06138
ndim = 1
print('ndim = ', ndim)
def calculate_contour(log_p_array, contour=0.9):
    ln_p_sort = -jnp.sort(-log_p_array) # sort from big to small
    cdf = jnp.cumsum(jnp.exp(ln_p_sort))
    cdf /= cdf[-1] # normalize

    p_boundary = jnp.interp(contour, cdf, log_p_array) # pdf boundary between inner and outer contour
    return p_boundary

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

def random_posterior(PRNGkey, mu, sigma, observations, npe, noise_sigma, dmu, dsigma, big=False):

    observations_array = draw_PE_sample(PRNGkey, npe, observations, noise_sigma)

    # lls, vs, wneff = [], [], []
    # for m, s in zip(mu, sigma): # can I jax.lax.map this...
    #     a, b, c = naive_log_likelihood_estimator(m, jnp.exp(jnp.log(10) * s), observations_array)
    #     lls.append(a), vs.append(b), wneff.append(c)
    # lls = jnp.array(lls); vs = jnp.array(vs); wneff = jnp.array(wneff)
    # lls, vs, wneff = jax.tree.map(
    #     lambda x, y: naive_log_likelihood_estimator(x, jnp.exp(jnp.log(10) * y), observations_array),
    #     mu, 
    #     sigma
    # )
    if big:
        lls, vs, wneff = jax.lax.map(
            lambda x: naive_log_likelihood_estimator(x[0], jnp.exp(jnp.log(10) * x[1]), observations_array),
            jnp.array([mu, sigma]).swapaxes(0,1)
        )
    else:
        lls, vs, wneff = naive_log_likelihood_estimator(mu, jnp.exp(jnp.log(10) * sigma), observations_array)
    log_evidence = LSE(lls) + jnp.log(dsigma * dmu)
    log_posterior = lls - log_evidence

    return log_posterior, log_evidence, vs, wneff

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

Nobs = 400
true_mean = 1
true_sd = 1
true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,ndim)) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
noise = 1
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,ndim))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42)

kl_list = []
pest_list = []
panal_list = []
exp_vlist = []
exp_wlist = []

npelist = np.logspace(1,4,101,dtype=int)

for npe in npelist:
    PRNGkey, _ = jr.split(PRNGkey)

    if npe >= 500:
        big = True
    else:
        big = False
    @jax.jit
    def kl_and_ptrue(PRNGkey):

        sigmas = jnp.linspace(-1,1,201)
        mus = jnp.linspace(0,2,201)

        dsigma = sigmas[1] - sigmas[0]
        dmu = mus[1] - mus[0]

        mu_mesh, sig_mesh = jnp.meshgrid(mus, sigmas)

        log_posterior, log_evidence, vs, wneff = random_posterior(PRNGkey, mu_mesh, sig_mesh, obs, npe, noise, dmu, dsigma, big=big)
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_mesh, sig_mesh, obs, noise, dmu, dsigma)

        KL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - log_posterior)) * dmu * dsigma
        # print(KL_div)
        exp_v = jnp.sum(jnp.exp(log_posterior) * jnp.sqrt(vs)) * dmu * dsigma
        exp_wneff = jnp.sum(jnp.exp(log_posterior) * wneff) * dmu * dsigma
        p_at_truth = log_posterior[100,100]
        analytic_p_at_truth = analytic_log_posterior[100,100]
        # print(p_at_truth, analytic_p_at_truth)
        return KL_div / jnp.log(2), p_at_truth, analytic_p_at_truth, exp_v, exp_wneff

    n_repeat = 1000
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

with open(f'corrected_data/Ndim{ndim}_N{Nobs}_noise_equal_population_1_to_4_divergences_dense.pkl', 'wb') as ff:
    pkl.dump((np.array(kl_list), np.array(pest_list), np.array(panal_list), np.array(exp_vlist), np.array(exp_wlist), npelist), ff)