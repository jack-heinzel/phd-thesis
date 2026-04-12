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
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'

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

    var_numerator = jnp.exp(LSE(2*obs_weights, axis=1)-jnp.log(NPE)-jnp.log(NPE-1)-2*numerator) - 1/ (NPE - 1)
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
    # worried about how this will eat memory
    # covariance_term(mu, jnp.exp(jnp.log(10)*sigma), mu, jnp.exp(jnp.log(10)*sigma), observations_array)

    if big:
        arr = tqdm(zip(mu, sigma))
        cov = np.array([[covariance_term(m[None,None], jnp.exp(jnp.log(10) * s[None,None]), 
                            mu, jnp.exp(jnp.log(10) * sigma), observations_array)[...,0,0] for m, s in zip(mm, ss)]
                            for mm, ss in arr])
    else:
        arr = zip(mu, sigma)
        cov = jnp.array([covariance_term(mu, jnp.exp(jnp.log(10) * sigma), 
                    m[:,None], jnp.exp(jnp.log(10) * s[:,None]), observations_array)[:,0,:,:] for m, s in arr
                    ])
    # print(cov)
    # cov = jnp.array([covariance_term(m[None], jnp.exp(jnp.log(10) * s[None]), 
    #                           mu, jnp.exp(jnp.log(10) * sigma), observations_array)[:,:,0,:] for m, s in tqdm(zip(mu, sigma))
    #                           ])
    # cov = jax.lax.map(
    #     lambda x: covariance_term(x[0], jnp.exp(jnp.log(10)*x[1]), x[2], jnp.exp(jnp.log(10)*x[3]), observations_array),
    #     jnp.array([mug.flatten(), sigmag.flatten(), mug.flatten(), sigmag.flatten()]).swapaxes(0,1)[...,None]
    # )
    # print(cov.shape)
    correction = jnp.sum(cov*jnp.exp(log_posterior[...,None,None]), axis=(0,1)) * dsigma * dmu
    # print(correction)
    # correction = jax.lax.map(lambda x: jnp.sum(x, axis=(-2,-1)), cov*jnp.exp(log_posterior[None,None,...]))
    corrected_posterior = log_posterior + correction
    corr_log_evidence = LSE(corrected_posterior) + jnp.log(dsigma * dmu)

    return log_posterior, log_evidence, corrected_posterior - corr_log_evidence, corr_log_evidence, vs, wneff, correction, cov

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


Nobs = [100, 1000, 5000][int(sys.argv[1])]
# Nobs = 100
print('Nobs', Nobs)
true_mean = 1
true_sd = 1
true_obs = true_sd * jr.normal(jr.PRNGKey(0), shape=(Nobs,ndim)) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
noise = 1
noise_obs = noise * jr.normal(jr.PRNGKey(1), shape=(Nobs,ndim))
obs = true_obs + noise_obs

PRNGkey = jr.PRNGKey(42+int(sys.argv[1]))

kl_list = []
ckl_list = []
pest_list = []
cpest_list = []
panal_list = []

npelist = np.logspace(1,3,60, dtype=int)
npelist = [10, 50, 200, 1000]

# print(npelist)
for npe in npelist:
    PRNGkey, _ = jr.split(PRNGkey)

    if npe >= 5:
        big = True
    else:
        big = False
    # @jax.jit

    if Nobs == 100:
        sigmas = jnp.linspace(-0.3,0.2,51)
        mus = jnp.linspace(0.5,1.5,51)
    elif Nobs == 1000:
        sigmas = jnp.linspace(-0.1,0.12,41)
        mus = jnp.linspace(0.8,1.1,41)
    elif Nobs == 5000:
        if npe < 100:
            upper = 0.1
        else:
            upper = 0.05
        sigmas = jnp.linspace(-0.05,upper,41)
        mus = jnp.linspace(0.9,1.05,41)

    dsigma = sigmas[1] - sigmas[0]
    dmu = mus[1] - mus[0]

    mu_mesh, sig_mesh = jnp.meshgrid(mus, sigmas)

    def kl_and_ptrue(PRNGkey):

        log_posterior, log_evidence, clog_posterior, clog_evidence, vs, wneff, int_cov, cov = random_posterior(PRNGkey, mu_mesh, sig_mesh, obs, npe, noise, dmu, dsigma, big=big)
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_mesh, sig_mesh, obs, noise, dmu, dsigma)

        cKL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - clog_posterior)) * dmu * dsigma
        KL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - log_posterior)) * dmu * dsigma
        # print(KL_div)
        exp_v = jnp.sum(jnp.exp(log_posterior) * vs) * dmu * dsigma
        exp_cov = jnp.sum(jnp.exp(log_posterior) * int_cov) * dmu * dsigma
        mean_sq_cov = jnp.sum(jnp.exp(log_posterior) * (int_cov - exp_cov)**2 ) * dmu * dsigma
        # exp_wneff = jnp.sum(jnp.exp(log_posterior) * wneff) * dmu * dsigma
        # p_at_truth = log_posterior[100,100]
        # analytic_p_at_truth = analytic_log_posterior[100,100]
        # print(p_at_truth, analytic_p_at_truth)
        return KL_div / jnp.log(2), cKL_div / jnp.log(2), log_posterior, clog_posterior, analytic_log_posterior, (exp_v - exp_cov) / 2 / jnp.log(2), mean_sq_cov / 2 / jnp.log(2), cov, vs

    n_repeat = 100
    keys = jr.split(PRNGkey, n_repeat)
    # kl, p_est, p_anal = jax.lax.map(kl_and_ptrue, keys, batch_size=1)
    # eh, I'd rather have a tqdm progress bar lol
    kl, ckl, pest, cpest, apest = [], [], [], [], []
    precision_stats, accuracy_stats = [], []
    c_weights, v_weights = [], []
    klist = tqdm(keys)
    klist.set_description(f'N_PE = {npe}')
    for key in klist:
        k, e, p, pv, pw, prec, accu, cov, vs = kl_and_ptrue(key)
        kl.append(k)
        ckl.append(e)
        pest.append(p)
        cpest.append(pv)
        apest.append(pw)
        precision_stats.append(prec)
        accuracy_stats.append(accu)
        c_weights.append(cov)
        v_weights.append(vs)
        
    # kl_list.append(kl)
    # ckl_list.append(ckl)
    # pest_list.append(pest)
    # cpest_list.append(cpest)
    # panal_list.append(pw)
    os.makedirs(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_posteriors/', exist_ok=True)
    os.makedirs(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_weights/', exist_ok=True)
    
    with open(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_posteriors/{int(sys.argv[1])}_npe_{npe}.pkl', 'wb') as ff:
        pkl.dump((np.array(kl), np.array(ckl), np.array(pest), np.array(cpest), np.array(pw), np.array(mu_mesh), np.array(sig_mesh), np.array(precision_stats), np.array(accuracy_stats)), ff)
    with open(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_weights/{int(sys.argv[1])}_npe_{npe}.pkl', 'wb') as ff:
        pkl.dump((np.array(c_weights), np.array(v_weights), np.array(mu_mesh), np.array(sig_mesh), np.array(pest)), ff)
    