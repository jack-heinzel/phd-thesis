import numpy as np
import scipy
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import logsumexp as LSE
from jax.scipy.special import erfc

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

# generate injections from population, with selection effects

ndim = 1
VT_selection = 3
print('ndim = ', ndim, 'vt_selection = ', VT_selection)
def draw_PE_sample(PRNGkey, NPE, observations, noise_sigma):
    scatter = jax.random.normal(PRNGkey, shape=(len(observations), NPE)) * noise_sigma
    r = jnp.expand_dims(observations, axis=2) + scatter
    # print(r.shape)
    return r

def draw_VT_sample(PRNGkey, NVT, noise_sigma, VT_base_mu=1, VT_base_sigma=1.5, selection=VT_selection):
    PRNGkey1, PRNGkey2 = jr.split(PRNGkey)
    scatter = jax.random.normal(PRNGkey1, shape=(NVT,)) * noise_sigma
    true = jax.random.normal(PRNGkey2, shape=(NVT,)) * VT_base_sigma + VT_base_mu
    
    keep = true + scatter > selection
    keepers = true[keep]

    log_p_vt = log_gaussian(keepers, VT_base_mu, VT_base_sigma)
    return keepers, log_p_vt

def log_gaussian(x, mu, sigma):
    p = -(x - mu)**2 / 2 / sigma**2 - 0.5*jnp.log(2*jnp.pi) - jnp.log(sigma)
    # print(p.shape)
    return p

@jax.jit
def naive_log_likelihood_estimator(mu, sigma, observations_centers, noise_sigma, VT_detected_sample, log_p_VT, NVT):
    '''
    check how this works for arbitrary ndim
    '''
    Nobs = len(observations_centers)
    log_l_norm = -(Nobs / 2) * jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2))
    expo = -jnp.sum((jnp.expand_dims(observations_centers, tuple(np.arange(1,len(mu.shape)+1))) - mu[None,...])**2, axis=0) / 2 / (sigma**2 + noise_sigma**2)
    log_numerator = expo + log_l_norm
    # numerator for Log Likelihood
    vt_weights = log_gaussian(VT_detected_sample[...,None,None], mu[None,...], sigma[None,...]) - log_p_VT[:,None,None]
    log_selection = LSE(vt_weights, axis=0) - jnp.log(NVT)
    
    log_selection_variance = jnp.expm1(LSE(2*vt_weights, axis=0) - jnp.log(NVT) - 2*log_selection)
    denom_variance = Nobs**2 * log_selection_variance / (NVT - 1)
    return log_numerator - Nobs * log_selection, denom_variance, (NVT - 1) / log_selection_variance

    # num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    # observations_array = jnp.expand_dims(observations_array, axis=tuple([3+ii for ii in range(num_dimension)]))
    
    # obs_weights = log_gaussian(observations_array, mu[None,None,None,...], sigma[None,None,...])
    # NPE = obs_weights.shape[1]

    # # runs out of memory, split up computation over mu and sigma subs...
    # numerator = LSE(obs_weights, axis=1) - jnp.log(NPE)
    # # print(numerator, denominator)

    # var_numerator = jnp.exp(LSE(2*obs_weights, axis=1)-jnp.log(NPE)-jnp.log(NPE-1)-2*numerator) - 1/ (NPE - 1)
    # num_variance = jnp.sum(var_numerator, axis=0)
    # neffs = jnp.exp(2*LSE(obs_weights, axis=1) - LSE(2*obs_weights, axis=1))
    # worst_neff = jnp.min(neffs, axis=0)

    return jnp.sum(numerator, axis=0), num_variance, worst_neff

@jax.jit
def covariance_term(mu, sigma, mup, sigmap, VT_detected_sample, log_p_VT, NVT, Nobs):
    # expand dims to right shapes
    
    num_dimension = len(jnp.shape(mu))    # assume shapes are same for mu and sigma
    # VT_detected_sample.shape = (NVT)
    VT_detected_sample = jnp.expand_dims(VT_detected_sample, axis=tuple([1+ii for ii in range(num_dimension)]))
    # (NVT, 1, 1)
    
    vt_weights = log_gaussian(VT_detected_sample, mu[None,...], sigma[None,...]) - log_p_VT[:,None,None]
    vt_weights_p = log_gaussian(VT_detected_sample, mup[None,...], sigmap[None,...]) - log_p_VT[:,None,None] # shape (NVT, mu1, mu2)

    vt_weights = vt_weights[:,None,None,:,:]
    vt_weights_p = vt_weights_p[...,None,None]

    selection = LSE(vt_weights, axis=0) - jnp.log(NVT)
    selection_p = LSE(vt_weights_p, axis=0) - jnp.log(NVT) 
    
    cov_selection = jnp.exp(LSE(vt_weights+vt_weights_p, axis=0)-selection-selection_p-jnp.log(NVT)-jnp.log(NVT-1)) - 1 / (NVT-1)
    # print(numerator, denominator)
    
    return cov_selection * Nobs**2

def random_posterior(PRNGkey, mu, sigma, observations, nvt, noise_sigma, dmu, dsigma, big=False, really_big=False):

    Nobs = observations.shape[0]
    VT_detected_sample, log_p_VT = draw_VT_sample(PRNGkey, nvt, noise_sigma)
    # print(VT_detected_sample.shape)
    if big:
        lls, vs, neff = jax.lax.map(
            lambda x: naive_log_likelihood_estimator(x[0], jnp.exp(jnp.log(10) * x[1]), observations, noise_sigma, VT_detected_sample, log_p_VT, nvt),
            jnp.array([mu, sigma]).swapaxes(0,1)
        )
        lls = lls[:,0,:]
        vs = vs[:,0,:]
        neff = neff[:,0,:]
    else:
        lls, vs, neff = naive_log_likelihood_estimator(mu, jnp.exp(jnp.log(10) * sigma), observations, noise_sigma, VT_detected_sample, log_p_VT, nvt)
    # lls, vs, wneff = naive_log_likelihood_estimator(mu, jnp.exp(jnp.log(10) * sigma), observations, noise_sigma, VT_detected_sample, log_p_VT, nvt)
    likelihood_correction = -vs * (1 + 1 / Nobs) / 2
    # print(lls.shape)
    log_evidence = LSE(lls) + jnp.log(dsigma * dmu)
    log_posterior = lls - log_evidence

    log_likecor_evidence = LSE(lls+likelihood_correction) + jnp.log(dsigma * dmu)
    log_likecor_posterior = lls+likelihood_correction - log_likecor_evidence
    
    # worried about how this will eat memory
    # cov = covariance_term(mu, jnp.exp(jnp.log(10)*sigma), mu, jnp.exp(jnp.log(10)*sigma), VT_detected_sample, log_p_VT, nvt, Nobs)

    if really_big:
        arr = tqdm(zip(mu, sigma))
        cov = np.array([[[covariance_term(m[None,None], jnp.exp(jnp.log(10) * s[None,None]), 
                            mu_p_bit, jnp.exp(jnp.log(10) * sigma_p_bit), VT_detected_sample, log_p_VT, nvt, Nobs)[0,:,0,0] 
                            for mu_p_bit, sigma_p_bit in zip(mu, sigma)]
                            for m, s in zip(mm, ss)]
                            for mm, ss in arr])
    
    elif big:
        arr = tqdm(zip(mu, sigma))
        cov = np.array([[covariance_term(m[None,None], jnp.exp(jnp.log(10) * s[None,None]), 
                            mu, jnp.exp(jnp.log(10) * sigma), VT_detected_sample, log_p_VT, nvt, Nobs)[...,0,0] for m, s in zip(mm, ss)]
                            for mm, ss in arr])
    
    else:
        arr = zip(mu, sigma)
        cov = np.array([covariance_term(mu, jnp.exp(jnp.log(10) * sigma), 
                    m[:,None], jnp.exp(jnp.log(10) * s[:,None]), VT_detected_sample, log_p_VT, nvt, Nobs)[:,0,:,:] for m, s in arr
                    ])
        
    # print(cov.shape, log_posterior.shape)
    correction = jnp.sum(cov*jnp.exp(log_likecor_posterior[...,None,None]), axis=(0,1)) * dsigma * dmu
    
    corrected_posterior = log_likecor_posterior + correction
    corr_log_evidence = LSE(corrected_posterior) + jnp.log(dsigma * dmu)

    return log_posterior, log_evidence, log_likecor_posterior, corrected_posterior - corr_log_evidence, corr_log_evidence, vs, neff, correction, cov

def analytical_posterior(mu, sigma_log, observations_centers, noise_sigma, dmu, dsigma, selection=VT_selection):
    '''
    check how this works for arbitrary ndim
    '''
    Nobs = len(observations_centers)
    sigma = jnp.exp(jnp.log(10) * sigma_log)
    log_l_norm = -(Nobs / 2) * jnp.log(2*jnp.pi*(sigma**2 + noise_sigma**2))
    expo = -jnp.sum((jnp.expand_dims(observations_centers, tuple(np.arange(1,len(mu.shape)+1))) - mu[None,...])**2, axis=0) / 2 / (sigma**2 + noise_sigma**2)
    selection_term = Nobs * jnp.log(erfc((selection-mu) / jnp.sqrt(2*(jnp.exp(2*jnp.log(10)*sigma_log)+noise_sigma**2))) / 2)

    ll = expo + log_l_norm - selection_term

    log_evidence = LSE(ll) + jnp.log(dsigma * dmu)
    log_posterior = ll - log_evidence
    return log_posterior, log_evidence

Nobs = 5000
true_mean = 1
true_sd = 1
noise = 1

nk = jr.PRNGKey(0)
k = jr.PRNGKey(1)

true_obs = []
obs = []

print('drawing samples')
while len(true_obs) < Nobs:
    test_obs = true_sd * jr.normal(k) + true_mean # model is a Gaussian with mean = mu (1, 1, 1, ..., ndim), and same for sigma = sig (1, ..., ndim)
    noise_obs = noise * jr.normal(nk)
    k, _ = jr.split(k)
    nk, _ = jr.split(nk)
    if test_obs + noise_obs > VT_selection:
        true_obs.append(test_obs)
        obs.append(test_obs + noise_obs)

true_obs = jnp.array(true_obs)
obs = jnp.array(obs)
# print(obs)

n_repeat = 20
if n_repeat < 100:
    PRNGkey = jr.PRNGKey(42+int(sys.argv[1]))
else:
    PRNGkey = jr.PRNGKey(42)

nvtlist = np.logspace(5,7,3, dtype=int)
nvtlist = [100_000_000]
# nvtlist = [50_000_000, 5_000_000]
print(nvtlist)
for nvt in nvtlist:
    PRNGkey, _ = jr.split(PRNGkey)

    if nvt >= 1e6:
        big = True
    else:
        big = False

    if nvt >= 5e6:
        really_big = True
    else:
        really_big = False

    # if Nobs == 1000:
    sigmas = jnp.linspace(-0.3,0.2,51)
    mus = jnp.linspace(0,2,51)

    sigma_analytic = jnp.linspace(-0.3,0.2,301)
    mu_analytic = jnp.linspace(0,2,301)
    # elif Nobs == 100:
    #     sigmas = jnp.linspace(-1,1,51)
    #     mus = jnp.linspace(-1,3,51)

    dsigma = sigmas[1] - sigmas[0]
    dmu = mus[1] - mus[0]
    dsigma_analytic = sigma_analytic[1] - sigma_analytic[0]
    dmu_analytic = mu_analytic[1] - mu_analytic[0]

    mu_mesh, sig_mesh = jnp.meshgrid(mus, sigmas)
    mu_analytic, sigma_analytic = jnp.meshgrid(mu_analytic, sigma_analytic)
    def kl_and_ptrue(PRNGkey):

        log_posterior, log_evidence, log_likecor_posterior, clog_posterior, clog_evidence, vs, neff, int_cov, cov = random_posterior(PRNGkey, mu_mesh, sig_mesh, obs, nvt, noise, dmu, dsigma, big=big, really_big=really_big)
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_mesh, sig_mesh, obs, noise, dmu, dsigma)

        cKL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - clog_posterior)) * dmu * dsigma
        lcKL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - log_likecor_posterior)) * dmu * dsigma
        KL_div = jnp.sum(jnp.exp(analytic_log_posterior) * (analytic_log_posterior - log_posterior)) * dmu * dsigma
        # print(KL_div)
        int_cov = jnp.sum(cov*jnp.exp(log_posterior[...,None,None]), axis=(0,1)) * dsigma * dmu
        exp_v = jnp.sum(jnp.exp(log_posterior) * vs) * dmu * dsigma
        exp_cov = jnp.sum(jnp.exp(log_posterior) * int_cov) * dmu * dsigma
        mean_sq_cov = jnp.sum(jnp.exp(log_posterior) * (int_cov - exp_cov)**2 ) * dmu * dsigma

        int_cov = jnp.sum(cov*jnp.exp(log_likecor_posterior[...,None,None]), axis=(0,1)) * dsigma * dmu
        lc_exp_v = jnp.sum(jnp.exp(log_likecor_posterior) * vs) * dmu * dsigma
        lc_exp_cov = jnp.sum(jnp.exp(log_likecor_posterior) * int_cov) * dmu * dsigma
        lc_mean_sq_cov = jnp.sum(jnp.exp(log_likecor_posterior) * (int_cov - lc_exp_cov)**2 ) * dmu * dsigma
        
        analytic_log_posterior, analytic_log_evidence = analytical_posterior(mu_analytic, sigma_analytic, obs, noise, dmu_analytic, dsigma_analytic)

        return KL_div / jnp.log(2), lcKL_div / jnp.log(2), cKL_div / jnp.log(2), log_posterior, log_likecor_posterior, clog_posterior, analytic_log_posterior, (exp_v - exp_cov) / 2 / jnp.log(2), mean_sq_cov / 2 / jnp.log(2), (lc_exp_v - lc_exp_cov) / 2 / jnp.log(2), lc_mean_sq_cov / 2 / jnp.log(2), cov, vs, neff

    keys = jr.split(PRNGkey, n_repeat)
    # kl, p_est, p_anal = jax.lax.map(kl_and_ptrue, keys, batch_size=1)
    # eh, I'd rather have a tqdm progress bar lol
    kl, lckls, ckl, pest, lcpest, cpest, apest = [], [], [], [], [], [], []
    precision_stats, accuracy_stats = [], []
    lc_precision_stats, lc_accuracy_stats = [], []
    covs, vs, neffs = [], [], [] # ADD STUFF FOR THIS

    klist = tqdm(keys)
    klist.set_description(f'N_VT = {nvt}')
    for key in klist:
        k, lck, e, p, plc, pv, pw, prec, accu, lc_prec, lc_accu, c, v, neff = kl_and_ptrue(key)
        kl.append(k)
        lckls.append(lck)
        ckl.append(e)
        pest.append(p)
        lcpest.append(plc)
        cpest.append(pv)
        apest.append(pw)
        precision_stats.append(prec)
        accuracy_stats.append(accu)
        lc_precision_stats.append(lc_prec)
        lc_accuracy_stats.append(lc_accu)

        covs.append(c)
        vs.append(v)
        neffs.append(neff)
        
    # kl_list.append(kl)
    # ckl_list.append(ckl)
    # pest_list.append(pest)
    # cpest_list.append(cpest)
    # panal_list.append(pw)
    os.makedirs(f'../../../data/two_dimensional/corrected_data/selection_{VT_selection}_Ndim{ndim}_N{Nobs}_posteriors/', exist_ok=True)
    os.makedirs(f'../../../data/two_dimensional/corrected_data/selection_{VT_selection}_Ndim{ndim}_N{Nobs}_weights/', exist_ok=True)
    
    if n_repeat < 100 and len(sys.argv) > 1:
        name = f'{sys.argv[1]}_nvt_{nvt}.pkl'
    else:
        name = f'nvt_{nvt}.pkl'
    with open(f'../../../data/two_dimensional/corrected_data/selection_{VT_selection}_Ndim{ndim}_N{Nobs}_posteriors/' + name, 'wb') as ff:
        pkl.dump((np.array(kl), np.array(lckls), np.array(ckl), np.array(pest), np.array(lcpest), np.array(cpest), np.array(pw), np.array(mu_analytic), np.array(sigma_analytic), np.array(mu_mesh), np.array(sig_mesh), np.array(precision_stats), np.array(accuracy_stats), np.array(lc_precision_stats), np.array(lc_accuracy_stats)), ff)

    with open(f'../../../data/two_dimensional/corrected_data/selection_{VT_selection}_Ndim{ndim}_N{Nobs}_weights/' + name, 'wb') as ff:
        pkl.dump((np.array(covs), np.array(vs), np.array(neffs), np.array(mu_mesh), np.array(sig_mesh), np.array(pest)), ff)
    