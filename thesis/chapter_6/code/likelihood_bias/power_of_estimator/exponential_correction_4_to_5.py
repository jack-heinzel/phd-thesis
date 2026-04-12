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
from jax.interpreters import xla

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

from tqdm import tqdm

prng_key = jr.PRNGKey(0)
larger_batch = 10
num_repeat = 1_000_000_000
batch_sizes = [500000, 50000, 5000, 2000, 500] # 2, 3, 4, 5, 6
Ms = np.logspace(4,5,51, dtype=int)
MMs = Ms[:,None] * np.ones(larger_batch, dtype=int)[None,:]
# MMs = Ms
MMs = MMs.flatten()
Ms_small = Ms
Nobs = 100
all_montecarlos = None
from jax_tqdm import loop_tqdm
gpus = jax.devices('gpu')
cpus = jax.devices('cpu')

m1s = []
stds = []
@jax.jit
def moments(arr):
    return jnp.mean(arr), jnp.exp(LSE(2*jnp.log(arr)) - jnp.log(arr.size))

ms_tqdm = Ms_small
for ii, M in enumerate(ms_tqdm):
    
    m1 = 0.
    m2 = 0.
    batch_size = batch_sizes[int(np.log10(M))-2]
    
    # batch_size = 500
    @jax.jit
    def draw_mc(prng_key):
        xi = 2*jr.exponential(prng_key, shape=(M,))
        means = jnp.mean(2*jnp.exp(-xi / 2))
        variances = jnp.var(2*jnp.exp(-xi / 2)) / (M - 1)
        montecarlo = means ** (-Nobs) * jnp.exp((-Nobs*(Nobs+1)/2)*(variances/means**2))
        return montecarlo
        # return montecarlo
    
    mc_batch = jax.vmap(draw_mc)
    # for ik in range(50):
    #   montecarlo = jnp.array([])
    ranger = int(num_repeat / batch_size)
    
    @loop_tqdm(ranger, print_rate=1, tqdm_type='auto')
    def loop_fn(_, val):
        prng_key, m1, m2 = val
        prng_key, _ = jr.split(prng_key)
        prng_keys = jr.split(prng_key, batch_size)
        montecarlo, second_moment = moments(mc_batch(prng_keys))
        m1 += montecarlo 
        m2 += second_moment
        return prng_key, m1, m2

    prng_key, m1, m2 = jax.lax.fori_loop(0, ranger, loop_fn, (prng_key, m1, m2))
        
        # montecarlo = jax.device_put(montecarlo, cpus[0])
        
        # if int(ij/batch_size) % 50 == 0:
        #     # print('cleared cache')
        #     jax.clear_caches()
    # else:
    #     all_montecarlos = np.append(all_montecarlos, mcs.flatten()[:,None], axis=1)
    mean = m1 / ranger
    m2_mean = m2 / ranger
    std = np.sqrt((m2_mean - mean**2)) / np.sqrt(num_repeat - 1)
    m1s.append(mean)
    stds.append(std)
    # ms_tqdm.set_description()
    print(f'M = {M}, mean = {mean} +/- {std}, fractional error = {(1 - mean)/std}')
with open('exponential_montecarlo_means_4_to_5.pkl', 'wb') as ff:
    pkl.dump((m1s, stds), ff)
