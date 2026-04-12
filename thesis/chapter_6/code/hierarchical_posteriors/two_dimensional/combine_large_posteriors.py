import numpy as np
import pickle as pkl
import os
from glob import glob

ndim = 1
Nobs = 100
true_mean = 1
true_sd = 1
noise = 2 
npes = [20_000]
for npe in npes:
    none = True
    if noise == 1:
        files = glob(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_posteriors/*_npe_{npe}.pkl')
    elif noise == 2:
        files = glob(f'../../../data/two_dimensional/corrected_data/large_noise_Ndim{ndim}_N{Nobs}_posteriors/*_npe_{npe}.pkl')
    for filename in files:
        with open(filename, 'rb') as ff:
            kl, ckl, log_p, log_cp, analytical_log_posterior, mu, sig, precision_stats, accuracy_stats = pkl.load(ff)
        if none:
            total_kl = kl
            total_ckl = ckl
            total_log_p = log_p
            total_log_cp = log_cp
            total_precision_stats = precision_stats
            total_accuracy_stats = accuracy_stats
            none = False
        else:
            total_kl = np.append(total_kl, kl, axis=0)
            total_ckl = np.append(total_ckl, ckl, axis=0)
            total_log_p = np.append(total_log_p, log_p, axis=0)
            total_log_cp = np.append(total_log_cp, log_cp, axis=0)
            total_precision_stats = np.append(total_precision_stats, precision_stats, axis=0)
            total_accuracy_stats = np.append(total_accuracy_stats, accuracy_stats, axis=0)
    if noise == 1:
        with open(f'../../../data/two_dimensional/corrected_data/Ndim{ndim}_N{Nobs}_posteriors/npe_{npe}.pkl', 'wb') as ff:
            pkl.dump((total_kl, total_ckl, total_log_p, total_log_cp, analytical_log_posterior, mu, sig, total_precision_stats, total_accuracy_stats), ff)
    elif noise == 2:
        with open(f'../../../data/two_dimensional/corrected_data/large_noise_Ndim{ndim}_N{Nobs}_posteriors/npe_{npe}.pkl', 'wb') as ff:
            pkl.dump((total_kl, total_ckl, total_log_p, total_log_cp, analytical_log_posterior, mu, sig, total_precision_stats, total_accuracy_stats), ff)

    
if noise == 2:
    for npe in npes:
        files = glob(f'../../../data/two_dimensional/corrected_data/large_noise_Ndim{ndim}_N{Nobs}_weights/*_npe_{npe}.pkl')
        none = True
        for filename in files:
            with open(filename, 'rb') as ff:
                c_weights, v_weights, mu_mesh, sig_mesh, pest = pkl.load(ff)
            if none:
                total_c_weights = c_weights
                total_v_weights = v_weights
                total_pest = pest  
                none = False
            else:
                total_c_weights = np.append(total_c_weights, c_weights, axis=0)
                total_v_weights = np.append(total_v_weights, v_weights, axis=0)
                total_pest = np.append(total_pest, pest, axis=0)
    with open(f'../../../data/two_dimensional/corrected_data/large_noise_Ndim{ndim}_N{Nobs}_weights/npe_{npe}.pkl', 'wb') as ff:
        pkl.dump((total_c_weights, total_v_weights, mu_mesh, sig_mesh, total_pest), ff)

