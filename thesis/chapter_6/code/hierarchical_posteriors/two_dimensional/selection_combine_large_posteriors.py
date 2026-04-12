import numpy as np
import pickle as pkl
import os
from glob import glob

ndim = 1
Nobs = 5000
true_mean = 1
true_sd = 1
selection = 3
nvts = [100_000_000]
for nvt in nvts:
    none = True
    files = glob(f'../../../data/two_dimensional/corrected_data/selection_{selection}_Ndim{ndim}_N{Nobs}_posteriors/*_nvt_{nvt}.pkl')
    if len(files) == 0:
        print('No files found with glob')
        quit()
    for filename in files:
        with open(filename, 'rb') as ff:
            x = pkl.load(ff)
            if len(x) == 15:
                kl, lckl, ckl, log_p, log_lcp, log_cp, analytical_log_posterior, mu_analytic, sig_analytic, mu, sig, precision_stats, accuracy_stats, lc_precision_stats, lc_accuracy_stats = x
                long = True
            elif len(x) == 13:
                kl, lckl, ckl, log_p, log_lcp, log_cp, analytical_log_posterior, mu, sig, precision_stats, accuracy_stats, lc_precision_stats, lc_accuracy_stats = x
                long = False
                mu_analytic = mu
                sig_analytic = sig
        if none:
            total_kl = kl
            total_lckl = lckl
            total_ckl = ckl
            total_log_p = log_p
            total_log_lcp = log_lcp
            total_log_cp = log_cp
            total_precision_stats = precision_stats
            total_accuracy_stats = accuracy_stats
            total_lc_precision_stats = lc_precision_stats
            total_lc_accuracy_stats = lc_accuracy_stats
            # analytical_log_posterior
            # mu_analytic, sig_analytic
            # mu, sig, 
            none = False
        else:
            total_kl = np.append(total_kl, kl, axis=0)
            total_lckl = np.append(total_lckl, lckl, axis=0)
            total_ckl = np.append(total_ckl, ckl, axis=0)
            total_log_p = np.append(total_log_p, log_p, axis=0)
            total_log_lcp = np.append(total_log_lcp, log_lcp, axis=0)
            total_log_cp = np.append(total_log_cp, log_cp, axis=0)
            total_precision_stats = np.append(total_precision_stats, precision_stats, axis=0)
            total_accuracy_stats = np.append(total_accuracy_stats, accuracy_stats, axis=0)
            total_lc_precision_stats = np.append(total_lc_precision_stats, lc_precision_stats, axis=0)
            total_lc_accuracy_stats = np.append(total_lc_accuracy_stats, lc_accuracy_stats, axis=0)
    with open(f'../../../data/two_dimensional/corrected_data/selection_{selection}_Ndim{ndim}_N{Nobs}_posteriors/nvt_{nvt}.pkl', 'wb') as ff:
        if long:
            t = (
                total_kl, total_lckl, total_ckl, total_log_p, total_log_lcp, total_log_cp, 
                analytical_log_posterior, mu_analytic, sig_analytic, mu, sig, 
                total_precision_stats, total_accuracy_stats, total_lc_precision_stats, 
                total_lc_accuracy_stats
                ) 
            pkl.dump(t, ff)
        else:
            t = (
                total_kl, total_lckl, total_ckl, total_log_p, total_log_lcp, total_log_cp, 
                analytical_log_posterior, mu, sig, total_precision_stats, total_accuracy_stats, 
                total_lc_precision_stats, total_lc_accuracy_stats
                ) 
            pkl.dump(t, ff)
            
    
for nvt in nvts:
    files = glob(f'../../../data/two_dimensional/corrected_data/selection_{selection}_Ndim{ndim}_N{Nobs}_weights/*_nvt_{nvt}.pkl')
    none = True
    for filename in files:
        with open(filename, 'rb') as ff:
            x = pkl.load(ff)
            if len(x) == 5:
                c_weights, v_weights, mu_mesh, sig_mesh, pest = x
                neffs = Nobs**2 / v_weights
            elif len(x) == 6:
                c_weights, v_weights, neffs, mu_mesh, sig_mesh, pest = x
        if none:
            total_c_weights = c_weights
            total_v_weights = v_weights
            total_neffs = neffs
            total_pest = pest  
            none = False
        else:
            total_c_weights = np.append(total_c_weights, c_weights, axis=0)
            total_v_weights = np.append(total_v_weights, v_weights, axis=0)
            total_neffs = np.append(total_neffs, neffs, axis=0)
            total_pest = np.append(total_pest, pest, axis=0)
with open(f'../../../data/two_dimensional/corrected_data/selection_{selection}_Ndim{ndim}_N{Nobs}_weights/nvt_{nvt}.pkl', 'wb') as ff:
    pkl.dump((total_c_weights, total_v_weights, total_neffs, mu_mesh, sig_mesh, total_pest), ff)

