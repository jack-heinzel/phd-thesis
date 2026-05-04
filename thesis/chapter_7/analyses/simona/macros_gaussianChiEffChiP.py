import numpy as np 
from popsummary.popresult import PopulationResult
import json
from tqdm import tqdm

"""
Define functions
"""

# Generate dict with the hyper-parameter quantiles
def get_param_dict(hyper_param_name):

    # fetch samples
    samples = result.get_hyperparameter_samples(hyperparameters=hyper_param_name)

    # get quantiles
    fifth, median, ninetyfifth = np.quantile(samples, [0.05, 0.5, 0.95])

    # get errors
    e_plus = ninetyfifth - median
    e_minus = median - fifth
    
    # format
    d =  {
        "median":np.round(median,2), 
        "5th percentile":np.round(fifth,2), 
        "95th percentile":np.round(ninetyfifth,2),
        "error plus":np.round(e_plus,2), 
        "error minus":np.round(e_minus,2) 
    }
    return d

# draw samples from a distribution y = y(x) where x is a grid
def sample_dist(x, y):
    
    random_x = np.random.uniform(min(x), max(x), size=int(1e6))
    random_y = np.interp(random_x, x, y)
    idxs = np.random.choice(range(len(random_x)), size=int(0.1 * len(random_x)), p=random_y/sum(random_y))
    
    s = random_x[idxs]
    s_y = random_y[idxs]

    return s, s_y

# Generate dict with the PPD information
def get_ppd_dict(param_name): 

    # fetch x values and p(x) values
    pos, rates = result.get_rates_on_grids(param_name)
    pos = pos.flatten()

    # calculate PPD
    ppd = np.average(rates, axis=0) 

    # draw samples from the PPD
    s, s_ppd = sample_dist(pos, ppd)

    # get quantiles
    fifth, median, ninetyfifth = np.quantile(s, [0.05, 0.5, 0.95])

    # get peak
    peak = s[np.argmax(s_ppd)]
    
    # format
    d =  {
        "median":np.round(median,2), 
        "5th percentile":np.round(fifth,2), 
        "95th percentile":np.round(ninetyfifth,2),
        "spin at peak":np.round(peak,2)
    }
    return d

"""
Generate macros file
"""

# Load in the popsummary result file
with open('popsummary_filepath_BBHSpin_GaussianChiEffChiP.txt', 'r') as file:
    popsummary_path = file.read()

result = PopulationResult(fname=popsummary_path)

# Get hyperparams dict
hyperparameters = ['mu_eff', 'sig_eff', 'mu_p', 'sig_p', 'cov', 'min_eff', 'f_neg_eff']
param_dict = {p:get_param_dict(p) for p in hyperparameters}

# Get PPDs dict
parameters = ['chi_eff', 'chi_p']
ppd_dict = {p:get_ppd_dict(p) for p in parameters}

# Combine 
macros_dict = {'param':param_dict, 'ppd':ppd_dict}

# Save
with open('gaussianChiEffChiP.json', 'w') as f:
	json.dump(macros_dict, f)
