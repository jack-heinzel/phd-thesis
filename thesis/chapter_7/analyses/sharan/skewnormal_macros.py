import numpy as np
from popsummary.popresult import PopulationResult
import json
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
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
    ppd = np.average(rates, axis=1)

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


def get_distribution_properties(param_name):
    # fetch x values and p(x) values
    pos, rates = result.get_rates_on_grids(param_name)
    pos = pos.flatten()


    index = pos<=0

    p_neg = np.trapz(rates[index, :], pos[index], axis=0)
    fifth_neg, median_neg, ninetyfifth_neg = np.quantile(p_neg, [0.05, 0.5, 0.95])
    e_plus_neg = ninetyfifth_neg - median_neg
    e_minus_neg = median_neg - fifth_neg


    p_min = []
    cumulative = cumulative_trapezoid(rates, pos, axis=0)
    for ii in range(rates.shape[1]):
        p_min.append(np.interp(0.01, cumulative[:, ii], pos[1:]))

    fifth_min, median_min, ninetyfifth_min = np.quantile(p_min, [0.05, 0.5, 0.95])
    e_plus_min = ninetyfifth_min - median_min
    e_minus_min = median_min - fifth_min

    # format
    f_neg_chi =  {
        "median":np.round(median_neg,2),
        "5th percentile":np.round(fifth_neg,2),
        "95th percentile":np.round(ninetyfifth_neg,2),
        "error plus":np.round(e_plus_neg,2), 
        "error minus":np.round(e_minus_neg,2)
    }

    min_chi =  {
        "median":np.round(median_min,2),
        "5th percentile":np.round(fifth_min,2),
        "95th percentile":np.round(ninetyfifth_min,2),
        "error plus":np.round(e_plus_min,2), 
        "error minus":np.round(e_minus_min,2)
    }

    return min_chi, f_neg_chi


# Load in the popsummary result file
with open('popsummary_filepath_BBHSpin_EpsSkewNormalChiEff.txt', 'r') as file:
    popsummary_path = file.read()

result = PopulationResult(fname=popsummary_path)

# Get hyperparams dict
hyperparameters = ['mu_chi_eff', 'sigma_chi_eff', 'mu_chi_p', 'sigma_chi_p', 'eps_chi_eff']
param_dict = {p:get_param_dict(p) for p in hyperparameters}

# Get PPDs dict
labels = {"chi_eff":"Effective inspiral spin",
               "chi_p":"Effective precessing spin",}
ppd_dict = {key:get_ppd_dict(labels[key]) for key in labels.keys()}


min_chi_eff, f_neg_chi_eff = get_distribution_properties(labels["chi_eff"])

param_dict["min_chi_eff"] = min_chi_eff
param_dict["f_neg_chi_eff"] = f_neg_chi_eff

# Combine
macros_dict = {'param':param_dict, 'ppd':ppd_dict}

# Get fraction of epsilon < 0 
eps_samples = result.get_hyperparameter_samples(hyperparameters=['eps_chi_eff']).flatten()
f_eps_negative = np.sum(eps_samples < 0)/len(eps_samples)
macros_dict['percent of epsilon less than zero'] = round(100*f_eps_negative, 1)

# Save
with open(f'EpsSkewNormalChiEff.json', 'w') as f:
	json.dump(macros_dict, f)
