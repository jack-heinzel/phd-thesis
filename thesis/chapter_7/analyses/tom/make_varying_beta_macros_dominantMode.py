import json
import h5py
import numpy as np
from popsummary.popresult import PopulationResult

def get_macro_values(samps):

    value_dict = {}
    value_dict['median'] = round(np.median(samps), 1)
    value_dict['5th percentile'] = round(np.quantile(samps, 0.05), 1)
    value_dict['95th percentile'] = round(np.quantile(samps, 0.95), 1)
    value_dict['error plus'] = round(value_dict['95th percentile'] - value_dict['median'], 1)
    value_dict['error minus'] = round(value_dict['median'] - value_dict['5th percentile'], 1)

    return value_dict

result = PopulationResult(fname='../../../o4a-pop-studies/data/popsummary_independentMassRatio_O4_dominantMode_cat_38214bd95_724.h5')
beta_peak1_values, beta_peak2_values, beta_pl_values = result.get_hyperparameter_samples(hyperparameters=['beta_peak1', 'beta_peak2', 'beta_pl']).T

macro_dict = {}
macro_dict['param'] = {}
macro_dict['param']['beta_q_low_mass_peak'] = get_macro_values(beta_peak1_values)
macro_dict['param']['beta_q_high_mass_peak'] = get_macro_values(beta_peak2_values)
macro_dict['param']['beta_q_power_law'] = get_macro_values(beta_pl_values)
macro_dict['param']['beta_q_high_mass_peak_minus_beta_q_power_law'] = get_macro_values(beta_peak2_values - beta_pl_values)

with open("VaryingBetaQsDominantMode.json", "w") as jf:
    json.dump(macro_dict, jf, sort_keys=True, indent=4)
