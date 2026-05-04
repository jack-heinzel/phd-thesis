import json
import h5py
import numpy as np
from popsummary.popresult import PopulationResult

def get_percentile_rank(arr, value):
    sorted_arr = np.sort(arr)
    count_le_value = np.sum(sorted_arr <= value)
    percentile_rank = (count_le_value / len(sorted_arr)) * 100
    return percentile_rank

result = PopulationResult(fname='/home/thomas.callister/CBC/o4a-pop-studies/data/rerun_cat_38214bd95_724/popsummary_independentMassRatio_O4_twoModes_cat_38214bd95_724.h5')
beta_peak2_values, beta_pl_values = result.get_hyperparameter_samples(hyperparameters=['beta_peak2', 'beta_pl']).T

tot = beta_peak2_values.shape[0]

macro_dict = {}

macro_dict['beta_peak_pl_diff_perc'] = round(get_percentile_rank(beta_pl_values - beta_peak2_values, 0.), 1)

with open("VaryingBetaQsTwoModesBetaDiffPerc.json", "w") as jf:
    json.dump(macro_dict, jf, sort_keys=True, indent=4)


result = PopulationResult(fname='/home/thomas.callister/CBC/o4a-pop-studies/data/rerun_cat_38214bd95_724/popsummary_independentMassRatio_O4_dominantMode_cat_38214bd95_724.h5')
beta_peak2_values, beta_pl_values = result.get_hyperparameter_samples(hyperparameters=['beta_peak2', 'beta_pl']).T

dom = beta_peak2_values.shape[0]
print(beta_peak2_values.shape)
macro_dict = {}

macro_dict['beta_peak_pl_diff_perc'] = round(get_percentile_rank(beta_pl_values - beta_peak2_values, 0.), 1)
macro_dict['dom_posterior_fraction'] = round(dom / tot * 100,1)

with open("VaryingBetaQsDominantModeBetaDiffPerc.json", "w") as jf:
    json.dump(macro_dict, jf, sort_keys=True, indent=4)