import numpy as np
from popsummary.popresult import PopulationResult
import json

result_file = '/home/christian.adamcewicz/public_html/o4a_astrodist/MinimumTiltModelPopsummaryV2.h5'

popresult = PopulationResult(result_file)
macro_dict = dict(param=dict(), ppd=dict())

hyperparameters = [
    'alpha_1',
    'alpha_2',
    'amax',
    'beta',
    'break_mass',
    'delta_m_1',
    'delta_m_2',
    'lam_0',
    'lam_1',
    'lamb',
    'mlow_1',
    'mlow_2',
    'mmax',
    'mpp_1',
    'mpp_2',
    'mu_chi',
    'mu_spin',
    'sigma_chi',
    'sigma_spin',
    'sigpp_1',
    'sigpp_2',
    'tmin',
    'xi_spin'
]

for hyperparameter in hyperparameters:
    median, perc_5, perc_95 = np.quantile(
        popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
        q=(0.5, 0.05, 0.95)
    )
    macro_dict['param'][hyperparameter] = {
        'median': round(float(median),2),
        '5th percentile': round(float(perc_5),2),
        '95th percentile': round(float(perc_95),2), 
        'error plus': round(float(perc_95-median),2),
        'error minus': round(float(median-perc_5),2)
    }

for parameter in ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']:
    median, perc_5, perc_95 = np.quantile(
        popresult.get_fair_population_draws(parameters=parameter),
        q=(0.5, 0.05, 0.95)
    )
    if 'a_' in parameter:
        pos, rate = popresult.get_rates_on_grids('magnitude')
    elif 'cos_tilt_' in parameter:
        pos, rate = popresult.get_rates_on_grids('tilt')
    ppd = np.mean(rate, axis=0)
    peak = pos[0][(ppd==max(ppd))][0]
    macro_dict['ppd'][parameter] = {
        'median': round(float(median),2),
        '5th percentile': round(float(perc_5),2),
        '95th percentile': round(float(perc_95),2),
        'peak': round(float(peak),2)
    }

# include fraction of cos_theta < 0 
pos_tilt, rate_tilt = popresult.get_rates_on_grids('tilt')
pos_tilt = pos_tilt.flatten()
p_neg = np.trapz(rate_tilt[:, pos_tilt<=0], pos_tilt[pos_tilt<=0], axis=1)
fifth_neg, median_neg, ninetyfifth_neg = np.quantile(p_neg, [0.05, 0.5, 0.95])

macro_dict['fraction negative cos theta'] = {
    'median': round(float(median_neg),2),
    '5th percentile': round(float(fifth_neg),2),
    '95th percentile': round(float(ninetyfifth_neg),2), 
    'error plus': round(float(ninetyfifth_neg-median_neg),2),
    'error minus': round(float(median_neg-fifth_neg),2)
}

with open('MinimumTiltModel.json', 'w') as ff:
    json.dump(macro_dict, ff, indent=4)