import numpy as np
from popsummary.popresult import PopulationResult
import json

result_files = {
    'qchieffLinearCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/linear_mass_ratio_popsummaryV2.h5',
    'zchieffLinearCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/linear_redshift_popsummaryV2.h5',
}

hyperparameters = [
    'alpha_1',
    'alpha_2',
    'beta',
    'break_mass',
    'delta_m_1',
    'delta_m_2',
    'lam_0',
    'lam_1',
    'lamb',
    'mlow_1',
    'mlow_2',
    'mpp_1',
    'mpp_2',
    'mu_chieff_0',
    'ln_sigma_chieff_0',
    'mu_chieff_1',
    'ln_sigma_chieff_1'
]

for model, result_file in result_files.items():
    popresult = PopulationResult(result_file)
    macro_dict = dict(param=dict(), ppd=dict())

    for hyperparameter in hyperparameters:
        median, perc_5, perc_95 = np.quantile(
            popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
            q=(0.5, 0.05, 0.95)
        )
        plus = perc_95 - median
        minus = median - perc_5
            
        macro_dict['param'][hyperparameter] = {
            'median': round(float(median),2),
            '5th percentile': round(float(perc_5),2),
            '95th percentile': round(float(perc_95),2),
            'error plus': round(float(plus), 2),
            'error minus': round(float(minus), 2),
        }

        if hyperparameter in ['mu_chieff_1', 'ln_sigma_chieff_1']:
            perc_correlated = 0.5
            val = median
            if val < 0:
                while val < 0:
                    perc_correlated += 0.001
                    if perc_correlated >= 1:
                        perc_correlated = 0.999
                        break
                    val = np.quantile(
                        popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
                        perc_correlated
                    )
            else:
                while val > 0:
                    perc_correlated -= 0.001
                    if perc_correlated <= 0:
                        perc_correlated = 0.001
                        break
                    val = np.quantile(
                        popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
                        perc_correlated
                    )
                perc_correlated = 1 - perc_correlated
            macro_dict['param'][hyperparameter]['percentile exclude zero'] = round(float(100*perc_correlated),1)

    with open(f'{model}.json', 'w') as ff:
        json.dump(macro_dict, ff, indent=4)