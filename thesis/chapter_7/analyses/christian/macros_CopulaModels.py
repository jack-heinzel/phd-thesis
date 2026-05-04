import numpy as np
from popsummary.popresult import PopulationResult
import json

result_files = {
    'qchieffCopulaCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/qchieffCopulaCorrelationModelPopsummaryV2.h5',
    'mchieffCopulaCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/mchieffCopulaCorrelationModelPopsummaryV2.h5',
    'zchieffCopulaCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/zchieffCopulaCorrelationModelPopsummaryV2.h5',
    'mzCopulaCorrelationModel':
        '/home/christian.adamcewicz/public_html/o4a_astrodist/mzCopulaCorrelationModelPopsummaryV2.h5',
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
    'mmax',
    'mpp_1',
    'mpp_2',
    'mu_chi_eff',
    'mu_chi_p',
    'sigma_chi_eff',
    'sigma_chi_p',
    'kappa',
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

        if hyperparameter == 'kappa':
            perc_correlated = 0.5
            kappa = median
            if kappa < 0:
                while kappa < 0:
                    perc_correlated += 0.001
                    kappa = np.quantile(
                        popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
                        perc_correlated
                    )
            else:
                while kappa > 0:
                    perc_correlated -= 0.001
                    kappa = np.quantile(
                        popresult.get_hyperparameter_samples(hyperparameters=hyperparameter),
                        perc_correlated
                    )
                perc_correlated = 1 - perc_correlated
            macro_dict['param'][hyperparameter]['correlated percentile'] = round(float(100*perc_correlated),1)

    with open(f'{model}.json', 'w') as ff:
        json.dump(macro_dict, ff, indent=4)