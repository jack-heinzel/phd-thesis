# https://git.ligo.org/publications/o4/cbc/o4a-astrodist/-/wikis/Home/Macros

# For all models:
#     Median, 5th percentile, and 95th percentile of the PPD for each spin parameter,
#     Spin value at which the PPD peaks
# For parametric models:
#     90% credible intervals each hyper-parameter
# For models looking at chi-eff: 90% credible intervals on:
#     1st percentile of the chi-eff distribution
#     The fraction of the distribution with negative chi-eff
# For models looking at (mass-sorted) spin magnitudes and tilts:
#     log bayes factors between IID and non-IID

import os
import numpy as np
import scipy
import json
from popsummary import PopulationResult


def recursive_round(d):
    for k in d:
        if type(d[k]) is dict:
            recursive_round(d[k])
        else:
            d[k] = round(d[k], 2)
    return d


def main():
    path = '/home/matthew.mould/o4a-pop/data'

    labels = sorted([label for label in os.listdir(path) if 'mag' in label])

    new_default_label = [label for label in labels if 'mag_truncnorm_iid_tilt_isotropic_truncnorm_nid' in label][0]
    new_default_file = f'{path}/{new_default_label}/{new_default_label}_evidences.json'
    new_default = json.load(open(new_default_file, 'r'))

    old_default_label = [label for label in labels if 'mag_beta_constrained_iid_tilt_isotropic_aligned_nid' in label][0]
    old_default_file = f'{path}/{old_default_label}/{old_default_label}_evidences.json'
    old_default = json.load(open(old_default_file, 'r'))

    for label in labels:
        if "mmax" in label:
            continue
        file = f'{path}/{label}/{label}_evidences.json'
        evidences = json.load(open(file, 'r'))

        data = {}

        data['ln_evidence'] = evidences['ln_evidence']
        data['ln_evidence_error'] = evidences['ln_evidence_error']

        data['ln_bayes_factor_over_new_default'] = data['ln_evidence'] - new_default['ln_evidence']
        data['ln_bayes_factor_over_old_default'] = data['ln_evidence'] - old_default['ln_evidence']

        summary = PopulationResult(
            fname = f'{path}/{label}/{label}_popsummary.h5',
        )

        data['param'] = {}
        params = summary.get_metadata('hyperparameters')
        samples = summary.get_hyperparameter_samples().T
        for param, sample in zip(params, samples):
            l, m, u = np.quantile(sample, (0.05, 0.5, 0.95))
            data['param'][param] = {
                'median': m,
                '5th percentile': l,
                '95th percentile': u,
                'error plus': u - m,
                'error minus': m - l,
            }

        data['ppd'] = {}
        for key in summary.get_rates_on_grids_keys():
            grid, ps = summary.get_rates_on_grids(key)
            grid = np.squeeze(grid)
            ppd = np.mean(ps, axis = 0)
            cdf = scipy.integrate.cumulative_trapezoid(ppd, grid, initial = 0)
            l, m, ninety, u = np.interp((0.05, 0.5, 0.90, 0.95), cdf, grid)
            peak = grid[np.argmax(ppd)]
            data['ppd'][key] = {
                'median': m,
                '5th percentile': l,
                '90th percentile': ninety,
                '95th percentile': u,
                'peak': peak,
            }

        data = recursive_round(data)

        name = label.split('-ref')[0]
        name = ''.join([s[0].upper() + s[1:] for s in name.split('_')])
        with open(f'./{name}.json', 'w') as f:
            json.dump(data, f)
            os.system(
                'cd ../../macro_data; '
                f'ln -s ../analyses/matt/{name}.json {name}.json',
            )

        with open(f'./popsummary_filepath_BBHSpin_{name}.txt', 'w') as f:
            f.write(f'{path}/{label}/{label}_popsummary.h5')


if __name__ == '__main__':
    main()
