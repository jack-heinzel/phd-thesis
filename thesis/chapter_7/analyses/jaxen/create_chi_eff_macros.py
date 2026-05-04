from popsummary.popresult import PopulationResult
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import json
from argparse import ArgumentParser

def load_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type = str)
    parser.add_argument('--name', type = str)
    return parser

def get_percentile(y, x, perc):
    xs = np.zeros(y.shape[0])
    for idx in range(y.shape[0]):
        i = y[idx].shape[0]
        cumulative_prob = cumulative_trapezoid(y[idx], initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob * perc / 100.0
        while prob > final_prob:
            i -= 1
            prob = cumulative_prob[i]
        xs[idx] = x[i]
    return xs


def get_percent_less_zero(y, x):
    sel = x < 0
    percent = trapezoid(y[:,sel], x[sel], axis = 1)
    return percent * 100

def get_cred_vals(x, axis = 0):
        med = np.median(x, axis = axis)
        low = np.percentile(x, 5, axis = axis)
        hi = np.percentile(x, 95, axis = axis)
        return low, med, hi

def record_cred_vals(x, decimals = 2):
        
        return {
                'median': str(np.round(x[1], decimals = decimals)),
                'error plus': str(np.round(x[2] - x[1], decimals = decimals).astype(str)),
                'error minus': str(np.round(x[1] - x[0], decimals = decimals).astype(str)),
                '5th percentile': str(np.round(x[0], decimals = decimals).astype(str)),
                '95th percentile': str(np.round(x[2], decimals = decimals).astype(str))
            }

def main():
     
    parser = load_parser()
    args = parser.parse_args()
    path = args.path
    name = args.name

    popfile = PopulationResult(fname = path + '/' + name + '_chi_eff_bspline.h5')

    chi, chi_pdfs = popfile.get_rates_on_grids(grid_key = 'p(chi_eff)')
    chi = chi[0]

    macros = {}
    macros['chi_eff'] = {}

    percent_less = get_percent_less_zero(chi_pdfs, chi)
    percent_less_cred = get_cred_vals(percent_less)
    macros['chi_eff']['percent_less_zero'] = record_cred_vals(percent_less_cred, decimals = 1)

    first_percentile = get_percentile(chi_pdfs, chi, 1)
    first_percentile_cred = get_cred_vals(first_percentile)
    macros['chi_eff']['1st_percentile'] = record_cred_vals(first_percentile_cred, decimals = 2)

    with open("chi_eff_bspline.json", 'w') as f:
        json.dump(macros, f)
    
    with open('path_to_chi_popsummary_file.txt', 'w') as f:
        f.write('/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/jaxen/november2025_pe_update/result_files/' + 'chi_eff_bspline_' + name + '.h5')
if __name__ == '__main__':
    main()