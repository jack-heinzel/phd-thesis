import json
import h5py
import argparse
import seaborn as sns
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from popsummary.popresult import PopulationResult

def round_sig(x, sig=3):
    """
    Round a float or array of floats to the given number of significant digits.

    Parameters:
    x (float or array-like): Number or list/array of numbers to round.
    sig (int): Number of significant digits.

    Returns:
    Rounded number or list of rounded numbers.
    """
    import numpy as np
    x = np.asarray(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.round(x, sig - np.floor(np.log10(np.abs(x))).astype(int) - 1)

def compute_cci_from_pdf(x_values, pdf_values, cred_mass=0.90):

    """
    Function to compute quantiles from grid and corresponding pdf values

    parameters
    -----------
    x_values: values of the parameters 
    pdf_values: corresponding pdf values
    cred_mass: crebible interval

    returns
    --------
    ll, med, ul: lower quantile, median, and upper quantile according to cred_mass
    """

    pdf_values = pdf_values.copy()  
    pdf_values /= sp.integrate.simpson(pdf_values, x=x_values) # normalizing the pdf
    cdf = np.cumsum(pdf_values) / np.sum(pdf_values) # cdf calculation
    median_idx = np.searchsorted(cdf, 0.5) # median
    lower_idx = np.searchsorted(cdf, (1 - cred_mass) / 2) # lower quantile
    upper_idx = np.searchsorted(cdf, 1 - (1 - cred_mass) / 2) # upper quantile
    ll, med, ul = x_values[lower_idx], x_values[median_idx], x_values[upper_idx]
    return ll, med, ul

def estimate_macros(basedirectory, popmodel, outdir):

    """
    Function to calculate macros for sodapop analysis
    
    parameters
    -----------
    basedirectory: basedirectory containing the results
    set: event set (either GWTC3 or GWTC4)
    popmodel: neutron star population mass model
    outdir: directory to save results
    plot_result: whether to plot and save results

    returns
    ----------
    macros: a python dictionary containing quantiles of hyperparameters and ppds
    """

    metadata = {}
    cred_mass = 0.90

    for temp in ['woGW190814', 'wGW190814']:

        metadata[temp] = {"param": {}, "ppd": {}}
        resultfile = basedirectory + f'{popmodel}_m1m2_{temp}/{popmodel}_m1m2.h5'
        result = PopulationResult(fname=resultfile)

        mgrid, ppd_m1 = result.get_rates_on_grids('mass1_source')
        _, ppd_m2 = result.get_rates_on_grids('mass2_source')
        ppd_m = 0.5*ppd_m1 + 0.5*ppd_m2 # averaged over individual m1 and m2 distributions

        ll_m1_ppd, med_m1_ppd, ul_m1_ppd = np.quantile(np.array([ppd_m1[i,:] for i in range(ppd_m1.shape[0])]), axis=0, q=[(1 - cred_mass) / 2, 0.5, 1 - (1 - cred_mass) / 2])
        ll_m2_ppd, med_m2_ppd, ul_m2_ppd = np.quantile(np.array([ppd_m2[i,:] for i in range(ppd_m2.shape[0])]), axis=0, q=[(1 - cred_mass) / 2, 0.5, 1 - (1 - cred_mass) / 2])
        ll_m_ppd, med_m_ppd, ul_m_ppd = np.quantile(np.array([ppd_m[i,:] for i in range(ppd_m.shape[0])]), axis=0, q=[(1 - cred_mass) / 2, 0.5, 1 - (1 - cred_mass) / 2])

        norm_m1 = np.trapz(med_m1_ppd, x=mgrid.flatten())
        ll_m1, median_m1, ul_m1 = compute_cci_from_pdf(mgrid.flatten(), med_m1_ppd/norm_m1, cred_mass=cred_mass)
        quantiles_ppd_m1 = [ll_m1, median_m1, ul_m1]

        norm_m2 = np.trapz(med_m2_ppd, x=mgrid.flatten())
        ll_m2, median_m2, ul_m2 = compute_cci_from_pdf(mgrid.flatten(), med_m2_ppd/norm_m2, cred_mass=cred_mass)
        quantiles_ppd_m2 = [ll_m2, median_m2, ul_m2]

        norm_m = np.trapz(med_m_ppd, x=mgrid.flatten()) 
        ll, median, ul = compute_cci_from_pdf(mgrid.flatten(), med_m_ppd/norm_m, cred_mass=cred_mass)
        quantiles_ppd = [ll, median, ul]

        # metadata[temp]["ppd"] = {"mass_1_source": {"median": np.round(quantiles_ppd_m1[1], 2), "5th percentile": np.round(quantiles_ppd_m1[0], 2), \
        # "95th percentile": np.round(quantiles_ppd_m1[2], 2)}, "mass_2_source": {"median": np.round(quantiles_ppd_m2[1], 2), "5th percentile": np.round(quantiles_ppd_m2[0], 2), \
        # "95th percentile": np.round(quantiles_ppd_m2[2], 2)}, "mass_source": {"median": np.round(quantiles_ppd[1], 2), "5th percentile": np.round(quantiles_ppd[0], 2), \
        #                                                                     "95th percentile": np.round(quantiles_ppd[2], 2)}}

        metadata[temp]["ppd"] = {"mass_1_source": {"median": round_sig(quantiles_ppd_m1[1], 2), "5th percentile": round_sig(quantiles_ppd_m1[0], 2), \
        "95th percentile": round_sig(quantiles_ppd_m1[2], 2)}, "mass_2_source": {"median": round_sig(quantiles_ppd_m2[1], 2), "5th percentile": round_sig(quantiles_ppd_m2[0], 2), \
        "95th percentile": round_sig(quantiles_ppd_m2[2], 2)}, "mass_source": {"median": round_sig(quantiles_ppd[1], 2), "5th percentile": round_sig(quantiles_ppd[0], 2), \
                                                                            "95th percentile": round_sig(quantiles_ppd[2], 2)}}
                                                                            
        id = []

        if popmodel == 'power':
            params = [r'alpha', r'm_min', r'm_max']
            for i in range(len(params)):
                id.append(i)

        if popmodel == 'peakcut':
            params = [r'mu', r'sigma', r'm_min', r'm_max']
            for i in range(len(params)):
                id.append(i)

        file = h5py.File(resultfile, 'r')
        samples = file['posterior']['hyperparameter_samples'][()]
        file.close()

        for i, p in enumerate(params):
            # if np.logical_and(popmodel=='power', p=='alpha'):
            #     quantiles_hyp = np.quantile(-samples[:,i], q=[(1 - cred_mass) / 2, 0.5, 1 - (1 - cred_mass) / 2])
            # # metadata[temp]['param'][p] = {"median": np.round(quantiles_hyp[1], 2), "error plus": np.round(quantiles_hyp[2] - quantiles_hyp[1], 2), \
            # #                         "error minus": np.round(quantiles_hyp[1] - quantiles_hyp[0], 2), "5th percentile": np.round(quantiles_hyp[0], 2), \
            # #                         "95th percentile": np.round(quantiles_hyp[2], 2)}
            # else:
            quantiles_hyp = np.quantile(samples[:,i], q=[(1 - cred_mass) / 2, 0.5, 1 - (1 - cred_mass) / 2])

            metadata[temp]['param'][p] = {"median": round_sig(quantiles_hyp[1], 2), "error plus": round_sig(quantiles_hyp[2] - quantiles_hyp[1], 2), \
                        "error minus": round_sig(quantiles_hyp[1] - quantiles_hyp[0], 2), "5th percentile": round_sig(quantiles_hyp[0], 2), \
                        "95th percentile": round_sig(quantiles_hyp[2], 2)}
    
    filename = f"{outdir}/{popmodel}sodapop.json"
    json.dump(metadata, open(filename, 'w'))
    print('Done')

# def str_to_bool(v):
#     return v.lower() in ('true', 't', '1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedirectory', help='the base directory containing results')
    parser.add_argument('--popmodel', choices=['peakcut', 'power'], help='NS mass models used for analysis: peakcut or power')
    parser.add_argument('--outdir', default='./', help='the output directory')
    args = parser.parse_args()
    print(args)
    estimate_macros(**vars(args))
