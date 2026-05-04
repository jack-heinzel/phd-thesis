import json
from popsummary.popresult import PopulationResult
import numpy as np
import pandas as pd

macro_fnames = ["FullPop.json","FullPopNoDip.json"]
labels = ["baseline6","baseline5_A0_widesigmachi"]
popsummary_fnames = [
    f"/home/amanda.farah/projects/O4/PDB-in-O4/analysis_results/{label}_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_medians_popsummary.h5"
    for label in labels]

def round_sig(x, n):
    if x == 0:
        return 0
    rounded = np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))
    # If the number is effectively an integer, return as int
    if np.isclose(rounded, int(rounded)):
        return int(rounded)
    return rounded

def calculate_stats(series, sigfig=3):
    """Calculate statistics including median, asymmetric errors, and percentiles"""
    median = np.median(series)
    lower_err = median - np.percentile(series, 5)
    upper_err = np.percentile(series, 95) - median
    
    return {
        "median": round_sig(median, sigfig),
        "error plus": round_sig(upper_err, sigfig),
        "error minus": round_sig(lower_err, sigfig),
        "5th percentile": round_sig(np.percentile(series, 5), sigfig),
        "95th percentile": round_sig(np.percentile(series, 95), sigfig)
    }

for popsummary_fname, macro_fname in zip(popsummary_fnames,macro_fnames):
    summary = PopulationResult(fname = popsummary_fname)
    samples = summary.get_hyperparameter_samples()
    sample_names = summary.get_metadata("hyperparameters")
    an_actually_useful_object = pd.DataFrame(data=samples,columns=sample_names)

    # calculate derived quantities
    beta1_less_beta2 = float(
        np.sum(an_actually_useful_object['beta_pair_1'] < an_actually_useful_object['beta_pair_2'])/len(samples)
        )
    ## peak at ~2 solar masses
    ms, lines = summary.get_rates_on_grids("component_mass")
    ### Truncate PPDs before the lower edge of the mass gap
    in_range = ms <=2.8
    ms = ms[in_range]
    rate_vs_m = np.delete(lines, np.where(~in_range),axis=1) 
    ### Find local maximum
    idx_at_maximum = np.argmax(rate_vs_m, axis=1)
    maxima = ms[idx_at_maximum]

    # save results as json
    result = {
        "rates":{
            "BNS": calculate_stats(an_actually_useful_object['rate_bns']) if 'rate_bns' in an_actually_useful_object.columns else {},
            "NSBH": calculate_stats(an_actually_useful_object['rate_nsbh']) if 'rate_nsbh' in an_actually_useful_object.columns else {},
            "BBH": calculate_stats(an_actually_useful_object['rate_bbh']) if 'rate_bbh' in an_actually_useful_object.columns else {},
            "NS-Gap": calculate_stats(an_actually_useful_object['rate_ns-gap']) if 'rate_ns-gap' in an_actually_useful_object.columns else {},
            "BH-Gap": calculate_stats(an_actually_useful_object['rate_bh-gap']) if 'rate_bh-gap' in an_actually_useful_object.columns else {},
            "Full": calculate_stats(an_actually_useful_object['rate_full']) if 'rate_full' in an_actually_useful_object.columns else {}
        },
        "pairing_function_difference": round(beta1_less_beta2, 2),
        "peak_locations":{
            "OneMsun": calculate_stats(maxima),
            "TenMsun": calculate_stats(an_actually_useful_object['mu2']),
            "ThirtyMsun": calculate_stats(an_actually_useful_object['mu1'])
        }
    }

    # Write to JSON file
    with open(macro_fname, 'w') as f:
        json.dump(result, f)