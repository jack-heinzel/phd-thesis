# ===========================================
# calculation of 99th of detectable redshifts
# ===========================================

import os
# make sure to run:
# export PYTHONPATH=$PYTHONPATH:/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/ 
# before running this!

print(
    "Moving to "
    "/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk"
)

old_dir = os.getcwd()
os.chdir("/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk")

import os
from argparse import ArgumentParser

import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.backend import set_backend
set_backend('jax')
from gwpopulation.utils import xp
from tqdm import tqdm

from gwpopulation_pipe.data_analysis import load_model
from gwpopulation_pipe.vt_helper import load_injection_data
from gwpopulation_pipe.common_format import *

parser = create_parser()

# from /home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk/init/submit/gwtc4_2pl2pk.sh common_format 
args = parser.parse_args([
    "--result-file", "init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_result.hdf5", 
    "--n-samples", "5000", 
    "--max-redshift", "1.9", 
    "--minimum-mass", "3.0", 
    "--maximum-mass", "300.0", 
    "--injection-file", "init/data/injections.pkl", 
    "--filename", "init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_full_posterior.hdf5", 
    "--samples-file", "init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_samples.pkl", 
    "--vt-ifar-threshold", "1.0", 
    "--vt-snr-threshold", "10.0", 
    "--backend", "jax", 
    "--cosmology", "Planck15_LAL", 
    "--make-popsummary-file", "True", 
    "--draw-population-samples", "False", 
    "--popsummary-file", "init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5", 
    "--event-data-file", "init/data/event_data.json"
])

set_backend(args.backend)

result = read_in_result(args.result_file)
posterior = result.posterior
args.models = result.meta_data["vt_models"]
model = load_model(args)

vt_data = load_injection_data(
    args.injection_file,
    ifar_threshold=args.vt_ifar_threshold,
    snr_threshold=args.vt_snr_threshold,
).to_dict()

from gwpopulation_pipe.utils import maybe_jit

# get sorted order of detectable redshifts
order = xp.argsort(vt_data['redshift'])

points = posterior.to_dict(orient='records')
# do empty call to get models initialized appropriately
model.parameters.update(points[0])
model.prob(vt_data)
print(xp)
@maybe_jit
def get_99th(parameters):
    model.parameters.update(parameters)
    weights = model.prob(vt_data) / vt_data["prior"]
    weights /= xp.sum(weights) # sum to 1
    sorted_weights = weights[order]
    cumsum = xp.cumsum(sorted_weights)
    _99th = xp.interp(0.99, cumsum, vt_data['redshift'][order])
    return _99th

from tqdm import tqdm
_99s = np.array([
    get_99th(point) for point in tqdm(points)
])

_50th = np.percentile(_99s, 50)
_5th = np.percentile(_99s, 5)
_95th = np.percentile(_99s, 95)

detectable = {
    '50th_percentile': float(np.round(_50th, 2)),
    '95th_percentile': float(np.round(_95th, 2)),
    '5th_percentile': float(np.round(_5th, 2)),
    'error_plus': float(np.round(_95th-_50th, 2)),
    'error_minus': float(np.round(_50th-_5th, 2)),
}

# ============================================
# calculation of 99th for GWTC-4-like catalogs
# ============================================

import popsummary
powerlaw_file = '/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk/init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

result = popsummary.popresult.PopulationResult(fname=powerlaw_file)

x = result.get_reweighted_injections(parameters=['redshift'])
zs = x[0,...,0]

# zs.shape = (153=Nobs, 3887=Nsamples)
# Nobs=153 catalogs produced for each hyperposterior sample 

zaxis = np.linspace(0,1.9,1000)
quantiles = np.array([np.mean(zs < z, axis=0) for z in zaxis])


z99 = [np.interp(0.99, quantiles[:,ii], zaxis) for ii in range(quantiles.shape[-1])]

_50th = np.percentile(z99, 50)
_5th = np.percentile(z99, 5)
_95th = np.percentile(z99, 95)

gwtc_4like = {
    '50th_percentile': float(np.round(_50th, 1)),
    '95th_percentile': float(np.round(_95th, 1)),
    '5th_percentile': float(np.round(_5th, 1)),
    'error_plus': float(np.round(_95th-_50th, 1)),
    'error_minus': float(np.round(_50th-_5th, 1)),
}

macros = {
    'detectable': detectable,
    'GWTC-4-like': gwtc_4like
    }

print(macros)
os.chdir(old_dir)

with open('../../macro_data/DefaultMaximumRedshift.json', 'w') as ff:
    json.dump(macros, ff)
