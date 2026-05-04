import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os
import h5ify, h5py
import json
from popsummary.popresult import PopulationResult
from gwpopulation import backend
backend.set_backend(backend="numpy")
import importlib.util
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('/home/simona.miller/o4a-rp-reweighting')
import o4a_default_models as o4a_models
sys.path.append('/home/simona.miller/o4a-gaussian-effective-spins/code')
import load_inputs 

def convert_numpy_arrays_to_lists(data):
    if isinstance(data, dict):
        return {k: convert_numpy_arrays_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
        
'''
Load GWTC4 hierarchical inference results
'''

# Impor BPL2P population model
model_path = "/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/"
mass_module_name = "final_mass_models"  
spin_module_name = "final_spin_mag_model"
for module_name in [mass_module_name, spin_module_name]: 
    pp = model_path+module_name.replace('_', '-')+'.py'
    spec = importlib.util.spec_from_file_location(module_name, pp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

# Load popsummary file with results from O1-O3
hyper_parameter_fname = '/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/default_bbh/popsummary/dec4/gwtc3_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'
popsummary_result = PopulationResult(fname=hyper_parameter_fname)

# Fetch hyperparameter names
hyperparam_keys = popsummary_result.get_metadata(field = 'hyperparameters')

# Extract hyperposterior into a dictionary
HYPER_POSTERIOR = {
    k:popsummary_result.get_hyperparameter_samples(hyperparameters=[k]) for k in hyperparam_keys
}

print('Hierarchical likelihood setup finished.')


'''
Load just O4a event posteriors and injections
'''
samples_O4a_fname = '/home/simona.miller/o4a-gaussian-effective-spins/input/sampleDict_O4a_pe_rerun.h5'
sampleDict = load_inputs.loadSampleDict(samples_O4a_fname)

print('Event samples loaded.')

# load just o4a injections:
injections_path = '/home/rp.o4/offline-injections/real/T2400372-v2/samples-rpo4a_v2_20250220153231UTC-1366933504-23846400.hdf'

with h5py.File(injections_path, 'r') as obj:
    attrs = dict(obj.attrs.items())
    injections_no_cuts = obj['events'][:]

# Implement SNR and FAR cuts: 
fars = [injections_no_cuts[key] for key in injections_no_cuts.dtype.names if 'far' in key]
min_fars = np.min(fars, axis = 0)
snrs = injections_no_cuts['observed_phase_maximized_snr_net']
found = (min_fars < 1) | (snrs > 10)
injections = injections_no_cuts[found]

# format for GWpopulation
injectionDict  = dict(
    mass_1     = injections['mass1_source'],
    mass_ratio = injections['mass2_source']/injections['mass1_source'], 
    redshift   = injections['z'], 
    a_1        = injections['spin1_magnitude'],
    a_2        = injections['spin2_magnitude'],
    cos_tilt_1 = np.cos(injections['spin1_polar_angle']),
    cos_tilt_2 = np.cos(injections['spin2_polar_angle']),
)

# calculate p_draw
ln_pdraw = injections['lnpdraw_mass1_source'] + \
            injections['lnpdraw_mass2_source_GIVEN_mass1_source'] + \
            injections['lnpdraw_z'] + \
            injections['lnpdraw_spin1_magnitude'] + \
            injections['lnpdraw_spin2_magnitude'] + \
            injections['lnpdraw_spin1_polar_angle'] + \
            injections['lnpdraw_spin2_polar_angle']
pdraw_spin_components = np.exp(ln_pdraw) / injections['weights']

# jacobians
def get_tilt_jacobian(posterior): 
    # dp/dx = |dtheta/dx| * dp/d(theta)
    # let x=cos(theta), then the jacobian |dtheta/dx| = (1 - x^2)^(-1/2)
    x1 = np.cos(posterior['spin1_polar_angle'])
    x2 = np.cos(posterior['spin2_polar_angle'])
    j1 = 1/np.sqrt(1 - x1**2)
    j2 = 1/np.sqrt(1 - x2**2)
    return j1*j2

jacobian_m1m2_m1q = injections['mass1_source']
jacobian_tilts_costilts = get_tilt_jacobian(injections)

# calling p-draw "prior" so we can use the same weights fxn as w/ individual-event PE
injectionDict['prior'] = 4*np.pi**2 * pdraw_spin_components * jacobian_tilts_costilts * jacobian_m1m2_m1q

print('Injections loaded.')

# print some helpful info
events = list(sampleDict.keys())
nEvents = len(events)
nInjections = len(injectionDict[list(injectionDict.keys())[0]])
nHyperParams = len(HYPER_POSTERIOR[list(HYPER_POSTERIOR.keys())[0]])
print('Number of events:', nEvents)
print('Number of injections:', nInjections)
print('Number of hyper-parameter samples:', nHyperParams)

# Get only the parameters we want and make sure the names line up with what gwpopulation expects
desired_keys = ['mass_1', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
print('Parameters to be reweighted:', desired_keys)

def get_desired_params(data): 
    data_with_desired_params = {
        'mass_1':data['mass_1_source'], 
        'mass_ratio':data['mass_2_source']/data['mass_1_source'],
        **{k:data[k] for k in desired_keys if k in data.keys()}
    }
    return data_with_desired_params

sampleDict_with_desired_params = {event:get_desired_params(sampleDict[event]) for event in events}

# Get denominator of weights, including Jacobians for mass
for event in events: 
    sampleDict_with_desired_params[event]['prior'] = sampleDict[event]['z_prior'] * sampleDict[event]['mass_1_source']

print('Injections and event samples formatted.')

'''
Reweight
'''   

def get_weights(data, hp, pop_model):

    # evaluate population distribution
    pop_model.parameters = hp
    pi_pop = pop_model.prob(data)
    
    # get weight
    w = pi_pop/data['prior']
    
    return w

# Generate PPC traces
nTraces = 1000
hp_idxs = np.random.choice(nHyperParams, size=nTraces, replace=False)
hyperparams = [{k:v[i] for k,v in HYPER_POSTERIOR.items()} for i in hp_idxs]

# Set up data product
PPC_traces_dict = {
    'predicted':{k:np.empty((nTraces, nEvents)) for k in desired_keys},
    'observed' :{k:np.empty((nTraces, nEvents)) for k in desired_keys}
}


#########################
# Get PREDICTED catalogs
#########################

print('injections')

O4A_MODEL = o4a_models.load_o4a_default_model()

for i in tqdm(range(nTraces)): 
    
    # get weights according to this hyper parameter sample
    weights_pred = get_weights(injectionDict, hyperparams[i], O4A_MODEL)
    
    weights_pred_normed = weights_pred / np.sum(weights_pred)
    
    # select nEvents injections according to weights
    predicted_trace_idxs = np.random.choice(nInjections, size=nEvents, p=weights_pred_normed, replace=False)

    # Add to data product
    for k in desired_keys: 
        PPC_traces_dict['predicted'][k][i] = injectionDict[k][predicted_trace_idxs]

########################
# Get OBSERVED catalogs
########################

# Draw one sample per event
for n,event in enumerate(events): 

    print(event)

    # get samples for the nth event
    samplesEvent = sampleDict_with_desired_params[event]
    nSamples = len(samplesEvent[list(samplesEvent.keys())[0]])

    # reload this because of how gwpopulation works??
    O4A_MODEL = o4a_models.load_o4a_default_model()

    for i in tqdm(range(nTraces)): 

        # get weights according to this hyper parameter sample
        weights_obs = get_weights(samplesEvent, hyperparams[i], O4A_MODEL)
        
        weights_obs_normed = weights_obs / np.sum(weights_obs)
    
        # select one sample from the event posterior according to weights
        observed_trace_idx = np.random.choice(nSamples, p=weights_obs_normed)
        
        # Add to data product
        for k in desired_keys: 
            PPC_traces_dict['observed'][k][i][n] = samplesEvent[k][observed_trace_idx]
    
# SAVE
h5ify.save('PPC_traces.h5', PPC_traces_dict, mode='w')

























    