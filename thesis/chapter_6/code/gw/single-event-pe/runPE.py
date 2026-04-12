import bilby
import numpy as np
import pandas as pd
import pickle
import os
from matplotlib import pyplot as plt
# from bilby_cython.time import greenwich_mean_sidereal_time
# import sys
import optparse

class dum_prior(object):
    def __init__(self):
        pass

    def ln_prob(self, anything):
        return 0.


def ParseCommandLine():
    parser = optparse.OptionParser()

    parser.add_option('--massOnly', action='store_true', default=False, help='Sample only from the chirp mass and mass ratio.')
    parser.add_option('--index', type=int, default=0, help='Index to run PE on.')
    parser.add_option('--nlive', type=int, default=1024)
    parser.add_option('--TimeMarg', action='store_true', default=False, help='Activate for longer signals, to use for time marginalization')
    parser.add_option('--overwrite', action='store_true', default=False, help='Overwrite saved result')
    parser.add_option('--sampler', type=str, default='dynesty')
    parser.add_option('--relativeBinning', action='store_true', default=False, help='Flag to use the relative binning likelihood')
    parser.add_option('--additionalLabel', type='str', default='')
    parser.add_option('--live-multi', action='store_true', default=False, help='Flag for sampling with live-multi method, which alternates proposals between \'diff\' which rwalks along vector between two live points, and \'volumetric\' which chooses randomly within multiple ellipsoids')
    parser.add_option('--noHM', action='store_true', default=True, help='Flag for running without and HM, uses a different injection set.')
    parser.add_option('--betterPrior', action='store_true', default=False, help='Flag for running with better prior')
    parser.add_option('--betterSample', action='store_true', default=False, help='Flag for running with better sample settings')
    parser.add_option('--replaceBad', action='store_true', default=False, help='Flag for not running on snr400 injection, index 136')
    parser.add_option('--rwalk', action='store_true', default=False, help='Flag for using rwalk sample method') # may be faster but live point chains more autocorrelated
    parser.add_option('--seed', default=0, type=int)
    parser.add_option('--loud_boi', action='store_true', default=False, help='Flag for running with better prior for loud bois')
    
    # lol dont do this
    parser.add_option('--roq', action='store_true', default=False, help='Flag for using ROQ likelihood approximation -- actually nevermind it restricts the prior too much...') 
    parser.add_option('--use_default_PSD', action='store_true', default=False) 
    parser.add_option('--dump_exact_likelihood', action='store_true', default=False)
    # parser.add_option('--fixExtrinsic', action='store_true', default=False)

    opts, args = parser.parse_args()

    return opts

use_better_prior_indices = []


opts = ParseCommandLine()
print(opts)

np.random.seed(opts.index)

dir_label = f'{opts.index}'
if opts.roq:
    dir_label += '_pv2'
elif opts.noHM:
    dir_label += f'_xp'
run_label = 'pe'
if opts.additionalLabel != '':
    run_label += '_' + opts.additionalLabel
if opts.use_default_PSD:
    run_label += '_designPSD'
if opts.roq:
    run_label += '_ROQ'
elif opts.relativeBinning:
    run_label += '_relBin'
if opts.roq:
    run_label += '_IMRPhenomPv2'
elif opts.noHM:
    run_label += '_IMRPhenomXP'
if opts.massOnly:
    run_label += '_MassOnly'
if opts.nlive != 16384:
    run_label += f'_nlive{opts.nlive}'
if opts.TimeMarg:
    run_label += '_timeMarg'
if opts.sampler != 'dynesty':
    run_label += f'_{opts.sampler}'
if opts.live_multi:
    run_label += '_lm'
if opts.rwalk:
    run_label += '_rwalk'
if opts.loud_boi:
    run_label += '_loudboiprior'
elif opts.betterPrior:
    run_label += '_broadprior'
if opts.betterSample:
    run_label += '_betterSample'

run_label = f'seed_{opts.seed}_' + run_label

run_dir = os.path.join('/home/jack.heinzel/public_html/bias_2024/mc-bias/data/gw/running-pe/pe/', dir_label)

os.makedirs(run_dir, exist_ok=True)

if opts.noHM:
    # filename = 'o4_xp_injected_detections'
    filename = 'new_o4_xp_injected_detections'
    
else:
    filename = 'found_injections_all'

with open(f'/home/jack.heinzel/public_html/bias_2024/mc-bias/data/gw/running-pe/{filename}.pkl', 'rb') as ff:
    injections = pickle.load(ff)

injection_index = opts.index
injection_parameters, injection_data = injections[injection_index]

# if opts.replaceBad and injection_index == 136:
#     print('Not running on the SNR 400 injection, going to next injection, index=400')
#     injection_parameters, injection_data = injections[400]
#     run_label += '_differentInjection'

# print(injection_parameters)
# print('\n')
# print(injection_data)

flow = injection_data.get('flow', 20.)
sampling_frequency = injection_data['sampling_frequency']
duration = injection_data['duration']
start_time = injection_data['start_time']

if opts.roq:
    approximant = 'IMRPhenomPv2'
elif opts.noHM:
    approximant = 'IMRPhenomXP'
else:
    approximant = 'IMRPhenomXPHM'

ifos_O4 = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifo_to_psd = dict(
    H1O4 = '/home/jack.heinzel/public_html/bias_2024/mc-bias/data/gw/running-pe/aligo_O4_160Mpc.txt',
    L1O4 = '/home/jack.heinzel/public_html/bias_2024/mc-bias/data/gw/running-pe/aligo_O4_160Mpc.txt',
)
for ifo in ifos_O4:
    # ifo.strain_data._frequency_mask_updated = False
    ifo.maximum_frequency = sampling_frequency/2. # could this affect the inference? SNR should be tiny... I don't think it possibly could
    if not opts.use_default_PSD:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=ifo_to_psd[ifo.name+'O4'])
    ifo.set_strain_data_from_frequency_domain_strain(
        injection_data[ifo.name + '_frequency_domain_strain'],
        start_time = start_time,
        frequency_array = injection_data[ifo.name + '_frequency_array']
    )
waveform_arguments = dict(reference_frequency=20., minimum_frequency=flow,
                                waveform_approximant=approximant) # ok I think everything was accidentally Pv2 unless I use waveform_approximant. If I accidentally use approximant, then it defaults to Pv2.
if opts.noHM: 
    waveform_arguments['PhenomXPrecVersion'] = 104

if opts.roq:
    roq_dir = f'/home/jack.heinzel/ROQ_data/IMRPhenomPv2/{injection_data["duration"]}s/'
    basis_matrix_quadratic = np.load(roq_dir + "B_quadratic.npy").T
    freq_nodes_quadratic = np.load(roq_dir + "fnodes_quadratic.npy")

    # Load the parameters describing the valid parameters for the basis.
    params = np.genfromtxt("params.dat", names=True)

    waveform_arguments.update({
        'frequency_nodes_linear': freq_nodes_linear, 
        'frequency_nodes_quadratic': freq_nodes_quadratic,
    })
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    )
if opts.relativeBinning:
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    )
else:
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    )
priors = bilby.gw.prior.BBHPriorDict()
priors.pop('mass_1')
priors.pop('mass_2')

m1 = injection_parameters['mass_1_source']
q = injection_parameters['mass_ratio']
z = injection_parameters['redshift']
tc = injection_parameters['geocent_time']
ra = injection_parameters['ra']
dec = injection_parameters['dec']
injection_parameters['psi'] = injection_parameters['psi'] % np.pi

# gmst = greenwich_mean_sidereal_time(tc) % 2*np.pi

dL = bilby.gw.conversion.redshift_to_luminosity_distance(injection_parameters['redshift'])

time_delay = ifos_O4[0].time_delay_from_geocenter(
    injection_parameters["ra"],
    injection_parameters["dec"],
    injection_parameters["geocent_time"],
)

inj_chirp_mass = m1 * (1+z) * (q**3 / (1+q))**(1/5)
injection_parameters['chirp_mass'] = inj_chirp_mass

if opts.loud_boi:
    luminosity_distance_2p3 = bilby.gw.conversion.redshift_to_luminosity_distance(2.3)
    priors['chirp_mass'] = bilby.gw.prior.Uniform(inj_chirp_mass*0.99, inj_chirp_mass*1.01, name='chirp_mass', latex_label='$\mathcal{M}_c$')
    priors['mass_ratio'] = bilby.gw.prior.Uniform(0.1, 1., name='mass_ratio', latex_label='$q$')
    priors['luminosity_distance'] = bilby.gw.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=np.maximum(np.minimum(dL*3, luminosity_distance_2p3), 400), unit='Mpc', latex_label='$d_L$')
    priors["H1_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] + time_delay - 0.0005,
        maximum=injection_parameters["geocent_time"] + time_delay + 0.0005,
        name="H1_time",
        latex_label="$t_H$",
        unit="$s$",
    )
elif opts.betterPrior:
    luminosity_distance_2p3 = bilby.gw.conversion.redshift_to_luminosity_distance(2.3)
    priors['chirp_mass'] = bilby.gw.prior.Uniform(inj_chirp_mass*0.5, inj_chirp_mass*2, name='chirp_mass', latex_label='$\mathcal{M}_c$')
    priors['mass_ratio'] = bilby.gw.prior.Uniform(0.1, 1., name='mass_ratio', latex_label='$q$')
    priors['luminosity_distance'] = bilby.gw.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=np.maximum(np.minimum(dL*5, luminosity_distance_2p3), 2000), unit='Mpc', latex_label='$d_L$')
    priors["H1_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] + time_delay - 0.025,
        maximum=injection_parameters["geocent_time"] + time_delay + 0.025,
        name="H1_time",
        latex_label="$t_H$",
        unit="$s$",
    )
else:
    luminosity_distance_2p3 = bilby.gw.conversion.redshift_to_luminosity_distance(2.3)
    priors['chirp_mass'] = bilby.gw.prior.Uniform(inj_chirp_mass*0.8, inj_chirp_mass*1.2, name='chirp_mass', latex_label='$\mathcal{M}_c$')
    priors['mass_ratio'] = bilby.gw.prior.Uniform(0.1, 1., name='mass_ratio', latex_label='$q$')
    priors['luminosity_distance'] = bilby.gw.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=np.maximum(np.minimum(dL*3, luminosity_distance_2p3), 2000), unit='Mpc', latex_label='$d_L$')
    priors["H1_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] + time_delay - 0.01,
        maximum=injection_parameters["geocent_time"] + time_delay + 0.01,
        name="H1_time",
        latex_label="$t_H$",
        unit="$s$",
    )
priors['a_1'] = bilby.gw.prior.Uniform(0., 0.999, name='a_1', latex_label='$a_1$')
priors['a_2'] = bilby.gw.prior.Uniform(0., 0.999, name='a_2', latex_label='$a_2$')

injection_parameters['luminosity_distance'] = dL
injection_parameters['H1_time'] = injection_parameters['geocent_time'] + time_delay

if opts.massOnly:
    for key in priors:
        if key in ['chirp_mass', 'mass_ratio']:
            continue
        priors[key] = injection_parameters[key]
        # print(key)

print(f'Running on injection {injection_parameters}')
print('Using priors:')
for key in priors:
    print(f'{key}: {priors[key]}')

if opts.dump_exact_likelihood:
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos_O4, 
        waveform_generator,
        distance_marginalization=False,
        time_marginalization=opts.TimeMarg,
        phase_marginalization=False,
        priors=priors,
        jitter_time=True,
        reference_frame='sky',
        time_reference='H1'
    )
    import pickle as pkl
    with open(os.path.join(run_dir, 'exact_likelihood.pkl'), 'wb') as ff:
        pkl.dump(likelihood, ff)
    quit()


if opts.relativeBinning:
    # at the moment, reweighting eff into posterior is very bad... noah colm are working on this
    fiducial_parameters = injection_parameters.copy()

    for key in injection_parameters:
        if 'prob' in key:
            fiducial_parameters.pop(key)
        if key in ['mass_1_source', 'redshift', 'geocent_time', 'snr', 'chi_eff']:
            fiducial_parameters.pop(key)

    print(fiducial_parameters)
    marg = False
    if opts.massOnly:
        marg = False
    reference = 'sky'
    eps = 0.025
    if opts.betterSample:
        # reference = 'H1L1'
        eps = 0.05
        # priors['azimuth'] = priors.pop('ra')
        # priors['zenith'] = priors.pop('dec')
    try:
        likelihood = bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient(
            ifos_O4, 
            waveform_generator,
            fiducial_parameters=fiducial_parameters,
            update_fiducial_parameters=True,
            distance_marginalization=marg,
            time_marginalization=opts.TimeMarg,
            phase_marginalization=False,
            priors=priors,
            jitter_time=True,
            reference_frame=reference, # ra dec
            time_reference='H1',
            epsilon=eps # 
        )
    except:
        likelihood = bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient(
            ifos_O4, 
            waveform_generator,
            fiducial_parameters=fiducial_parameters,
            update_fiducial_parameters=False,
            distance_marginalization=marg,
            time_marginalization=opts.TimeMarg,
            phase_marginalization=False,
            priors=priors,
            jitter_time=True,
            reference_frame=reference, # ra dec
            time_reference='H1',
            epsilon=eps # 
        )
else:
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos_O4, 
        waveform_generator,
        distance_marginalization=False,
        time_marginalization=opts.TimeMarg,
        phase_marginalization=False,
        priors=priors,
        jitter_time=True,
        reference_frame='sky',
        time_reference='H1'
    )
    


# delete result if overwrite is flagged

resume = not opts.overwrite

proposals = ['diff']
bound = 'live'
if opts.live_multi:
    proposals.append('volumetric')
    bound += '-multi'


np.random.seed(opts.seed)
if opts.betterSample:
    if opts.rwalk:
        result = bilby.run_sampler(likelihood=likelihood, priors=priors, label=run_label, outdir=run_dir, sample='rwalk',
                            proposals=proposals, sampler=opts.sampler, injection_parameters=injection_parameters, 
                            nlive=opts.nlive, resume=resume, bound=bound,
                            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                            )
    else:
        result = bilby.run_sampler(likelihood=likelihood, priors=priors, label=run_label, outdir=run_dir, sample='acceptance-walk',
                            proposals=proposals, sampler=opts.sampler, injection_parameters=injection_parameters, 
                            nlive=opts.nlive, resume=resume, bound=bound,
                            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                            )

else:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, label=run_label, outdir=run_dir,
                            proposals=proposals, sampler=opts.sampler, injection_parameters=injection_parameters, 
                            nlive=opts.nlive, resume=resume, bound=bound,
                            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
                            )


# use bound='live-multi' when using both proposals options: proposals=['diff', 'volumetric']. 'diff' uses multiples of the difference of two random points for proposals, volumetric randomly selects from inside multiple ellipsoids.

result.plot_corner()
print(result.nested_samples.keys())

if opts.relativeBinning:
    true_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments.copy(),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    )
    true_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos_O4, 
        true_waveform_generator,
        distance_marginalization=marg,
        time_marginalization=opts.TimeMarg,
        phase_marginalization=False,
        priors=priors,
        jitter_time=True,
        reference_frame='sky',
        time_reference='H1'
    ) # it DOESN'T MAKE SENSE THAT IT BREAKS!!
    from copy import copy

    _result = copy(result)
    
    for key in priors:
        if isinstance(priors[key], bilby.core.prior.DeltaFunction):
            _result.nested_samples[key] = priors[key].sample()
        
    # breakpoint()

    d = dum_prior()

    true_result = bilby.core.result.reweight(
        _result, label=run_label+'_reweighted', new_likelihood=true_likelihood, 
        old_likelihood=likelihood, use_nested_samples=False, old_prior=d,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        )

    true_result.save_to_file()
    true_result.plot_corner()

