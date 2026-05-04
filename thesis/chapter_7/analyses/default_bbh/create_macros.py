from popsummary.popresult import PopulationResult
import numpy as np
from scipy.integrate import cumtrapz
import json

class GetMacroData(PopulationResult):

    def __init__(self, fname=None, prior_file=None, rate_keys = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift'], alt_model = False, peak_diff = False, hyperparameters=None, hyperparameter_descriptions=None, hyperparameter_latex_labels=None, references=None, model_names=None, events=None, event_waveforms=None, event_sample_IDs=None, event_parameters=None):
        super().__init__(fname, hyperparameters, hyperparameter_descriptions, hyperparameter_latex_labels, references, model_names, events, event_waveforms, event_sample_IDs, event_parameters)

        self.prior_file = prior_file
        self.hyper_post = self.get_hyperparameters()
        self.rate_keys = rate_keys
        self.pdfs = self.get_dict_from_popsum()

        self.hyper_creds = {param: self.record_cred_vals(self.get_cred_vals(self.hyper_post[param])) for param in self.hyper_post.keys()}
        if '2pl1pk' in self.fname:
            self.hyper_creds['break_mass'] = self.record_cred_vals(self.get_cred_vals(self.hyper_post['break_mass']), med_decimals=3)
        self.macros = self.generate_macros(alt_model = alt_model, peak_diff = peak_diff)

    def get_ppd(self, pxs, rate = None):
        if rate is not None:
            ppd = np.mean(pxs / rate, axis = 0)
        else:
            ppd = np.mean(pxs, axis = 0)
        return ppd

    def get_hyperparameters(self):

        f = open(self.prior_file, 'r')
        x = f.read()
        hyper = [elem.split(' =')[0] for elem in x.split('\n')[1:-1]]
        hyper.remove('mmax')
        hyper.append('rate')
        post_dict = {param: self.get_hyperparameter_samples(hyperparameters=param).T[0] for param in hyper}

        return post_dict

    def get_dict_from_popsum(self):

        pdf_dict = {}

        for param in self.rate_keys:

            x, px = self.get_rates_on_grids(grid_key = param)
            x = x[0]

            pdf_dict[param + '_pdfs'] = px
            pdf_dict[param] = x

        return pdf_dict

    def get_max_loc(self, param):

        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]

        return x[np.argmax(y, axis = 1)]

    def get_zero_slope_loc(self, param):
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
        diff = y - np.roll(y,1, axis = 1) > 0
        idxs = [np.max(np.where(~diff[i] & np.roll(diff[i], 1))[0]) for i in range(diff.shape[0])]
        return x[idxs]
    
    def get_powerlaw_slopes(self, param, msel = None):
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
        if msel.tolist():
            y = y[:,msel]
            x = x[msel]
        logp = np.log10(y)
        logx = np.log10(x)
        return (logp[:,-1] - logp[:,0]) / (logx[-1] - logx[0])

    def get_cred_vals(self, x, axis = 0):
        med = np.median(x, axis = axis)
        low = np.percentile(x, 5, axis = axis)
        hi = np.percentile(x, 95, axis = axis)
        return low, med, hi

    def get_ppd_percentile(self, param, perc, rate = None):
        pdfs = self.pdfs[param + '_pdfs']
        xs = self.pdfs[param]
        ppd = self.get_ppd(pdfs, rate=rate)
        i = ppd.shape[0]
        cumulative_prob = cumtrapz(ppd, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob * perc / 100.0
        while prob > final_prob:
            i -= 1
            prob = cumulative_prob[i]
        return xs[i]
    
    def get_ppd_cred_values(self, param):
        low = self.get_ppd_percentile(param, 5)
        med = self.get_ppd_percentile(param, 50)
        hi = self.get_ppd_percentile(param, 95)
        return low, med, hi
    
    def round_sig(self, x, n):
        if x == 0:
            return 0
        sig = -int(np.floor(np.log10(np.abs(x)))) + (n - 1)
        if sig == -1:
            sig = 1
        return f'{x:.{sig}f}'


    def record_cred_vals(self, x, med_decimals = 2, err_decimals = 2, INT = False):
        med = self.round_sig(x[1], n = med_decimals)
        med = int(med) if INT else med

        return {
                'median': str(med),
                'error plus': self.round_sig(x[2] - x[1], n = err_decimals),
                'error minus': self.round_sig(x[1] - x[0], n = err_decimals),
                '5th percentile': self.round_sig(x[0], n = err_decimals),
                '95th percentile': self.round_sig(x[2], n = err_decimals)
            }
    
    def get_percentile_rank(self, arr, value):
        sorted_arr = np.sort(arr)
        count_le_value = np.sum(sorted_arr <= value)
        percentile_rank = (count_le_value / len(sorted_arr)) * 100
        return percentile_rank
    
    
    def generate_macros(self, alt_model, peak_diff = False):

        macros = self.hyper_creds

        if not alt_model:
            for param in self.rate_keys:
                macros[param] = {}

                if param == 'mass_1':

                    ms = self.pdfs['mass_1']
                    sel = (ms > 18) & (ms <= 19)
                    alpha1 = self.get_powerlaw_slopes('mass_1', msel = sel)
                    alpha1_cred = self.get_cred_vals(-alpha1)
                    diff = self.get_cred_vals(self.hyper_post['alpha_1'] - self.hyper_post['alpha_2'])
                    macros['alpha_diff'] = self.record_cred_vals(diff, med_decimals = 2, err_decimals = 2)
                    macros['alpha_diff_perc'] = round(self.get_percentile_rank(self.hyper_post['alpha_1'] - self.hyper_post['alpha_2'], 0.), 1)
                    macros['mass_1']['powerlaw_slope_18-19'] = self.record_cred_vals(alpha1_cred)

                    peak_1 = self.get_max_loc('mass_1')
                    peak_1_cred = self.get_cred_vals(peak_1)

                    peak_2 = self.get_zero_slope_loc('mass_1')
                    sel = peak_2 > 18
                    perc = peak_2[sel].shape[0] / peak_2.shape[0]
                    peak_2_cred = self.get_cred_vals(peak_2[sel])

                    macros['mass_1']['peak_1_location'] = self.record_cred_vals(peak_1_cred, med_decimals=2, err_decimals=1)
                    macros['mass_1']['peak_2_location'] = self.record_cred_vals(peak_2_cred, med_decimals=3)
                    macros['mass_1']['peak_2_percent'] = round(perc, 2) * 100
                
                elif param == 'redshift':

                    idx = np.sum(self.pdfs['redshift'] <= 0.2)
                    z02 = self.pdfs['redshift_pdfs'][:,idx]
                    z02_cred = self.get_cred_vals(z02)
                    macros['redshift']['rate_at_z_0-2'] = self.record_cred_vals(z02_cred)

                else:
                    peak = self.get_max_loc(param)
                    peak_cred = self.get_cred_vals(peak)
                    macros[param]['peak'] = self.record_cred_vals(peak_cred)

            ppd_cred = self.get_ppd_cred_values(param)
            macros[param]['ppd'] = self.record_cred_vals(ppd_cred)

        if peak_diff:
            summ = self.get_cred_vals(self.hyper_post['mpp_2'] + self.hyper_post['sigpp_2'])
            macros['mu_plus_sig'] = self.record_cred_vals(summ)

        return macros
    

def compute_bayes_factor_and_error(result_1, result_2):
    """
    Compute the Bayes factor and error between two results
    Get an estimate of the error, which seems to not be correct
    """
    log_bayes_factor = np.log10(np.exp(result_1.get_metadata("log_bayes_factor_scaled") - result_2.get_metadata("log_bayes_factor_scaled")))
    # log_error = np.sqrt(result_1.get_metadata("log_evidence_scaled_err")**2 + result_2.get_metadata("log_evidence_scaled_err")**2)

    return f'{round(log_bayes_factor, 2):.2f}'
    
def main():

    path = '/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/default_bbh/popsummary/dec4'
    name = 'gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    # name_no23 = 'gwtc4_2pl2pk_noS231123cg_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    popfile = path + '/' + name
    dom_popfile = path + '/dominant_mode_' + name
    sub_popfile = path + '/subdominant_mode_' + name
    # popfile_no23 = path + '/' + name_no23
    popfile_pl2pk1 = path + '/gwtc4_2pl1pk_mass_OnePeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'
    popfile_pl2pk3 = path + '/gwtc4_2pl3pk_mass_ThreePeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'
    popfile_plp = path + '/gwtc4_1pl1pk_mass_SinglePeakSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'
    # popfile_lognorm = path + '/gwtc4_lognormal_mass_MultiPeakLogNormalMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    popfile_gwtc3 = path + '/gwtc3_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    pl2pk2 = GetMacroData(fname = popfile, prior_file='final-default.prior', peak_diff=True)
    dom_pl2pk2 = GetMacroData(fname = dom_popfile, prior_file = 'final-default.prior', peak_diff = True)
    sub_pl2pk2 = GetMacroData(fname = sub_popfile, prior_file = 'final-default.prior', peak_diff = True)
    # pl2pk2_no23 = GetMacroData(fname = popfile_no23, prior_file='noGW231123_prior.prior')
    pl2pk1 = GetMacroData(fname = popfile_pl2pk1, prior_file='2pl1pk-prior.prior', alt_model=True)
    pl2pk3 = GetMacroData(fname = popfile_pl2pk3, prior_file='2pl3pk-prior.prior', alt_model=True)
    plp = GetMacroData(fname = popfile_plp, prior_file = '1pl1pk-prior.prior', alt_model = True)
    # lognorm = GetMacroData(fname = popfile_lognorm, prior_file = 'lognormal_prior.prior', alt_model = True)
    gwtc3_pl2pk2 = GetMacroData(fname = popfile_gwtc3, prior_file = 'final-default.prior', peak_diff = True)

    dom_n = dom_pl2pk2.pdfs['mass_1_pdfs'].shape[0]
    sub_n = sub_pl2pk2.pdfs['mass_1_pdfs'].shape[0]
    dom_per = dom_n / (dom_n + sub_n) * 100
    sub_per = sub_n / (dom_n + sub_n) * 100

    log_bayes_factors = {}

    log_bayes_factors['default_dominant_mode'] = round(dom_per)
    log_bayes_factors['default_subdominant_mode'] = round(sub_per)

    pl2pk1_logBF = compute_bayes_factor_and_error(pl2pk1, pl2pk2)   
    pl2pk3_logBF = compute_bayes_factor_and_error(pl2pk3, pl2pk2)
    plp_logBF = compute_bayes_factor_and_error(plp, pl2pk2)
    # lognorm_logBF, lognorm_logBF_err = compute_bayes_factor_and_error(lognorm, pl2pk2)

    log_bayes_factors['pl2pk1'] = pl2pk1_logBF#, 'error plus': pl2pk1_logBF_err, 'error minus': pl2pk1_logBF_err}
    log_bayes_factors['pl2pk3'] = pl2pk3_logBF#, 'error plus': pl2pk3_logBF_err, 'error minus': pl2pk3_logBF_err}
    log_bayes_factors['plp'] = plp_logBF#, 'error plus': plp_logBF_err, 'error minus': plp_logBF_err}
    # log_bayes_factors['lognorm'] = {'median': lognorm_logBF, 'error plus': lognorm_logBF_err, 'error minus': lognorm_logBF_err}

    files = [popfile, dom_popfile, sub_popfile, popfile_pl2pk1, popfile_pl2pk3, popfile_plp, popfile_gwtc3] #popfile_no23, popfile_lognorm
    with open('/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/default_bbh/path_to_popsummary_files.txt', 'w') as f:
        for file in files:
            f.write(file + '\n')

    with open("default_bbh.json", 'w') as f:
        json.dump(pl2pk2.macros, f)
    
    with open("default_bbh_dominant_mode.json", 'w') as f:
        json.dump(dom_pl2pk2.macros, f)

    with open("default_bbh_subdominant_mode.json", 'w') as f:
        json.dump(sub_pl2pk2.macros, f)

    # with open("default_bbh_no231123.json", 'w') as f:
    #     json.dump(pl2pk2_no23.macros, f)
    
    with open("default_gwtc3_bbh.json", 'w') as f:
        json.dump(gwtc3_pl2pk2.macros, f)

    with open("pl2pk1_bbh.json", 'w') as f:
        json.dump(pl2pk1.macros, f)

    with open("pl2pk3_bbh.json", 'w') as f:
        json.dump(pl2pk3.macros, f)

    with open("plp_bbh.json", 'w') as f:
        json.dump(plp.macros, f)

    # with open("lognorm_bbh.json", 'w') as f:
    #     json.dump(lognorm.macros, f)

    with open("mass_model_log_bayes_factors.json", 'w') as f:
        json.dump(log_bayes_factors, f)
    
if __name__ == '__main__':
    main()