from popsummary.popresult import PopulationResult
import numpy as np
from scipy.integrate import cumulative_trapezoid
from argparse import ArgumentParser
import json

def load_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type = str)
    parser.add_argument('--name', type = str)
    return parser

class GetMacroData(PopulationResult):

    def __init__(self, fname=None, iid = False, isopeak = False, hyperparameters=None, hyperparameter_descriptions=None, hyperparameter_latex_labels=None, references=None, model_names=None, events=None, event_waveforms=None, event_sample_IDs=None, event_parameters=None):
        super().__init__(fname, hyperparameters, hyperparameter_descriptions, hyperparameter_latex_labels, references, model_names, events, event_waveforms, event_sample_IDs, event_parameters)
        self.isopeak = isopeak
        if iid:
            self.rate_keys = ['rate_vs_mass_1_at_z0-2', 'rate_vs_mass_ratio_at_z0-2', 'p(a)', 'p(cos_tilt)', 'rate_vs_redshift']
            self.params = ['mass_1', 'mass_ratio', 'a', 'cos_tilt', 'redshift',]
        else:
            self.rate_keys = ['rate_vs_mass_1_at_z0-2', 'rate_vs_mass_ratio_at_z0-2', 'p(a_1)', 'p(a_2)', 'p(cos_tilt_1)', 'p(cos_tilt_2)', 'rate_vs_redshift']
            self.params = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
        if self.isopeak:
            keys = self.rate_keys.copy()
            keys.remove('rate_vs_redshift')
            self.subpop_keys = {sub: [sub + '_' + key for key in keys] for sub in ['peak', 'continuum']}
        
        self.pdfs = self.get_dict_from_popsum()
        self.macros = self.generate_macros()
        

    def get_ppd(self, pxs, rate = None):
        if rate is not None:
            ppd = np.mean(pxs / rate, axis = 0)
        else:
            ppd = np.mean(pxs, axis = 0)
        return ppd

    def get_dict_from_popsum(self):

        pdf_dict = {}

        if self.isopeak:
            for sub in self.subpop_keys.keys():
                keys = self.subpop_keys[sub]
                for idx, key in enumerate(keys):
                    x, px = self.get_rates_on_grids(grid_key = key)
                    x = x[0]

                    pdf_dict[sub + '_' + self.params[idx] + '_pdfs'] = px
                    pdf_dict[sub + '_' + self.params[idx]] = x

                    if sub == 'peak':
                        z, pz = self.get_rates_on_grids(grid_key = 'rate_vs_redshift')
                        z = z[0]
                        pdf_dict['redshift_pdfs'] = pz
                        pdf_dict['redshift'] = z

        else:
            for idx, key in enumerate(self.rate_keys):
                x, px = self.get_rates_on_grids(grid_key = key)
                x = x[0]

                pdf_dict[self.params[idx] + '_pdfs'] = px
                pdf_dict[self.params[idx]] = x

        return pdf_dict

    def get_max_loc(self, param, xrange = None):
    
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
    
        if xrange is not None:
            y = y[:,xrange[0]:xrange[1]]
            x = x[xrange[0]:xrange[1]]
        
        return x[np.argmax(y, axis = 1)]

    def get_zero_slope_loc(self, param):
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
        diff = y - np.roll(y,1, axis = 1) > 0
        idxs = [np.max(np.where(~diff[i] & np.roll(diff[i], 1))[0]) for i in range(diff.shape[0])]
        return x[idxs]

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
        cumulative_prob = cumulative_trapezoid(ppd, initial = 0)
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
    
    # def record_cred_vals(self, x, decimals = 2):
        
    #     return {
    #             'median': str(np.round(x[1], decimals = decimals)),
    #             'error plus': str(np.round(x[2] - x[1], decimals = decimals).astype(str)),
    #             'error minus': str(np.round(x[1] - x[0], decimals = decimals).astype(str)),
    #             '5th percentile': str(np.round(x[0], decimals = decimals).astype(str)),
    #             '95th percentile': str(np.round(x[2], decimals = decimals).astype(str))
    #         }

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
    def generate_macros(self):

        macros = {}

        if self.isopeak:

            peak_mu = self.get_hyperparameter_samples(hyperparameters='peak_mu').T[0]
            peak_mu_cred = self.get_cred_vals(peak_mu)
            peak_logsigp = self.get_hyperparameter_samples(hyperparameters='peak_logsig').T[0]
            peak_logsigp_cred = self.get_cred_vals(peak_logsigp)
            macros['mass_1'] = {'peak': {}}
            macros['mass_1']['peak']['peak_mu'] = self.record_cred_vals(peak_mu_cred, med_decimals=3, err_decimals=1)
            macros['mass_1']['peak']['peak_logsigp'] = self.record_cred_vals(peak_logsigp_cred)

            
            macros['mass_ratio'] = {'peak': {}, 'continuum': {}}
            hyperparams = self.get_metadata('hyperparameters')

            if ('q_mu' in hyperparams) | ('q_sig' in hyperparams):
                qmu = self.get_hyperparameter_samples(hyperparameters = 'q_mu').T[0]
                qmu_cred = self.get_cred_vals(qmu)
                qsig = self.get_hyperparameter_samples(hyperparameters = 'q_sig').T[0]
                qsig_cred = self.get_cred_vals(qsig)
                macros['mass_ratio']['peak']['q_mu'] = self.record_cred_vals(qmu_cred)
                macros['mass_ratio']['peak']['q_sig'] = self.record_cred_vals(qsig_cred)

            macros['a'] = {'peak': {}, 'continuum': {}}
            macros['cos_tilt'] = {'peak': {}, 'continuum': {}}

            for sub in self.subpop_keys.keys():

                q_peak = self.get_max_loc(sub + '_mass_ratio')
                q_peak_cred = self.get_cred_vals(q_peak)
                macros['mass_ratio'][sub]['peak_location'] = self.record_cred_vals(q_peak_cred)

                for param in ['a', 'cos_tilt']:

                    peak = self.get_max_loc(sub + '_' + param)
                    peak_cred = self.get_cred_vals(peak)
                    macros[param][sub]['peak_location'] = self.record_cred_vals(peak_cred)
        else:
            for param in self.params:
                macros[param] = {}

                if param == 'mass_1':

                    peak_1 = self.get_max_loc('mass_1')
                    peak_1_cred = self.get_cred_vals(peak_1)

                    peak_2 = self.get_max_loc('mass_1', xrange = [68, -1])
                    peak_2_cred = self.get_cred_vals(peak_2)

                    macros['mass_1']['peak_1_location'] = self.record_cred_vals(peak_1_cred, med_decimals=3, err_decimals=1)
                    macros['mass_1']['peak_2_location'] = self.record_cred_vals(peak_2_cred, med_decimals=3)
                
                elif param == 'redshift':

                    idx = np.sum(self.pdfs['redshift'] < 0.2)
                    z02 = self.pdfs['redshift_pdfs'][:,idx]
                    z02_cred = self.get_cred_vals(z02)
                    macros['rate_at_z_0-2'] = self.record_cred_vals(z02_cred)
        

                else:
                    peak = self.get_max_loc(param)
                    peak_cred = self.get_cred_vals(peak)
                    macros[param]['peak'] = self.record_cred_vals(peak_cred)


                ppd_cred = self.get_ppd_cred_values(param)
                macros[param]['ppd'] = self.record_cred_vals(ppd_cred)

        return macros
    

def main():

    parser = load_parser()
    args = parser.parse_args()
    path = args.path
    name = args.name

    models = ['bspline_iid', 'isopeak_iid']

    for model in models:

        iid = True if '_iid' in model else False
        isopeak = True if 'isopeak' in model else False

        model_macros = GetMacroData(fname = path + '/' + name + '_' + model + '.h5', iid = iid, isopeak = isopeak)

        with open(model + '.json', 'w') as f:
            json.dump(model_macros.macros, f)

    with open('path_to_popsummary_files.txt', 'w') as f:
        for model in models:
            if 'peak_q' in model:
                file = '/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/jaxen/november2025_pe_update/result_files/' + 'dec4_' + model + '.h5'
                f.write(file + '\n')
            else:
                file = '/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/jaxen/november2025_pe_update/result_files/' + name + '_' + model + '.h5'
                f.write(file + '\n')
        f.write('/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/jaxen/november2025_pe_update/result_files/dec4_bspline_iid_noMinM.h5')

if __name__ == '__main__':
    main()