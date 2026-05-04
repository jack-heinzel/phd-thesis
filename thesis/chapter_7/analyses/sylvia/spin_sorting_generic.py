import h5py 
import argparse
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, beta
from scipy.integrate import cumulative_trapezoid
import numpy as np
import json
from matplotlib import rcParams
#rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'

def main(result_file, outdir, load_result, plot_result, run_type):
    basename = result_file.rsplit('.', 1)[0].split('/')[-1]
    
    if not load_result:
        data = h5py.File(result_file, 'r')
        mu_chi = np.array(data['posterior']['mu_chi'])
        sigma_chi = np.array(data['posterior']['sigma_chi'])
        alphas = ((1 - mu_chi) / sigma_chi**2 - 1./mu_chi) * mu_chi**2
        betas = alphas*(1./mu_chi - 1)
        lower = (0 - mu_chi) / sigma_chi
        upper = (1 - mu_chi) / sigma_chi
        chis = np.linspace(0,1,1000)
        np.random.seed(1234)
        inds = np.random.choice(range(0, len(lower)), size=1000, replace=False)

        p_chiA = []
        p_chiB = []
        lower_chiA = []
        upper_chiA = []
        lower_chiB = []
        upper_chiB = []
        for i in inds:
            if run_type == 'truncnorm':
                pdf = truncnorm.pdf(chis, lower[i], upper[i], loc=mu_chi[i], scale=sigma_chi[i])
                cdf = truncnorm.cdf(chis, lower[i], upper[i], loc=mu_chi[i], scale=sigma_chi[i])
            elif run_type == 'beta':
                pdf = beta.pdf(chis, alphas[i], betas[i])
                cdf = beta.cdf(chis, alphas[i], betas[i])
            pdf_chiA = 2*pdf*cdf
            pdf_chiB = 2*pdf*(1-cdf)
            cdf_chiA = cdf**2
            cdf_chiB = cdf*(2-cdf)
            p_chiA.append(list(pdf_chiA))
            p_chiB.append(list(pdf_chiB))
            lower_chiA.append(np.interp(0.01, cdf_chiA, chis))
            upper_chiA.append(np.interp(0.99, cdf_chiA, chis))
            lower_chiB.append(np.interp(0.01, cdf_chiB, chis))
            upper_chiB.append(np.interp(0.99, cdf_chiB, chis))
        ppd_chiA = np.mean(p_chiA, axis=0)
        ppd_chiB = np.mean(p_chiB, axis=0)
        bounds = {'chi_A_1st': lower_chiA, 'chi_A_99th': upper_chiA, 'chi_B_1st': lower_chiB, 'chi_B_99th': upper_chiB}
        metadata = {'param': {}, 'ppd': {}}
        for key in bounds:
            metadata['param'][key] = {'median': np.round(float(np.median(bounds[key])), decimals=2), 'error plus': np.round(float(np.quantile(bounds[key], 0.95) - np.median(bounds[key])), decimals=2), 
                    'error minus': np.round(float(np.median(bounds[key]) - np.quantile(bounds[key], 0.05)), decimals=2), '5th percentile': np.round(float(np.quantile(bounds[key], 0.05)), decimals=2),
                    '95th percentile': np.round(float(np.quantile(bounds[key], 0.95)), decimals=2)}
            print('{} = {} ^ {} _ {}'.format(key, metadata['param'][key]['median'], metadata['param'][key]['error plus'], metadata['param'][key]['error minus']))
        results_dict = {'chis': chis.tolist(), 'ppd_chiA': ppd_chiA.tolist(),
                        'ppd_chiB': ppd_chiB.tolist(), 'pdf_list_chiA': p_chiA, 'pdf_list_chiB': p_chiB}
        json.dump(results_dict, open('{}/{}_chiA_chiB.json'.format(outdir, basename), 'w'))

        ppd_chiA_cdf = cumulative_trapezoid(ppd_chiA, chis, initial=0)
        ppd_chiA_cdf /= ppd_chiA_cdf[-1]
        ppd_chiB_cdf = cumulative_trapezoid(ppd_chiB, chis, initial=0)
        ppd_chiB_cdf /= ppd_chiB_cdf[-1]
        param_names = ['chi_A', 'chi_B']
        param_vals = [ppd_chiA_cdf, ppd_chiB_cdf]
        for i, param in enumerate(param_vals):
            metadata['ppd'][param_names[i]] = {'median': np.round(np.interp(0.5, param, chis), decimals=2), '5th percentile': np.round(np.interp(0.05, param, chis), decimals=2),
                                               '95th percentile': np.round(np.interp(0.95, param, chis), decimals=2)}
        metadata['ppd']['chi_A']['spin at peak'] = np.round(chis[np.argmax(np.nan_to_num(ppd_chiA, posinf=0))], decimals=2)
        metadata['ppd']['chi_B']['spin at peak'] = np.round(chis[np.argmax(np.nan_to_num(ppd_chiB, posinf=0))], decimals=2)
        filename = "".join(x.capitalize() for x in basename.split('_nid')[0].split('_'))+"SpinSorting.json"
        json.dump(metadata, open(f"{outdir}/{filename}", 'w'))

    else:
        # load results
        results_dict = json.load(open('{}/{}_chiA_chiB.json'.format(outdir, basename), 'r'))

    if plot_result:
        # plot
        fig = plt.figure(figsize=(12,15))
        ax1 = fig.add_subplot(211)
        for i, pdf in enumerate(results_dict['pdf_list_chiA']):
            ax1.plot(results_dict['chis'], pdf, color='#1f77b4', alpha=0.1)
        ax1.plot(results_dict['chis'], results_dict['ppd_chiA'], color='k')
        for param in ['median', '5th percentile', '95th percentile']:
            ax1.axvline(metadata['ppd']['chi_A'][param], color='k', ls='--')
        ax1.axvline(metadata['ppd']['chi_A']['spin at peak'], color='k')
        ax1.set_xlabel(r'$\chi_{A}$', fontsize=28)
        ax1.set_ylabel(r'$p(\chi_{A})$', fontsize=28)
        ax1.tick_params(axis='both', which='major', labelsize=24)
        ax1.set_ylim(0,)
        ax1.set_xlim(0,1)
        ax1.grid(False)

        ax2 = fig.add_subplot(212)
        for pdf in results_dict['pdf_list_chiB']:
            ax2.plot(results_dict['chis'], pdf, color='#1f77b4', alpha=0.1)
        ax2.plot(results_dict['chis'], results_dict['ppd_chiB'], color='k')
        for param in ['median', '5th percentile', '95th percentile']:
            ax2.axvline(metadata['ppd']['chi_B'][param], color='k', ls='--')
        ax2.axvline(metadata['ppd']['chi_B']['spin at peak'], color='k')
        ax2.set_xlabel(r'$\chi_{B}$', fontsize=28)
        ax2.set_ylabel(r'$p(\chi_{B})$', fontsize=28)
        ax2.tick_params(axis='both', which='major', labelsize=24)
        ax2.set_ylim(0,10)
        ax2.set_xlim(0,1)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig('{}/{}_ppd.pdf'.format(outdir, basename))
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', help='the base gwpopulation result file')
    parser.add_argument('--outdir', default='./', help='the output directory')
    parser.add_argument('--run-type', choices=['truncnorm', 'beta'], help='the spin magnitude model used in the original run')
    parser.add_argument('--load-result', action='store_true', help='whether to load in the chi_A/B ppds from json')
    parser.add_argument('--plot-result', action='store_true', help='whether to plot the ppds')
    args = parser.parse_args()
    print(args)
    main(**vars(args))
