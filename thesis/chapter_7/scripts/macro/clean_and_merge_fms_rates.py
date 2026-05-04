import json
import numpy as np

with open('../../macro_data/FullPop.json', 'r') as ff:
    pdb = json.load(ff)
with open('../../macro_data/OriginalFullMassSpectrumBGPRate.json', 'r') as ff:
    bgp = json.load(ff)

pairing = pdb['pairing_function_difference']

pdb = pdb['rates']

bgp_translator = {
    'bns': 'R_BNS',
    'nsbh': 'R_NSBH',
    'bbh': 'R_BBH',
    'ns-gap': 'R_NS-Gap',
    'bh-gap': 'R_BH-Gap',
    'full': 'R_Full',
    'median': '50th_percentile',
    '95': '95th_percentile',
    '5': '5th_percentile',
    'plus': 'error_plus',
    'minus': 'error_minus' 
    }

pdb_translator = {
    'bns': 'BNS',
    'nsbh': 'NSBH',
    'bbh': 'BBH',
    'ns-gap': 'NS-Gap',
    'bh-gap': 'BH-Gap',
    'full': 'Full',
    'median': 'median',
    '95': '95th percentile',
    '5': '5th percentile',
    'plus': 'error plus',
    'minus': 'error minus' 
    }

def round_2_sigs(x, sig_figs=None):
    if sig_figs is None:
        sig_figs = 1 - int(np.log10(x))
    if sig_figs <= 0:
        return int(np.round(x, sig_figs))
    else:
        return float(np.round(x, sig_figs))

merged_rates = {}
for key in ['bns', 'nsbh', 'bbh', 'ns-gap', 'bh-gap', 'full']:
    w = bgp[bgp_translator[key]]
    s = pdb[pdb_translator[key]]
    
    add_dict = {}
    median = (w[bgp_translator['median']] + s[pdb_translator['median']]) / 2

    add_dict[bgp_translator['median']] = round_2_sigs((w[bgp_translator['median']] + s[pdb_translator['median']]) / 2)
    add_dict[bgp_translator['95']] = round_2_sigs(np.max([w[bgp_translator['95']], s[pdb_translator['95']]]))
    add_dict[bgp_translator['5']] = round_2_sigs(np.min([w[bgp_translator['5']], s[pdb_translator['5']]]))
    add_dict[bgp_translator['plus']] = round_2_sigs(add_dict[bgp_translator['95']] - add_dict[bgp_translator['median']])
    add_dict[bgp_translator['minus']] = round_2_sigs(-add_dict[bgp_translator['5']] + add_dict[bgp_translator['median']])

    merged_rates[bgp_translator[key]] = add_dict

    f_sig = 1 - int(np.log10(pdb[pdb_translator[key]][pdb_translator['median']]))
    w_sig = 1 - int(np.log10(bgp[bgp_translator[key]][bgp_translator['median']]))
    
    for stat in ['median', '95', '5', 'plus', 'minus']:
        pdb[pdb_translator[key]][pdb_translator[stat]] = round_2_sigs(pdb[pdb_translator[key]][pdb_translator[stat]], sig_figs=f_sig)
        bgp[bgp_translator[key]][bgp_translator[stat]] = round_2_sigs(bgp[bgp_translator[key]][bgp_translator[stat]], sig_figs=w_sig)

#print(merged_rates)
with open('../../macro_data/FullMassSpectrumMerged.json', 'w') as ff:
    json.dump(merged_rates, ff)

with open('../../macro_data/FullMassSpectrumPDB.json', 'w') as ff:
    json.dump({'rates': pdb, 'pairing_function_difference': 100*pairing}, ff)

with open('../../macro_data/FullMassSpectrumBGPRate.json', 'w') as ff:
    json.dump(bgp, ff)