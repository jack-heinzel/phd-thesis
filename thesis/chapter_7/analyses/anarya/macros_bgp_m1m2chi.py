from popsummary import popresult
import numpy as np
import json



popsummary = popresult.PopulationResult(fname = '/home/anarya.ray/public_html/gppop_spin_popsummary_rerun_ulogm.hdf5')

chi, Rp_chi = popsummary.get_rates_on_grids('effective_inspiral_spin_norm')
Rp_chi_1 = Rp_chi[0,:,:]
Rp_chi_2 = Rp_chi[1,:,:]
chi = chi[:,0]

Rp_chi=Rp_chi_1/np.trapz(Rp_chi_1,chi,axis=1)[:,None]
zeros = np.zeros_like(Rp_chi)
zeros[:, chi<=0.0] = 1.0
frac_chi_negative = np.trapz(Rp_chi*zeros, chi, axis=1)
zeros = np.zeros_like(Rp_chi)
zeros[:, chi>=0.0] = 1.0
frac_chi_positive = np.trapz(Rp_chi*zeros, chi, axis=1)
ratio_1 = frac_chi_positive/frac_chi_negative

Rp_chi=Rp_chi_2/np.trapz(Rp_chi_2,chi,axis=1)[:,None]
zeros = np.zeros_like(Rp_chi)
zeros[:, chi<=0.0] = 1.0
frac_chi_negative = np.trapz(Rp_chi*zeros, chi, axis=1)
zeros = np.zeros_like(Rp_chi)
zeros[:, chi>=0.0] = 1.0
frac_chi_positive = np.trapz(Rp_chi*zeros, chi, axis=1)
ratio_2 = frac_chi_positive/frac_chi_negative

macros = {}
r1m = {}
r1m["median"] = np.quantile(ratio_1,0.5)
r1m["5th percentile"] = np.quantile(ratio_1,0.05)
r1m["95th percentile"] = np.quantile(ratio_1,0.95)
r1m["error plus"] = r1m["95th percentile"] - r1m["median"]
r1m["error minus"] = r1m["median"] - r1m["5th percentile"]
macros["param"] = {"positive-to-negative chi_eff ratio for m in 30-40":r1m}
r1m = {}
r1m["median"] = np.quantile(ratio_2,0.5)
r1m["5th percentile"] = np.quantile(ratio_2,0.05)
r1m["95th percentile"] = np.quantile(ratio_2,0.95)
r1m["error plus"] = r1m["95th percentile"] - r1m["median"]
r1m["error minus"] = r1m["median"] - r1m["5th percentile"]
macros["param"]["positive-to-negative chi_eff ratio for m outside 30-40"] = r1m

for key in macros["param"]:
    for val in macros["param"][key]:
        macros["param"][key][val] = round(macros["param"][key][val], 1)

print(macros)
import json
with open("MassSpinCorrelatedBGPModel.json", "w") as jf:
    json.dump(macros, jf, indent=4)
    