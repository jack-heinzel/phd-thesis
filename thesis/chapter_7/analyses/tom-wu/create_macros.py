import popsummary
import json
import numpy as np

h5_file = 'popsummary.h5'

result = popsummary.popresult.PopulationResult(fname = h5_file)
bnsrate_samples = result.get_hyperparameter_samples(hyperparameters=['BNS'])
nsbh5msunrate_samples = result.get_hyperparameter_samples(hyperparameters=['NSBH5Msun'])
nsbh3msunrate_samples = result.get_hyperparameter_samples(hyperparameters=['NSBH3Msun'])
def round_sig(x, n):
    if x == 0:
        return 0
    rounded = np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))
    # If the number is effectively an integer, return as int
    if np.isclose(rounded, int(rounded)):
        return int(rounded)
    return rounded

save_dict = dict()
n_sig = 2
save_dict["RBNS"] = {"5th percentile": round_sig(np.percentile(bnsrate_samples,5), n_sig), "median": round_sig(np.percentile(bnsrate_samples,50), n_sig), "95th percentile": round_sig(np.percentile(bnsrate_samples,95), n_sig)}
save_dict["RNSBH3Msun"] = {"5th percentile": round_sig(np.percentile(nsbh3msunrate_samples,5), n_sig), "median": round_sig(np.percentile(nsbh3msunrate_samples,50), n_sig), "95th percentile": round_sig(np.percentile(nsbh3msunrate_samples,95), n_sig)}
save_dict["RNSBH5Msun"] = {"5th percentile": round_sig(np.percentile(nsbh5msunrate_samples,5), n_sig), "median": round_sig(np.percentile(nsbh5msunrate_samples,50), n_sig), "95th percentile": round_sig(np.percentile(nsbh5msunrate_samples,95), n_sig)}


with open("simplerates.json", "w", encoding="utf-8") as f:
    json.dump(save_dict, f, ensure_ascii=False, indent=4)
