import json
import numpy as np

d = dict()


def assign_vals_in_dict(data_dict, target_dict, model_key):
    target_dict[model_key] = dict()
    for key in data_dict.keys():
        target_dict[model_key][key] = dict()
        for stat_key in data_dict[key].keys():
            target_dict[model_key][key][stat_key] = int(
                np.round(data_dict[key][stat_key], 2) * 100
            )


mmax300_data = {
    "total_mass_source": {
        "median": 0.8738821995043657,
        "error plus": 0.06063255968773451,
        "error minus": 0.09663131806887681,
    },
    "mass_1_source": {
        "median": 0.8151243840441235,
        "error plus": 0.0909868749985644,
        "error minus": 0.10996741097533846,
    },
    "mass_2_source": {
        "median": 0.8727103943177483,
        "error plus": 0.08416790626100612,
        "error minus": 0.1163671814520576,
    },
}

mmaxlessthan300_data = {
    "total_mass_source": {
        "median": 0.9388283308141553,
        "error plus": 0.039755238459005415,
        "error minus": 0.05101163890379534,
    },
    "mass_1_source": {
        "median": 0.9162384028141263,
        "error plus": 0.05155455390496322,
        "error minus": 0.06951571460603478,
    },
    "mass_2_source": {
        "median": 0.9381816285468584,
        "error plus": 0.04352945419836962,
        "error minus": 0.0853175444182247,
    },
}

assign_vals_in_dict(mmax300_data, d, "percentilemmax300")
assign_vals_in_dict(mmaxlessthan300_data, d, "percentilemmaxlessthan300")

json.dump(d, open("OutlierTestMassiveEvent.json", "w"), indent=4)

