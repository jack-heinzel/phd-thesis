from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import corner

def get_params(file, params = ['mpp_2', 'sigpp_2']):
    popfile = PopulationResult(fname = file)
    return popfile.get_hyperparameter_samples(hyperparameters=params), params

def downselect_popfile(popfile, outfile, sel, params = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']):
    hypersamples = popfile.get_hyperparameter_samples(hyper_sample_idx = sel)
    hyperparams = popfile.get_metadata(field='hyperparameters')

    new_popfile = PopulationResult(fname = outfile, hyperparameters = hyperparams)
    new_popfile.set_hyperparameter_samples(hyperparameter_samples=hypersamples)
    for param in params:
        x, px = popfile.get_rates_on_grids(grid_key = param, hyper_sample_idx = sel)
        new_popfile.set_rates_on_grids(grid_key = param, grid_params = param, positions = x, rates = px, overwrite = True)


def main():

    file = '/home/jack.heinzel/public_html/o4a_population_paper/review/o4a-pop-default/code/2pl2pk/init/result/gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    data, params = get_params(file, params = ['mpp_2', 'sigpp_2'])
    initial_clusters = [[33, 2], [31, 4]]
    km = KMeans(n_clusters=2, init=initial_clusters)
    km_results = km.fit(data)
    labels = km_results.labels_

    fig = corner.corner(data, labels = params, alpha = 0.2);
    corner.corner(data[labels==0], fig = fig, labels = params, color = 'r');
    corner.corner(data[labels==1], fig = fig, labels = params, color = 'b');
    plt.show()
    plt.savefig('cluster_corner.pdf')
    plt.close()

    default_popfile = PopulationResult(fname = file)
    path = '/home/jaxen.godfrey/o4a-astro-dist-clean/o4a-astrodist/analyses/default_bbh/popsummary/dec4'
    name = 'gwtc4_2pl2pk_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_gaussian_isotropic_redshift_PowerLawRedshift_popsummary_result.h5'

    file1 = 'dominant_mode_' + name
    dom_sel = np.argwhere(labels == 0).T[0]
    file2 = 'subdominant_mode_' + name
    sub_sel = np.argwhere(labels == 1).T[0]

    downselect_popfile(default_popfile, path + '/' + file1, dom_sel)
    downselect_popfile(default_popfile, path + '/' + file2, sub_sel)



if __name__ == '__main__':
    main()