import h5py

def load_posteriors(fname, fixed_parameters=False, log_probs=False):
    """
    Loads samples from a PopModels H5CleanedPosteriorSamples file without
    adding any dependencies.

    By default only loads free parameters. Set `fixed_parameters=True` to
    resolve constants and duplicates as well.  Set `log_probs=True` to load the
    log posterior and prior.
    """
    with h5py.File(fname, "r") as f:
        # Get full set of parameter names, and then downselect to the variable
        # ones (i.e., that don't appear in 'constants' or 'duplicates') as those
        # correspond to the columns of the 'samples' dataset in the file.
        param_names = list(f["param_names"][()].astype(str))
        unfree_param_names = (
            set(f["constants"].attrs.keys()) | set(f["duplicates"].attrs.keys())
        )
        variable_names = [
            param_name for param_name in param_names
            if param_name not in unfree_param_names
        ]

        # Load free parameters
        result = dict(zip(variable_names, f["samples"][()].T))

        # Load constants and duplicates if requested.
        if fixed_parameters:
            # Resolve constants
            n_samples = f["samples"].shape[0]
            for param_name, value in f["constants"].attrs.items():
                result[param_name] = np.broadcast_to(n_samples, value)
            # Resolve duplicates
            for target_name, source_name in f["duplicates"].attrs.items():
                result[target_name] = result[source_name]

        # Load log posterior and prior if requested.
        if log_probs:
            result["posterior_log_probs"] = f["posterior_log_probs"][()]
            result["prior_log_probs"] = f["prior_log_probs"][()]                

    return result
