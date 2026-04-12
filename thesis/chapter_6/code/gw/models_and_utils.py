import gwpopulation
gwpopulation.set_backend('jax')
xp = gwpopulation.utils.xp
import astropy
Planck15 = astropy.cosmology.Planck15
z_at_value = astropy.cosmology.z_at_value
import numpy as np
import jax
import jax_tqdm

class DagNabbitCustomLikelihood(gwpopulation.hyperpe.HyperparameterLikelihood):
    def __init__(
        self, 
        posteriors, 
        hyper_prior, 
        ln_evidences=None, 
        max_samples=1e100, 
        selection_function=lambda args: 1, 
        minimum_neff=0.,
        conversion_function=lambda args: (args, None), 
        cupy=False, 
        maximum_uncertainty=xp.inf, 
        posterior_length_corrections=None,
    ):
        super(DagNabbitCustomLikelihood, self).__init__(posteriors, hyper_prior, ln_evidences=ln_evidences, max_samples=max_samples, selection_function=selection_function, conversion_function=conversion_function, cupy=cupy, maximum_uncertainty=maximum_uncertainty)
        self.minimum_neff = minimum_neff
        if posterior_length_corrections is None:
            self.posterior_length_corrections = xp.ones(self.n_posteriors)
        else:
            self.posterior_length_corrections = xp.array(posterior_length_corrections)
        print(f'Using posterior length corrections of mean {xp.mean(self.posterior_length_corrections)} +/- {xp.std(self.posterior_length_corrections)}, minimum = {xp.min(self.posterior_length_corrections)}')
        # self.posterior_length_corrections should be fraction of length of posterior to number of samples
    
    def _compute_per_event_ln_bayes_factors(self, return_uncertainty=True):
        weights = self.hyper_prior.prob(self.data) / self.sampling_prior
        expectation = xp.mean(weights, axis=-1)
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            variance /= self.posterior_length_corrections
            neffs = 1 / (variance + (1/self.samples_per_posterior/self.posterior_length_corrections))
            # print(neffs)
            # print(xp.sum(variance))
            return xp.log(expectation * (neffs > self.minimum_neff)), variance
        else:
            return xp.log(expectation)
        


class CorrectedResamplingVT(gwpopulation.vt.ResamplingVT):
    def __init__(
        self, 
        model,
        data,
        original_posterior,
        n_events=np.inf,
        enforce_convergence=True,
        conversion_function=lambda args: (args, None),
    ):
        super(CorrectedResamplingVT, self).__init__(model=model, data=data, n_events=n_events, marginalize_uncertainty=False, enforce_convergence=enforce_convergence) #marginalize uncertainty is the Farr thing--we should not do this
        self.original_posterior = original_posterior
        self.mean_vt_weights = self._compute_weights_for_correction(conversion_function)

    def _compute_weights_for_correction(self, conversion_function):
        print('Computing weights for selection effects posterior correction factor')
        mean_vt_weights = xp.zeros_like(self.data['prior']) # (Nevents, NPE)
        n = self.original_posterior.shape[0]
        keys = self.original_posterior.keys()
        data = xp.array([self.original_posterior[k] for k in keys])
        from copy import copy
        model_copy = copy(self.model)
        @jax_tqdm.loop_tqdm(n, print_rate=1, tqdm_type='std')
        @jax.jit
        def weights_for_single_sample(ii, mean_vt_weights):
            parameters = {k: data[ik,ii] for ik, k in enumerate(keys)}
            parameters, added_keys = conversion_function(parameters)
            model_copy.parameters.update(parameters) # cannot alter state of object as this results in jax tracer leak
            weights = model_copy.prob(self.data) / self.data["prior"]
            mu = gwpopulation.utils.to_number(xp.sum(weights) / self.total_injections, float)

            return mean_vt_weights + weights / mu / self.original_posterior.shape[0]

        mean_vt_weights = jax.lax.fori_loop(0, n, weights_for_single_sample, mean_vt_weights)
        del model_copy
        print('...done')
        return mean_vt_weights

    def detection_efficiency(self, parameters):
        r"""
        Compute the expected fraction of detections given a set of injections
        and the variance in the Monte Carlo estimate.

        Parameters
        ----------
        parameters: dict
            The population parameters

        Returns
        -------
        mu: float
            The expected fracion of detections :math:`P_{\rm det}`.
        var: float
            The variance in the estimate of :math:`P_{\rm det}`.
        cov: float
            The average covariance in the log-estimator over the uncorrected posterior
        """
        self.model.parameters.update(parameters)
        weights = self.model.prob(self.data) / self.data["prior"]
        mu = gwpopulation.utils.to_number(
            xp.sum(weights) / self.total_injections, 
            float
        )
        var = gwpopulation.utils.to_number(
            xp.sum(weights**2) / self.total_injections**2
            - mu**2 / self.total_injections,
            float,
        )
        cov = gwpopulation.utils.to_number(
            xp.sum(weights * self.mean_vt_weights) / self.total_injections**2 / mu,
            float
        ) - 1 / self.total_injections 
        # really the minus 1 doesn't matter as it's a constant in the log-likelihood
        # we also approximate 1 / (total_injections - 1) as 1 / total_injections
        return mu, var, cov

    def __call__(self, parameters):
        if not self.marginalize_uncertainty:
            mu, var, cov = self.detection_efficiency(parameters)
            if self.enforce_convergence:
                _, correction = self.check_convergence(mu, var)
                mu += correction
            
            return mu, var, cov
        else:
            vt_factor = self.vt_factor(parameters)
            return vt_factor

class CorrectedPlusDagNabbitCustomLikelihood(gwpopulation.hyperpe.HyperparameterLikelihood):
    def __init__(
        self, 
        posteriors, 
        hyper_prior, 
        original_posterior,
        ln_evidences=None, 
        max_samples=1e100, 
        selection_function=lambda args: 1,
        minimum_neff=0., 
        conversion_function=lambda args: (args, None), 
        cupy=False, 
        maximum_uncertainty=xp.inf, 
        posterior_length_corrections=None,
    ):
        super(CorrectedPlusDagNabbitCustomLikelihood, self).__init__(posteriors, hyper_prior, ln_evidences=ln_evidences, max_samples=max_samples, selection_function=selection_function, conversion_function=conversion_function, cupy=cupy, maximum_uncertainty=maximum_uncertainty)
        self.minimum_neff = minimum_neff
        self.original_posterior = original_posterior
        if posterior_length_corrections is None:
            self.posterior_length_corrections = xp.ones(self.n_posteriors)
        else:
            self.posterior_length_corrections = xp.array(posterior_length_corrections)
        print(f'Using posterior length corrections of mean {xp.mean(self.posterior_length_corrections)} +/- {xp.std(self.posterior_length_corrections)}, minimum = {xp.min(self.posterior_length_corrections)}')
        # self.posterior_length_corrections should be fraction of length of posterior to number of samples
        self.mean_event_weights = self._compute_weights_for_correction()
        
    def _compute_weights_for_correction(self):
        print('Computing weights for posterior correction factor')
        mean_event_weights = xp.zeros_like(self.sampling_prior) # (Nevents, NPE)
        n = self.original_posterior.shape[0]
        keys = self.original_posterior.keys()
        data = xp.array([self.original_posterior[k] for k in keys])
        from copy import copy
        model_copy = copy(self.hyper_prior)
        @jax_tqdm.loop_tqdm(n, print_rate=1, tqdm_type='std')
        @jax.jit
        def weights_for_single_sample(ii, mean_event_weights):
            parameters = {k: data[ik,ii] for ik, k in enumerate(keys)}
            parameters, added_keys = self.conversion_function(parameters)
            model_copy.parameters.update(parameters) # cannot alter state of self object, as this results in a jax tracer leak
            weights = model_copy.prob(self.data) / self.sampling_prior
            expectation = xp.mean(weights, axis=-1)
            return mean_event_weights + weights / expectation[..., None] / self.original_posterior.shape[0]

        mean_event_weights = jax.lax.fori_loop(0, n, weights_for_single_sample, mean_event_weights)
        del model_copy
        print('...done')
        # print(mean_event_weights)
        return mean_event_weights

    def _compute_per_event_ln_bayes_factors(self, return_uncertainty=True, return_cov=False):
        weights = self.hyper_prior.prob(self.data) / self.sampling_prior
        expectation = xp.mean(weights, axis=-1)
        cross_moment = xp.mean(weights * self.mean_event_weights, axis=-1)
        cov = ((cross_moment / expectation) - 1) / self.samples_per_posterior
        cov /= self.posterior_length_corrections
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            variance /= self.posterior_length_corrections
            neffs = 1 / (variance + (1/self.samples_per_posterior/self.posterior_length_corrections))
            
            if return_cov:
                return xp.log(expectation * (neffs > self.minimum_neff)) + cov, variance, cov
            else:
                return xp.log(expectation * (neffs > self.minimum_neff)) + cov, variance
        else:
            return xp.log(expectation) + cov

    def _get_selection_factor(self, return_uncertainty=True):
        selection, variance, cov = self._selection_function_with_uncertainty()
        total_selection = -self.n_posteriors * xp.log(selection)
        if return_uncertainty:
            total_variance = self.n_posteriors**2 * xp.divide(
                variance, selection**2
            )
            likelihood_correction = -0.5 * (1+1/self.n_posteriors) * total_variance
            posterior_correction = self.n_posteriors**2 * cov
            return total_selection + likelihood_correction + posterior_correction, total_variance
        else:
            return total_selection

    def _selection_function_with_uncertainty(self):
        result = self.selection_function(self.parameters)
        if isinstance(result, tuple):
            selection, variance, cov = result
        else:
            selection = result
            variance = 0.0
            cov = 0.0
        return selection, variance, cov
    def generate_extra_statistics(self, sample):
        r"""
        Given an input sample, add extra statistics

        Adds:

        - :code:`ln_bf_idx`: :math:`\frac{\ln {\cal L}(d_{i} | \Lambda)}
          {\ln {\cal L}(d_{i} | \varnothing)}`
          for each of the events in the data
        - :code:`selection`: :math:`P_{\rm det}`
        - :code:`var_idx`, :code:`selection_variance`: the uncertainty in
          each Monte Carlo integral
        - :code:`total_variance`: the total variance in the likelihood

        .. note::

            The quantity :code:`selection_variance` is the variance in
            :code:`P_{\rm det}` and not the total variance from the contribution
            of the selection function to the likelihood.

        Parameters
        ----------
        sample: dict
            Input sample to compute the extra things for.

        Returns
        -------
        sample: dict
            The input dict, modified in place.
        """
        self.parameters.update(sample.copy())
        self.parameters, added_keys = self.conversion_function(self.parameters)
        self.hyper_prior.parameters.update(self.parameters)
        ln_ls, variances, cov = self._compute_per_event_ln_bayes_factors(
            return_uncertainty=True, return_cov=True
        )
        total_variance = sum(variances)
        total_cov_corr = sum(cov)
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = gwpopulation.utils.to_number(ln_ls[ii], float)
            sample[f"var_{ii}"] = gwpopulation.utils.to_number(variances[ii], float)
        selection, variance, cov_sel = self._selection_function_with_uncertainty()
        likelihood_correction = -0.5*self.n_posteriors*(self.n_posteriors+1)*xp.divide(
                variance, selection**2
        )
        posterior_correction = self.n_posteriors**2 * cov_sel
        sample["total_correction"] = gwpopulation.utils.to_number(total_cov_corr + likelihood_correction + posterior_correction, float)
        sample["likelihood_correction"] = gwpopulation.utils.to_number(likelihood_correction, float)
        variance /= selection**2
        selection_variance = variance * self.n_posteriors**2
        sample["selection"] = selection
        sample["selection_variance"] = variance
        total_variance += selection_variance
        sample["variance"] = gwpopulation.utils.to_number(total_variance, float)
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return sample

class OldLinearInterpolateModel(object):
    def __init__(self, nodes, minimum=0, maximum=1, parameters=['a_1', 'a_2']):
        self.nodes = xp.linspace(minimum, maximum, nodes)
        self.n_nodes = nodes
        self.parameters = parameters
        self.base = parameters[0].replace('_1', '')
        self.fkeys = [f"f{self.base}{ii}" for ii in range(self.n_nodes)]
    @property
    def variable_names(self):
        keys = self.fkeys
        return keys

    def __call__(self, dataset, **kwargs):
        fs = xp.array([kwargs[key] for key in self.fkeys])
        
        norm = self.norm(fs)
        probs = xp.ones_like(dataset[self.parameters[0]])
        for p in self.parameters:
            probs *= xp.exp(xp.interp(dataset[p], self.nodes, fs)) / norm
        
        return probs
    
    def norm(self, fs):
        dxs = xp.ediff1d(self.nodes)
        dfs = xp.ediff1d(fs)
        edfs = xp.ediff1d(xp.exp(fs))
        return xp.sum(edfs * dxs / dfs)
    
class SemiParametricLinearInterpolateModel(object):
    def __init__(self, nodes, alpha=1.67, beta=4.43, minimum=0, maximum=1, parameters=['a_1', 'a_2']):
        self.nodes = xp.linspace(minimum, maximum, nodes)
        self.n_nodes = nodes
        self.parameters = parameters
        self.base = parameters[0].replace('_1', '')
        self.fkeys = [f"f{self.base}{ii}" for ii in range(self.n_nodes)]
        self.xaxis = xp.linspace(minimum, maximum, 1000)
        self.alpha = alpha
        self.beta = beta

    @property
    def variable_names(self):
        keys = self.fkeys
        return keys
    def spin_model(self, spins, fs):
        
        p_scatter = xp.interp(spins, self.nodes, xp.exp(fs))
        p_base = gwpopulation.utils.beta_dist(
            spins, self.alpha, self.beta, scale=1
        )
        return p_base * p_scatter

    def normalized_spin_distribution(self, fs):
        spins = self.spin_model(self.xaxis, fs)
        norm = xp.trapezoid(spins, x=self.xaxis)

        return spins / norm

    def __call__(self, dataset, **kwargs):
        fs = xp.array([kwargs[key] for key in self.fkeys])

        norm = xp.trapezoid(self.spin_model(self.xaxis, fs), x=self.xaxis)

        probs = xp.ones_like(dataset[self.parameters[0]])
        for p in self.parameters:
            probs *= self.spin_model(dataset[p], fs) / norm
        
        return probs

class LinearInterpolateModel(object):
    def __init__(self, nodes, minimum=0, maximum=1, parameters=['a_1', 'a_2'], scale=lambda a: 10*a, exp=True, expm1=False):
        self.nodes = xp.linspace(minimum, maximum, nodes)
        self.n_nodes = nodes
        self.parameters = parameters
        self.base = parameters[0].replace('_1', '')
        self.fkeys = [f"f{self.base}{ii}" for ii in range(self.n_nodes-1)]
        self.scale = scale
        self.exp = exp
        self.expm1 = expm1

    @property
    def variable_names(self):
        keys = self.fkeys
        return keys

    def __call__(self, dataset, **kwargs):
        fs = xp.array([kwargs[key] for key in self.fkeys])
        fs = xp.append(fs, 1 - xp.sum(fs))
        fs = self.scale(fs)

        norm = self.norm(fs)
        # print(fs, norm)
        probs = xp.ones_like(dataset[self.parameters[0]])
        for p in self.parameters:
            if self.exp:
                probs *= xp.exp(xp.interp(dataset[p], self.nodes, fs)) / norm
            elif self.expm1:
                probs *= xp.expm1(xp.interp(dataset[p], self.nodes, fs)) / norm
            else:
                probs *= xp.interp(dataset[p], self.nodes, fs) / norm
        return probs
    
    def norm(self, fs):
        if self.exp:
            dxs = xp.ediff1d(self.nodes)
            dfs = xp.ediff1d(fs)
            edfs = xp.ediff1d(xp.exp(fs))
            return xp.sum(edfs * dxs / dfs)
        elif self.expm1:
            dxs = xp.ediff1d(self.nodes)
            dfs = xp.ediff1d(fs)
            edfs = xp.ediff1d(xp.exp(fs))
            return xp.sum(edfs * dxs / dfs) - xp.sum(dxs)
        else:
            dxs = xp.ediff1d(self.nodes)
            favs = (fs[1:] + fs[:-1]) / 2
            return xp.sum(favs * dxs)

class NoSecondarySmoothingMass(gwpopulation.models.mass.SinglePeakSmoothedMassDistribution):
    def p_q(self, dataset, beta, mmin, delta_m):
        # include delta_m because the parent class wants to give it, but don't use it
        p_q = gwpopulation.utils.powerlaw(dataset["mass_ratio"], beta, 1, 1e-3)
        return xp.nan_to_num(p_q)
    
def iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, mu_spin, sigma_spin):
    prior = (1 - xi_spin) / 4 + xi_spin * gwpopulation.utils.truncnorm(
        dataset["cos_tilt_1"], mu_spin, sigma_spin, 1, -1
    ) * gwpopulation.utils.truncnorm(dataset["cos_tilt_2"], mu_spin, sigma_spin, 1, -1)
    return prior

def g_q(q, n):
    return xp.exp(xp.abs(q - 0.1)**n - 0.9**n)

def salvo_spin_orientation_gaussian_isotropic(dataset, f_spin, n_spin, mu_spin, sigma_spin):
    q = dataset['mass_ratio']
    g = g_q(q, n_spin)

    xi_spin = f_spin * (g - g_q(0.1, n_spin)) / (g_q(1, n_spin) - g_q(0.1, n_spin))
    prior = (1 - xi_spin) / 4 + xi_spin * gwpopulation.utils.truncnorm(
        dataset["cos_tilt_1"], mu_spin, sigma_spin, 1, -1
    ) * gwpopulation.utils.truncnorm(dataset["cos_tilt_2"], mu_spin, sigma_spin, 1, -1)
    return prior
    
def flexible_spin_mag_model(nodes, exp=True, expm1=False):
    if expm1:
        return LinearInterpolateModel(nodes, exp=False, expm1=expm1)
    if exp:
        return LinearInterpolateModel(nodes, exp=exp, expm1=False, scale=lambda a: 20*a - 10)
    else:
        return LinearInterpolateModel(nodes, exp=False, expm1=False)
    

def salvos_spin_model(dataset, f_spin, n_spin, mu_spin, sigma_spin, amax, alpha_chi, beta_chi):

    prior = salvo_spin_orientation_gaussian_isotropic(
        dataset, f_spin, n_spin, mu_spin, sigma_spin
    ) * gwpopulation.models.spin.iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi)
    return prior

def mu_variable_spin_model(dataset, xi_spin, mu_spin, sigma_spin, amax, alpha_chi, beta_chi):

    prior = iid_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, mu_spin, sigma_spin
    ) * gwpopulation.models.spin.iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi)
    return prior