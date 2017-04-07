from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.PDF as pdf
import likelihoodfree.io as io
import numpy as np

from likelihoodfree.Simulator import SimulatorBase

class GaussSimulator(SimulatorBase):
    def __init__(self, ndim=1, nsamples=100, seed=None):
        """Gaussian simulator

        Parameters
        ----------
        ndim : Number of dimensions
        seed : int or None (default: None)
            If set, randomness across runs is disabled

        Attributes
        ----------
        obs : observation
        prior_log : bool
            whether or not prior is in log space
        prior_min
        prior_max
        """
        # call init of base class
        super().__init__(seed=seed)

        # parameters
        self.true_params = 1.5*np.ones((ndim,))
        self.noise_cov = 0.1*np.eye(ndim)

        # attributes
        self.ndim = ndim
        self.nsamples = nsamples

        self.prior_log = False
        self.prior_uniform = False
        self.prior_cov = 20.*np.eye(ndim)
        self.prior_mu = 0.
        self.prior_gauss = pdf.Gaussian(m=self.prior_mu, S=self.prior_cov, seed=seed)
        self.prior_min = -np.inf
        self.prior_max = +np.inf

        seed = self.rng.randint(0, 2**31)
        self.x0_dist = pdf.Gaussian(m=self.true_params, S=self.noise_cov, seed=seed)  # *1./N
        self.x0_sample = self.x0_dist.gen(self.nsamples)
        self.x0_sample_mean = np.mean(self.x0_sample)

        self.obs = np.array([self.x0_sample_mean])

        self.forward_model = lambda x: self.sim_likelihood(x, n_samples=self.nsamples)
        self.calc_summary_stats = lambda x: np.mean(x)

    def sim_prior(self, num_sims=1):
        """Simulate from prior

        Assumes a uniform prior (in log domain or not)

        Parameters
        ----------
        num_sims : int

        Returns
        -------
        output : if num_sims is 1: n_params, else: num_sims x n_params
            If prior was in log domain, we inverse transform back to normal
        """
        if self.prior_uniform:
            z = self.rng.rand(len(self.true_params)) if num_sims == 1 else self.rng.rand(num_sims, len(self.true_params))
            return self.param_invtransform((self.prior_max - self.prior_min) * z + self.prior_min)
        else:
            z = self.prior_gauss.gen(num_sims).reshape(-1)
            if self.ndim > 1:
                raise ValueError('todo: rethink reshape above')
            return self.param_invtransform(z)

    def sim_likelihood(self, ms, n_samples=1):
        """Given a mean parameter, simulates the likelihood."""

        seed = self.rng.randint(0, 2**31)
        xs = pdf.Gaussian(m=ms, S=self.noise_cov, seed=seed).gen(n_samples)

        return xs[0] if n_samples == 1 else xs

    def sim_joint(self, n_samples=1):
        """Simulates (m,x) pairs from joint."""

        ms = self.sim_prior(n_samples)
        xs = self.sim_likelihood(ms)

        return ms, xs

    def calc_posterior(self):
        """Calculates posterior analytically. Note that this assumes a flat improper prior."""

        posterior_cov = (self.noise_cov*self.prior_cov)/(self.nsamples*self.prior_cov+self.noise_cov)
        posterior_mu = posterior_cov*(self.prior_mu/self.prior_cov + self.nsamples*self.x0_sample_mean/self.noise_cov)

        return pdf.Gaussian(m=posterior_mu, S=posterior_cov)
