from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import pdb

from likelihoodfree.PDF import discrete_sample
from likelihoodfree.Simulator import lazyprop, SimulatorBase

class GaussSimulator(SimulatorBase):
    def __init__(self,
                 dim=1,
                 n_summary=100,
                 seed=None):
        """Gaussian simulator

        Parameters
        ----------
        dim : int
            Number of dimensions
        n_summary : int
            Number of samples per draw entering summary statistics
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
        super().__init__(prior_uniform=False, seed=seed)

        # parameters
        self.true_params = 1.5*np.ones((dim,))
        self.noise_cov = 0.1*np.eye(dim)

        # attributes
        self.dim = dim
        self.n_summary = n_summary

    @lazyprop
    def obs(self):
        self.x0_distrib = pdf.Gaussian(m=self.true_params, S=self.noise_cov,
                                       seed=self.gen_newseed())  # *1./N
        self.x0_sample = self.x0_distrib.gen(self.n_summary)
        self.x0_sample_mean = np.mean(self.x0_sample)
        return np.array([self.x0_sample_mean]).reshape(1, -1)  # 1 x dim summary stats

    @lazyprop
    def prior(self):
        self.prior_cov = 20.*np.eye(self.dim)
        self.prior_mu = 0.
        return pdf.Gaussian(m=self.prior_mu, S=self.prior_cov,
                            seed=self.gen_newseed())

    @lazyprop
    def posterior(self):
        """Calculates posterior analytically

        Note that this assumes a Gaussian prior
        """
        posterior_cov = (self.noise_cov*self.prior_cov)/(self.n_summary*self.prior_cov+self.noise_cov)
        posterior_mu = posterior_cov*(self.prior_mu/self.prior_cov + self.n_summary*self.x0_sample_mean/self.noise_cov)

        return pdf.Gaussian(m=posterior_mu, S=posterior_cov)

    @staticmethod
    def calc_summary_stats(x):
        """Calculate summary statistics

        Computes the mean

        Parameters
        ----------
        x : n_samples x dim theta

        Return
        ------
        n_samples x dim summary stats
        """
        return np.mean(x, axis=0)

    def forward_model(self, theta, n_samples=1):
        """Given a mean parameter, simulates the likelihood

        Parameters
        ----------
        theta : 1 x dim theta
        n_samples : int
            If greater than 1, generate multiple samples given theta

        Returns
        -------
        n_samples x dim data
        """
        assert theta.ndim == 1, 'theta.ndim must be 1'
        assert theta.shape[0] == self.dim, 'theta.shape[0] must be dim theta long'

        samples = pdf.Gaussian(m=theta, S=self.noise_cov,
                               seed=self.gen_newseed()).gen(n_samples)
        return samples
