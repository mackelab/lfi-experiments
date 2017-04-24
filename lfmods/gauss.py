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
                 prior_uniform=False,
                 seed=None,
                 true_mean=1.5):
        """Gaussian simulator

        Parameters
        ----------
        dim : int
            Number of dimensions
        n_summary : int
            Number of samples per draw entering summary statistics
        prior_uniform : bool
            Switch between Gaussian and uniform prior
        seed : int or None (default: None)
            If set, randomness across runs is disabled
        true_mean : float
            Location of mean

        Attributes
        ----------
        obs : observation
        prior_log : bool
            whether or not prior is in log space
        prior_min
        prior_max
        """
        super().__init__(prior_uniform=False, seed=seed)

        # attributes
        self.dim = dim
        self.n_summary = n_summary
        self.prior_uniform = prior_uniform

        # model parameters
        self.true_params = true_mean*np.ones((dim,))
        self.noise_cov = 0.1*np.eye(dim)

        # gaussian prior parameters
        self.prior_cov = 20.*np.eye(self.dim)
        self.prior_mu = 0.*np.ones((self.dim,))

        # uniform prior parameters
        self.prior_min = np.array([-10.0 for d in range(dim)])  # dim,
        self.prior_max = np.array([ 10.0 for d in range(dim)])  # dim,

        # generate observation
        self.x0_distrib = pdf.Gaussian(m=self.true_params, S=self.noise_cov,
                                       seed=self.gen_newseed())  # *1./N
        self.x0_sample = self.x0_distrib.gen(self.n_summary)
        self.x0_sample_mean = np.mean(self.x0_sample, axis=0)

    @lazyprop
    def obs(self):
        return np.array([self.x0_sample_mean]).reshape(1, -1)  # 1 x dim summary stats

    @lazyprop
    def prior(self):
        if not self.prior_uniform:
            return pdf.Gaussian(m=self.prior_mu, S=self.prior_cov,
                                seed=self.gen_newseed())
        else:
            return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                               seed=self.gen_newseed())

    @lazyprop
    def posterior(self):
        """Calculates posterior analytically
        """
        if not self.prior_uniform:
            posterior_cov = np.linalg.inv(np.linalg.inv(self.prior_cov)+self.n_summary*np.linalg.inv(self.noise_cov))
            posterior_mu = np.dot(posterior_cov, (self.n_summary*np.dot(np.linalg.inv(self.noise_cov), self.x0_sample_mean)+np.dot(np.linalg.inv(self.prior_cov), self.prior_mu)))
            return pdf.Gaussian(m=posterior_mu, S=posterior_cov)
        else:
            posterior_mu = self.x0_sample_mean
            posterior_cov = self.noise_cov/self.n_summary
            return pdf.Gaussian(m=posterior_mu, S=posterior_cov)

    @staticmethod
    def calc_summary_stats(x):
        """Calculate summary statistics

        Computes the mean

        Parameters
        ----------
        x : n_samples x dim data x features (=self.dim)

        Returns
        -------
        n_samples x dim summary stats (=1)
        """
        return np.mean(x, axis=1)

    def forward_model(self, theta, n_samples=1):
        """Given a mean parameter, simulates the likelihood

        # TODO: use n_samples parameter to generate multiple samples,
        will be clearer

        Parameters
        ----------
        theta : 1 x dim theta
        n_samples : int
            If greater than 1, generate multiple samples given theta

        Returns
        -------
        n_samples(=1) x dim data x features (= self.dim)
        """
        assert theta.ndim == 1, 'theta.ndim must be 1'
        assert theta.shape[0] == self.dim, 'theta.shape[0] must be dim theta long'
        assert n_samples == 1, 'assert n_samples > 1 not supported'

        samples = pdf.Gaussian(m=theta, S=self.noise_cov,
                               seed=self.gen_newseed()).gen(self.n_summary)
        return samples[np.newaxis, :, :]
