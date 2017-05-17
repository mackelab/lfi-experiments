from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import pdb

from likelihoodfree.PDF import discrete_sample
from likelihoodfree.Simulator import lazyprop, SimulatorBase

class SqrtSimulator(SimulatorBase):
    def __init__(self,
                 dim=1,
                 n_summary=1,
                 noise_cov=0.1,
                 prior_uniform=False,
                 prior_abslim=10.0,
                 prior_cov=1.0,
                 prior_mean=5.0,
                 seed=None,
                 seed_obs=None,
                 true_mean=2.230):
        """Square-root simulator

        Parameters
        ----------
        dim : int
            Number of dimensions
        n_summary : int
            Number of samples per draw entering summary statistics
        noise_cov : float
            Covariance of noise on observations
        prior_uniform : bool
            Switch between Gaussian and uniform prior
        prior_abslim : float
            Uniform prior: absolute limits
        prior_cov : float
            Gaussian prior: covariance
        prior_mean : float
            Gaussian prior: mean
        seed : int or None
            If set, randomness across runs is disabled
        seed_obs : int or None
            If set, randomness of obs is controlled independently of seed.
            Important: If only `seed` is set, `obs` is not random
        true_mean : float
            Location of mean

        Attributes
        ----------
        obs : observation
        prior_log : bool
            whether or not prior is in log space
        """
        super().__init__(prior_uniform=False, seed=seed)  # prior_uniform is overwritten below
        self.seed_obs = seed_obs

        self.dim = dim
        self.n_summary = n_summary
        self.prior_uniform = prior_uniform

        # for gaussian prior
        self.prior_cov = prior_cov * np.eye(dim)
        self.prior_mean = prior_mean * np.ones((dim,))  # dim,

        # for uniform prior
        self.prior_min = np.array([-prior_abslim for d in range(dim)])  # dim,
        self.prior_max = np.array([+prior_abslim for d in range(dim)])  # dim,

        # model parameters
        self.true_params = true_mean*np.ones((dim,))
        self.noise_cov = noise_cov*np.eye(dim)

    @lazyprop
    def obs(self):
        # generate observation
        if self.seed_obs is None:
            seed = self.gen_newseed()
        else:
            seed = self.seed_obs

        self.x0_distrib = pdf.Gaussian(m=np.sqrt(self.true_params), S=self.noise_cov,
                                       seed=seed)  # *1./N
        self.x0_sample = self.x0_distrib.gen(self.n_summary)
        self.x0_sample_mean = np.mean(self.x0_sample, axis=0)

        return 2.230 * np.ones(self.n_summary).reshape(1,-1)
        return np.array([self.x0_sample_mean]).reshape(1, -1)  # 1 x dim summary stats

    @lazyprop
    def prior(self):
        if not self.prior_uniform:
            return pdf.Gaussian(m=self.prior_mean, S=self.prior_cov,
                                seed=self.gen_newseed())
        else:
            return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                               seed=self.gen_newseed())

    def posterior(self, x_obs1):
        """Calculates posterior analytically
        """
        if not self.prior_uniform:
            posterior_cov = np.linalg.inv(np.linalg.inv(self.prior_cov)+self.n_summary*np.linalg.inv(self.noise_cov))
            posterior_mu = np.dot(posterior_cov, (self.n_summary*np.dot(np.linalg.inv(self.noise_cov), x_obs1.reshape(-1))+np.dot(np.linalg.inv(self.prior_cov), self.prior_mean)))
            return pdf.Gaussian(m=posterior_mu, S=posterior_cov)
        else:
            posterior_mu = x_obs1.reshape(-1)
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

        samples = pdf.Gaussian(m=np.sqrt(theta), S=self.noise_cov,
                               seed=self.gen_newseed()).gen(self.n_summary)
        return samples[np.newaxis, :, :]
