from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import pdb

from likelihoodfree.PDF import discrete_sample
from likelihoodfree.Simulator import lazyprop, SimulatorBase

class MoGSimulator(SimulatorBase):
    def __init__(self, bimodal=False, dim=1, seed=None):
        """Mixture of Gaussians simulator

        Parameters
        ----------
        bimodal : bool (default: False)
            If True, will add a second mode
        dim : int (default: 1)
            Dimensionality of the data as well as of theta
        seed : int or None (default: None)
            If set, randomness across runs is disabled

        Attributes
        ----------
        obs : observation
        prior_min
        prior_max
        """
        super().__init__(seed=seed)

        self.bimodal = bimodal
        self.dim = dim

        # true parameters of mixture
        self.n_components = 2
        self.alphas = np.array([0.5, 0.5])  # n_components,
        self.ms = np.stack([np.zeros((dim,)) for d in range(2)])  # n_components, dim
        self.Ss = np.stack([np.eye(dim) for d in range(2)])  # n_components, dim, dim
        self.Ss[1:] *= 5.0  # wide component

        # second peak
        if bimodal:
            raise ValueError('not implemented')

        # prior parameters
        self.prior_min = np.array([-10.0 for d in range(dim)])  # dim,
        self.prior_max = np.array([ 10.0 for d in range(dim)])  # dim,

    @property
    def obs(self):
        return np.zeros((1, self.dim))  # 1 x dim data

    @lazyprop
    def prior(self):
        return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                           seed=self.get_seed())

    @staticmethod
    def calc_summary_stats(x):
        return x

    def forward_model(self, theta, n_samples=1):
        """Given a mean parameter, simulates the likelihood.

        Parameters
        ----------
        theta : 1 x dim theta
        n_samples : int (default: 1)

        Returns
        -------
        n_samples x dim data
        """
        theta = np.asarray(theta)
        assert theta.ndim == 2, 'theta should be a 2d array'
        assert theta.shape[0] == 1, 'theta.shape[0] should be 1'
        assert theta.shape[1] == self.dim, 'theta.shape[1] should be dim long'

        sim_seed = self.rng.randint(0, 2**31)

        # list of n_components len with elements dim, (means)
        ms = [theta[0,:] for l in range(len(self.ms))]

        # list of n_components len with elements dim, dim (covariances)
        Ss = [self.Ss[i,:,:] for i in range(len(self.Ss))]

        mog = pdf.MoG(self.alphas, ms=ms, Ss=Ss, seed=self.get_seed())

        return mog.gen(n_samples=n_samples)

    def calc_posterior(self):
        """Calculates posterior analytically.

        Note that this assumes a flat improper prior.
        """
        # list of n_components len with elements dim, dim (covariances)
        Ss = [self.Ss[i,:,:] for i in range(len(self.Ss))]
        return pdf.MoG(a=self.alphas, ms=self.ms, Ss=Ss)
