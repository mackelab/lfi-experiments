from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import pdb

from likelihoodfree.PDF import discrete_sample
from likelihoodfree.Simulator import lazyprop, SimulatorBase

class MoGSimulator(SimulatorBase):
    def __init__(self,
                 bimodal=False,
                 dim=1,
                 prior_abslim=10.0,
                 seed=None):
        """Mixture of Gaussians simulator

        Parameters
        ----------
        bimodal : bool
            If True, will add a second mode
        dim : int
            Dimensionality of the data as well as of theta
        prior_abslim : float
            Uniform prior: absolute limits
        seed : int or None
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
        if not self.bimodal:
            self.Ss[1,:,:] *= 0.1**2  # std narrow component
        else:
            #self.Ss[1,:,:] *= 0.5**2  # std narrow component
            pass

        # prior parameters
        self.prior_min = np.array([-prior_abslim for d in range(dim)])  # dim,
        self.prior_max = np.array([+prior_abslim for d in range(dim)])  # dim,

    @lazyprop
    def obs(self):
        if self.bimodal:
            return 5*np.ones((1, self.dim))  # 1 x dim data
        else:
            return 0*np.ones((1, self.dim))  # 1 x dim data

    @lazyprop
    def prior(self):
        return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                           seed=self.gen_newseed())

    @lazyprop
    def posterior(self):
        """Calculates posterior analytically.

        Returns
        -------
        PDF

        Note that this assumes a flat improper prior.
        """
        Ss = [self.Ss[i,:,:] for i in range(len(self.Ss))]

        if not self.bimodal:
            # list of n_components len with elements dim, dim (covariances)
            #Ss = [self.Ss[i,:,:] for i in range(len(self.Ss))]
            return pdf.MoG(a=self.alphas, ms=self.ms, Ss=Ss)
        else:
            ms = self.ms
            ms[0] = +1*self.obs
            ms[1] = -1*self.obs
            return pdf.MoG(a=self.alphas, ms=self.ms, Ss=Ss)

    @staticmethod
    def calc_summary_stats(x):
        """Calculate summary statistics

        Returns the identity

        Parameters
        ----------
        x : n_samples x dim data

        Returns
        -------
        n_samples x dim summary stats
        """
        return x

    def forward_model(self, theta, n_samples=1):
        """Given a mean parameter, simulates the likelihood

        Parameters
        ----------
        theta : dim theta
        n_samples : int
            If greater than 1, generate multiple samples given theta

        Returns
        -------
        n_samples x dim data
        """
        assert theta.ndim == 1, 'theta.ndim must be 1'
        assert theta.shape[0] == self.dim, 'theta.shape[0] must be dim theta long'

        # list of n_components len with elements dim, (means)
        ms = [theta for l in range(len(self.ms))]

        if self.bimodal:
            ms[0] = +1*theta
            ms[1] = -1*theta

        # list of n_components len with elements dim, dim (covariances)
        Ss = [self.Ss[i,:,:] for i in range(len(self.Ss))]

        mog = pdf.MoG(self.alphas, ms=ms, Ss=Ss, seed=self.gen_newseed())
        samples = mog.gen(n_samples=n_samples)

        return samples
