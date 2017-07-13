from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import pdb

from likelihoodfree.PDF import discrete_sample
from likelihoodfree.Simulator import lazyprop, SimulatorBase

standardcoeffs = 0.5 * np.array([1, -2, -2, 1])

def gauss1D(x, m):
    return ((2 * np.pi) ** 0.5) * np.exp(-0.5 * np.linalg.norm(x - m) ** 2)

class CustomPDF:
    def __init__(self, xlist, ylist):
        self.xlist = xlist
        self.ylist = ylist

    def eval(self, pts, log=False):
        ret = np.interp(pts, self.xlist, self.ylist, left=0, right=0)
        if log == True:
            ret = np.log(ret)

        return ret

def polynomial(coeffs):
    def ret(x):
        return np.dot(coeffs, x ** range(len(coeffs)))

    return ret

class CustomSimulator(SimulatorBase):
    def __init__(self,
                 prior_abslim=4.0,
                 dim=1,
                 coeffs = standardcoeffs,
                 obs_data=[0.0],
                 seed=None):
        """Mixture of Gaussians simulator

        Parameters
        ----------
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

        # prior parameters
        self.dim = dim
        assert(self.dim == 1)

        self.prior_min = np.array([-prior_abslim])
        self.prior_max = np.array([+prior_abslim])

        self.meanfunc = polynomial(coeffs)
        self.obs_data = np.asarray(obs_data)

    @lazyprop
    def obs(self):
        return np.reshape(self.obs_data, (1,-1))

    @lazyprop
    def prior(self):
        return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                           seed=self.gen_newseed())

    @lazyprop
    def posterior(self):
        return self.get_posterior(self.obs)

    def get_posterior(self, p):
        """Calculates posterior.

        Returns
        -------
        PDF

        Note that this assumes a flat improper prior.
        """
        if p == None:
            p = self.obs

        stepsize = (self.prior_max - self.prior_min) / 100
        xlist = np.arange(self.prior_min,self.prior_max, stepsize)
        ylist = np.array([ gauss1D(p, self.meanfunc(x)) for x in xlist ])
        ylist /= np.sum(ylist) * stepsize

        return CustomPDF(xlist,ylist)



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

        mean = self.meanfunc(theta)
        gauss = pdf.Gaussian([mean], [[1]], seed=self.gen_newseed())

        samples = gauss.gen(n_samples=n_samples)

        return samples
