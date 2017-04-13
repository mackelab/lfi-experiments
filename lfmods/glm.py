from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.PDF as pdf
import likelihoodfree.io as io
import numpy as np
import os
import pdb
import shelve
import time

from likelihoodfree.Simulator import lazyprop, SimulatorBase
from scipy import stats as spstats
from tqdm import tqdm

class GLMSimulator(SimulatorBase):
    def __init__(self,
                 cached_pilot=True,
                 cached_sims=True,
                 dir_cache='results/glm/data/',
                 duration=100,
                 pilot_samples=1000,
                 seed=None,
                 summary_stats=1,
                 verbose=False):
        """GLM simulator

        Parameters
        ----------
        cached_pilot : bool
            If True, will try to use cached pilot data (only works if seed is set)
        cached_sims : bool
            If True, and iff seed is specified, will cache simulations (only works if seed is set)
        dir_cache : str
            Sets dir for cache
        duration : int
            Duration of traces in ms
        pilot_samples : bool
            Number of pilot samples to generate, set to 0 to disable normalize
            of summary statistics
        seed : int or None
            If set, randomness across runs is disabled
        summary_stats : int
            Serves as a switch to change between different ways to calculate
            summary statistics:
                0 : no summary stats (returns identity)
                1 : sufficient statistics
        verbose : bool
            Print extra info or not

        Attributes
        ----------
        obs : observation
            x0 summary statistics
        prior_min
        prior_max
        """
        super().__init__(seed=seed)

        self.cached_pilot = cached_pilot
        self.cached_sims = cached_sims
        self.dir_cache = dir_cache
        self.duration = duration
        self.pilot_samples = pilot_samples
        self.summary_stats = summary_stats
        self.verbose = verbose

        import lfmods.glm_bm as bm
        self.bm = bm

        if pilot_samples > 0:
            self.pilot_norm = True
        else:
            self.pilot_norm = False

        if self.seed is None:
            self.cached_pilot = False
            self.cached_sims = False

        # true parameters: (b0, h) = offset, temporal filter
        b0=-2.
        M = 9   # Length of the filter
        a = 0.5  # inverse time constant of the filter
        tau = np.linspace(1, M, M) # Support for the filter
        h = (a * tau)**3 * np.exp(-a * tau) # Temporal filter
        true_params = np.concatenate((np.array([b0]),h))

        self.true_params = np.concatenate((np.array([b0]),h))

        self.labels_params = ['b0']
        for i in range(M):
            self.labels_params.append('h'+str(i+1))
        self.n_params = len(self.true_params)

        # parameters that globally govern the simulations
        self.dt = 1
        self.t = np.arange(0, self.duration, self.dt)

        # input: gaussian white noise N(0, 1)
        self.I = self.rng.randn(len(self.t))
        self.I_obs = self.I.copy()

        self.max_n_steps = 10000

        # summary statistics
        if self.summary_stats == 0:  # no summary (whole time-series)
            self.signal_ds = 1
            self.n_summary_stats = len(self.t[::self.signal_ds])  # quick hack: downsampling
            if self.pilot_norm:
                print('Warning: pilot_norm is True, while using identity summary stats')
        elif self.summary_stats == 1:  # sufficient statistics
            self.n_summary_stats = self.n_params
        else:
            raise ValueError('summary_stats invalid')

    @lazyprop
    def obs(self):
        # generate observed data from simulation
        glm = self.bm.GLM(self.true_params.reshape(1, -1),
                        seed=self.gen_newseed())
        states = glm.sim_time(self.t, self.I).reshape(1, -1, 1)
        stats = self.calc_summary_stats(states)

        return stats

    @lazyprop
    def prior(self):
        range_lower = 0.5*self.true_params
        range_upper = 1.5*self.true_params

        # Smoothing prior on h; N(0, 1) on b0. Smoothness encouraged by
        # penalyzing 2nd order differences of filter elements
        D = np.diag(np.ones(self.n_params-1)) - np.diag(np.ones(self.n_params-2), -1)
        F = np.dot(D, D)
        # Binv is block diagonal
        Binv = np.zeros(shape=(self.n_params,self.n_params))
        Binv[0,0] = 1    # offset (b0)
        Binv[1:,1:] = np.dot(F.T, F) # filter (h)

        prior_mn = self.true_params*0.
        prior_prec = Binv
        return pdf.Gaussian(m=prior_mn, P=prior_prec, seed=self.gen_newseed())

    @lazyprop
    def pilot_means(self):
        if self.pilot_norm:
            if not hasattr(self, '_pilot_means'):
                self._pilot_means, self._pilot_stds = self.pilot_run()
                return self._pilot_means
            else:
                return self._pilot_means
        else:
            return 0.

    @lazyprop
    def pilot_stds(self):
        if self.pilot_norm:
            if not hasattr(self, '_pilot_stds'):
                self._pilot_means, self._pilot_stds = self.pilot_run()
                return self._pilot_stds
            else:
                return self._pilot_stds
        else:
            return 1.

    def _hash(self, *args):
        """Hashing function to generate key for shelve dicts

        Hashing will be based on args passed to function, plus a set
        of attributes describing global simulator settings.
        """
        key = [self.seed, self.duration, self.t[-1], \
               np.sum(self.I), self.max_n_steps, self.summary_stats]

        for arg in args:
            key.append(arg)

        return str(hash(tuple(key)))

    def forward_model(self, theta, n_samples=1):
        """Runs the model

        Parameters
        ----------
        theta : dim theta
        n_samples : int
            If greater than 1, generate multiple samples given theta

        Returns
        -------
        n_samples (=1) x tracelength x features (=1)
        """
        assert theta.ndim == 1, 'theta.ndim must be 1'
        assert theta.shape[0] == self.n_params, 'theta.shape[0] must be dim theta long'
        assert n_samples == 1, 'assert n_samples > 1 not supported'

        if self.cached_sims:
            cached_sims_path = self.dir_cache + 'cached_sims_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_sims_path)

        glm_seed = self.gen_newseed()
        key = self._hash(glm_seed, np.sum(theta))

        if self.cached_sims and key in d:
            states = d[key]
        else:
            glm = self.bm.GLM(theta.reshape(1, -1), seed=glm_seed)
            states = glm.sim_time(self.t, self.I, max_n_steps=self.max_n_steps)

        if self.cached_sims:
            d[key] = states
            d.close()

        return states.reshape(n_samples, -1, 1)

    def pilot_run(self):
        """Pilot run

        Runs a number of simulations, and it calculates and saves the mean and
        standard deviation of the summary statistics across simulations.
        """
        stats = []

        if self.cached_pilot:
            cached_pilot_path = self.dir_cache + 'cached_pilot_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_pilot_path)

        for i in tqdm(range(self.pilot_samples)):
            params = self.sim_prior()
            glm_seed = self.gen_newseed()

            key = self._hash(glm_seed, np.sum(params))

            if self.cached_pilot and key in d:
                states = d[key]
            else:
                glm = self.bm.GLM(params.reshape(1, -1), seed=glm_seed)
                states = glm.sim_time(self.t, self.I, max_n_steps=self.max_n_steps)
                states = states.reshape(1, -1, 1)

            stats.append(self.calc_summary_stats(states, skip_norm=True))

            if self.cached_pilot and key not in d:
                d[key] = states

        if self.cached_pilot:
            d.close()

        stats = np.array(stats)
        means = np.mean(stats, axis=0).reshape(-1)
        stds = np.std(stats, axis=0, ddof=1).reshape(-1)

        return means, stds

    def calc_summary_stats(self,
                           states,
                           skip_norm=False):
        """Calculate summary stats

        Parameters
        ----------
        states : n_samples (=1) x timesteps x features (=1)
        skip_norm : bool
            If True, will skip pilot run normalization disregarding self.pilot_norm setting

        Return
        ------
        n_samples (=1) x dim summary stats

        Notes
        -----
        Output will be based on summary_stats property
        """
        assert states.ndim == 3, 'input must be 3d'
        assert states.shape[0] == 1, 'n_samples dim must be 1'
        assert states.shape[2] == 1, 'feature dim must be 1'

        states = states[0, :, :]

        # no summary stats?
        if self.summary_stats == 0:
            stats = np.hstack((states[::self.signal_ds].reshape(-1,1),
                              self.I[::self.signal_ds].reshape(-1,1)))
            return stats[np.newaxis, :]  # 1 x time series x 2 features

        x = states.reshape(-1)
        len_x = x.shape[0]

        # sufficient statistics
        if self.summary_stats == 1:

            n_xcorr = self.n_summary_stats-1
            sta = np.correlate(x,self.I_obs,'full')[len_x-1:len_x+n_xcorr-1]

            sum_stats_vec = np.concatenate( (np.array([np.sum(x)]),sta) )

        else:
            raise ValueError('summary_stats is invalid')

        if np.isnan(sum_stats_vec).any():
            print(sum_stats_vec)
            import pdb; pdb.set_trace()

        # pilot run normalization
        if not skip_norm and self.pilot_norm:
            sum_stats_vec -= self.pilot_means
            sum_stats_vec /= self.pilot_stds

        return sum_stats_vec.reshape(1, -1)
