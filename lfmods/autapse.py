from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lfmods.autapse_bm as bm
import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import os
import pdb
import shelve
import time

from likelihoodfree.Simulator import lazyprop, SimulatorBase
from scipy import stats as spstats
from tqdm import tqdm

class AutapseSimulator(SimulatorBase):
    def __init__(self,
                 cached_pilot=False,
                 cached_sims=False,
                 dir_cache='results/autapse/data/',
                 duration=120,
                 pilot_samples=0,
                 prior_uniform=True,
                 seed=None,
                 seed_obs=None,
                 summary_stats=2,
                 verbose=False):
        """Autapse simulator

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
        prior_uniform : bool
            Flag to switch between Gaussian and uniform prior
        seed : int or None
            If set, randomness across runs is disabled
        seed_obs : int or None
            If set, randomness of obs is controlled independently of seed.
            Important: If only `seed` is set, `obs` is not random
        summary_stats : int
            Serves as a switch to change between different ways to calculate
            summary statistics:
                0 : no summary stats (returns identity)
                1 : moments
        verbose : bool
            Print extra info or not

        Attributes
        ----------
        obs : observation
            x0 summary statistics
        prior_min
        prior_max
        """
        super().__init__(prior_uniform=prior_uniform,seed=seed)
        self.seed_obs = seed_obs

        self.cached_pilot = cached_pilot
        self.cached_sims = cached_sims
        self.dir_cache = dir_cache
        self.duration = duration
        self.pilot_samples = pilot_samples
        self.summary_stats = summary_stats
        self.verbose = verbose

        self.bm = bm

        if pilot_samples > 0:
            self.pilot_norm = True
        else:
            self.pilot_norm = False

        if self.seed is None:
            self.cached_pilot = False
            self.cached_sims = False

        # true parameters
        self.true_params = np.array([0.95,1])
        self.labels_params = ['J','tau_x']
        self.labels_params = self.labels_params[0:len(self.true_params)]
        self.n_params = len(self.true_params)

        # parameters that globally govern the simulations
        self.init = [0]  # =x[0]
        self.t_offset = 0.
        self.dt = 0.01  # 10 to 100 times smaller than tau
        self.t = np.arange(0, self.duration+self.dt, self.dt)
        self.t_on = 10
        self.t_off = self.duration - self.t_on

        # external current
        self.curr_level = 1
        step_current = np.zeros_like(self.t)
        step_current[int(np.round(self.t_on/self.dt)):int(np.round(self.t_off/self.dt))] = self.curr_level
        self.I = step_current
        self.I_obs = self.I.copy()

        self.max_n_steps = 10000

        # summary statistics
        if self.summary_stats == 0:  # no summary (whole time-series)
            self.signal_ds = 20
            self.n_summary_stats = len(self.t[::self.signal_ds])  # quick hack: downsampling
            self.labels_sum_stats = []
            if self.pilot_norm:
                print('Warning: pilot_norm is True, while using identity summary stats')
        elif self.summary_stats == 1:  # moments
            self.n_xcorr = 5
            self.n_summary_stats = self.n_xcorr + 2
        elif self.summary_stats == 2:  # mean
            pass
        else:
            raise ValueError('summary_stats invalid')

    @lazyprop
    def obs(self):
        # seed for observed data
        if self.seed_obs is None:
            seed = self.gen_newseed()
        else:
            seed = self.seed_obs

        # generate observed data from simulation
        autapse = self.bm.Autapse(self.init, self.true_params.reshape(1, -1),
                        seed=seed)
        states = autapse.sim_time(self.dt, self.t, self.I).reshape(1, -1, 1)
        stats = self.calc_summary_stats(states)

        self.obs_trace = states
        return stats

    @lazyprop
    def prior(self):
        range_lower = np.array([0.1,-1.])  #0.5*self.true_params
        range_upper = np.array([2.0,2.5])  #3.*self.true_params

        if self.prior_uniform:
            self.prior_min = range_lower
            self.prior_max = range_upper
            return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                               seed=self.gen_newseed())
        else:
            prior_mn = 0.*self.true_params
            prior_cov = np.diag(0.1*(range_upper - range_lower))
            return pdf.Gaussian(m=prior_mn, S=prior_cov, seed=self.gen_newseed())

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
        key = [self.seed, self.seed_obs, self.init[0], self.dt, self.duration, self.t[-1], \
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

        autapse_seed = self.gen_newseed()
        key = self._hash(autapse_seed, np.sum(theta))

        if self.cached_sims and key in d:
            states = d[key]
        else:
            autapse = self.bm.Autapse(self.init, theta.reshape(1, -1), seed=autapse_seed)
            states = autapse.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)

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
            autapse_seed = self.gen_newseed()

            key = self._hash(autapse_seed, np.sum(params))

            if self.cached_pilot and key in d:
                states = d[key]
            else:
                autapse = self.bm.Autapse(self.init, params.reshape(1, -1), seed=autapse_seed)
                states = autapse.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)
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

        # moment based summary statistics
        if self.summary_stats == 1:

            # mean during stimulation
            x_mn = np.mean(x[(self.t > self.t_on) & (self.t < self.t_off)])

            # standard deviation during stimulation
            x_std = np.std(x[(self.t > self.t_on) & (self.t < self.t_off)])

            # auto-correlations
            x_on_off = x[(self.t > self.t_on) & (self.t < self.t_off)]-np.mean(x[(self.t > self.t_on) & (self.t < self.t_off)])
            x_corr_val = np.dot(x_on_off,x_on_off)

            xcorr_steps = np.linspace(0.1/self.dt,self.n_xcorr*0.1/self.dt,self.n_xcorr).astype(int)
            x_corr_full = np.zeros(self.n_xcorr)
            for ii in range(self.n_xcorr):
                x_on_off_part = np.concatenate((x_on_off[xcorr_steps[ii]:],np.zeros(xcorr_steps[ii])))
                x_corr_full[ii] = np.dot(x_on_off,x_on_off_part)

            x_corr1 = x_corr_full/x_corr_val

            # concatenation of summary statistics
            sum_stats_vec = np.concatenate((
                    np.array([x_mn,x_std]),
                    x_corr1
                ))

        elif self.summary_stats == 2:
            return np.array([np.mean(x[(self.t > self.t_on) & (self.t < self.t_off)])]).reshape(-1,1)

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
