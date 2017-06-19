from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import os
import pdb
import pickle
import shelve
import time

from likelihoodfree.Simulator import lazyprop, SimulatorBase
from tqdm import tqdm

class COSimulator(SimulatorBase):
    def __init__(self,
                 cached_pilot=False,
                 cached_sims=False,
                 dir_cache='results/competosc/data/',
                 duration=1e4,
                 pilot_samples=1000,
                 prior_uniform=True,
                 seed=None,
                 seed_obs=None,
                 summary_stats=0,
                 verbose=False):
        """Competing oscillators simulator

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
                0 : oscillating frequency of hub neuron
        verbose : bool
            Print extra info or not

        Attributes
        ----------
        obs : observation
            x0 summary statistics
        prior_log : bool
            whether or not prior is in log space
        prior_min
        prior_max
        """
        super().__init__(prior_log=True,
                         prior_uniform=prior_uniform,
                         seed=seed)
        self.seed_obs = seed_obs

        self.cached_pilot = cached_pilot
        self.cached_sims = cached_sims
        self.dir_cache = dir_cache
        self.pilot_samples = pilot_samples
        self.summary_stats = summary_stats
        self.verbose = verbose

        import lfmods.competosc_bm as bm
        self.bm = bm

        if pilot_samples > 0:
            self.pilot_norm = True
        else:
            self.pilot_norm = False

        if self.seed is None:
            self.cached_pilot = False
            self.cached_sims = False

        # true parameters
        self.true_params = np.array([0.0035, .0005])
        self.labels_params = ['g_synA', 'g_el']
        self.labels_params = self.labels_params[0:len(self.true_params)]
        self.n_params = len(self.true_params)

        # parameters that globally govern the simulations
        self.init = [-70]  # =V0
        self.duration = duration
        self.dt = 1
        self.t = np.arange(0, self.duration+self.dt, self.dt)

        self.max_n_steps = 10000

        # summary statistics
        if self.summary_stats == 0:  # oscillating frequency of hub neuron
            self.n_summary_stats = 1
            self.labels_sum_stats = ['hn']
        # elif self.summary_stats == 1:  # oscillating frequencies of all 5 neurons
        #     self.n_summary_stats = 5
        #     self.labels_sum_stats = ['f1','f2','hn','s1', 's2']
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
        co = self.bm.CO(self.init, self.true_params.reshape(1, -1),
                        seed=seed)
        states = co.sim_time(self.dt, self.t)
        states = states.reshape(1, -1, 1)
        stats = self.calc_summary_stats(states)
        self.obs_trace = states
        return stats

    @lazyprop
    def prior(self):
        range_lower = self.param_transform(np.array([0.0001,0.0001]))
        range_upper = self.param_transform(np.array([0.01,0.0075]))

        if self.prior_uniform:
            self.prior_min = range_lower
            self.prior_max = range_upper
            return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                               seed=self.gen_newseed())
        else:
            prior_mn = self.param_transform(self.true_params)
            prior_cov = np.diag(0.01*(range_upper - range_lower))
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
               self.max_n_steps, self.summary_stats]

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

        co_seed = self.gen_newseed()
        key = self._hash(co_seed, np.sum(theta))

        if self.cached_sims and key in d:
            states = d[key]
        else:
            co = self.bm.CO(self.init, theta.reshape(1, -1), seed=co_seed)
            states = co.sim_time(self.dt, self.t,max_n_steps=self.max_n_steps)

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

        i = 0
        progressbar = tqdm(total=self.pilot_samples)
        with progressbar as pbar:
            while i < self.pilot_samples:
                #for i in tqdm(range(self.pilot_samples)):
                params = self.sim_prior()
                co_seed = self.gen_newseed()

                key = self._hash(co_seed, np.sum(params))

                if self.cached_pilot and key in d:
                    states = d[key]
                else:
                    co = self.bm.CO(self.init, params.reshape(1, -1), seed=co_seed)
                    states = co.sim_time(self.dt, self.t,max_n_steps=self.max_n_steps)
                    states = states.reshape(1, -1, 1)


                sum_stats = self.calc_summary_stats(states, skip_norm=True)

                if sum_stats is None or np.any(np.isnan(sum_stats)):
                    continue

                stats.append(sum_stats)

                if self.cached_pilot and key not in d:
                    d[key] = states

                i = i+1
                pbar.update(1)

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

        x = states.reshape(-1)
        len_x = x.shape[0]

        thr = 0

        t = 1e-3*self.t

        datath = x-thr

        sdat = np.sign(datath)
        thrc = np.diff(sdat)
        thrc[np.isnan(thrc)] = 0.

        ONi = np.nonzero(thrc>0)
        ONi1 = tuple(x+1 for x in ONi)
        ONt = t[ONi1]
        pers = np.diff(ONt)
        mpers = np.mean(pers)

        mfrq = 1./mpers
        if np.isnan(mfrq):
            mfrq = 0.

        # oscillating frequency of hub neuron
        if self.summary_stats == 0:
            sum_stats_vec = np.array([mfrq])
        else:
            raise ValueError('summary_stats is invalid')

        #if np.isnan(sum_stats_vec).any():
            #print(sum_stats_vec)
            #import pdb; pdb.set_trace()

        # pilot run normalization
        if not skip_norm and self.pilot_norm:
            sum_stats_vec -= self.pilot_means
            sum_stats_vec /= self.pilot_stds

        return sum_stats_vec.reshape(1, -1)
