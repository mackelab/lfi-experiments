import time
import numpy as np

from likelihoodfree.Simulator import SimulatorBase
import likelihoodfree.PDF as pdf
import likelihoodfree.io as io

import MarkovJumpProcess as mjp

from likelihoodfree.Simulator import SimTooLongException

class LVSimulator(SimulatorBase):
    def __init__(self,
                 cached_pilot=False,
                 cached_obs=False,
                 seed=None,
                 verbose=False):
        """Lotka-Volterra simulator

        Parameters
        ----------
        cached_pilot : bool (default: False)
        cached_obs : bool (default: False)
        seed : int or None (default: None)
            If set, randomness across runs is disabled
        verbose : bool (default: False)
            Print extra info or not

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
        self.init = [50, 100]
        self.dt = 0.2
        self.duration = 30
        self.true_params = [0.01, 0.5, 1.0, 0.01]

        self.prior_log = True
        self.prior_min = -5  # in log space
        self.prior_max = 2  # in log space

        self.max_n_steps = 10000
        self.verbose = verbose

        # pilot run
        if not cached_pilot:
            t_sim = time.time()
            self.pilot_run()
            if self.verbose:
                print(('pilot run took {:.1f}ms'.format(time.time() - t_sim)))
        self.pilot_means, self.pilot_stds = io.load(self.datadir + 'pilot_run_results.pkl')

        # load obs or run
        if not cached_obs:
            self.get_obs_stats()  # generate x0
        self.obs = io.load(self.datadir + 'obs_stats.pkl')


    def pilot_run(self, n_sims=1000):
        """Pilot run

        Runs a number of simulations, and it calculates and saves the mean and standard deviation of the summary statistics
        across simulations. The intention is to use these to normalize the summary statistics when doing distance-based
        inference, like rejection or mcmc abc. Due to the different scales of each summary statistic, the euclidean distance
        is not meaningful on the original summary statistics. Note that normalization also helps when using mdns, since it
        normalizes the neural net input.
        """
        stats = []
        i = 1

        while i <= n_sims:
            params = self.sim_prior()
            lv = mjp.LotkaVolterra(self.init, params, seed=self.seed)

            try:
                states = lv.sim_time(self.dt, self.duration, max_n_steps=self.max_n_steps)
            except SimTooLongException:
                continue

            stats.append(self.calc_summary_stats(states, pilot_norm=False))

            if self.verbose:
                print('pilot simulation {0}'.format(i))
            i += 1

        stats = np.array(stats)
        means = np.mean(stats, axis=0)
        stds = np.std(stats, axis=0, ddof=1)

        io.save((means, stds), self.datadir + 'pilot_run_results.pkl')

    def forward_model(self, prop_params):
        lv = mjp.LotkaVolterra(self.init, prop_params, seed=self.seed)
        states = lv.sim_time(self.dt, self.duration, max_n_steps=self.max_n_steps)
        return states

    def get_obs_stats(self, show_figure=False):
        """Observed statistics

        Runs the lotka volterra simulation once with the true parameters, and saves the observed summary statistics.

        The intention is to use the observed summary statistics to perform inference on the parameters.
        """
        lv = mjp.LotkaVolterra(self.init, self.true_params, seed=self.seed)
        states = lv.sim_time(self.dt, self.duration)
        stats = self.calc_summary_stats(states)

        io.save(stats, self.datadir + 'obs_stats.pkl')

        if show_figure:
            import matplotlib.pyplot as plt
            plt.figure()
            times = np.linspace(0.0, self.duration, int(self.duration / self.dt) + 1)
            plt.plot(times, states[:, 0], label='predators')
            plt.plot(times, states[:, 1], label='prey')
            plt.xlabel('time')
            plt.ylabel('counts')
            plt.title('params = {0}'.format(self.true_params))
            plt.legend(loc='upper right')
            plt.show()

    def calc_summary_stats(self, states, pilot_norm=True):
        """Calculate summary stats

        Given a sequence of states produced by a simulation, calculates and returns a vector of summary statistics.
        Assumes that the sequence of states is uniformly sampled in time.

        Parameters
        ----------
        states
        pilot_norm : bool
        """
        N = states.shape[0]
        x, y = states[:, 0].copy(), states[:, 1].copy()

        # means
        mx = np.mean(x)
        my = np.mean(y)

        # variances
        s2x = np.var(x, ddof=1)
        s2y = np.var(y, ddof=1)

        # standardize
        x = (x - mx) / np.sqrt(s2x)
        y = (y - my) / np.sqrt(s2y)

        # auto correlation coefficient
        acx = []
        acy = []
        for lag in [1, 2]:
            acx.append(np.dot(x[:-lag], x[lag:]) / (N-1))
            acy.append(np.dot(y[:-lag], y[lag:]) / (N-1))

        # cross correlation coefficient
        ccxy = np.dot(x, y) / (N-1)

        sum_stats = np.array([mx, my, np.log(s2x + 1), np.log(s2y + 1)] + acx + acy + [ccxy])

        # pilot run normalization
        if pilot_norm:
            sum_stats -= self.pilot_means
            sum_stats /= self.pilot_stds

        return sum_stats

    def sim_prior(self, num_sims=1):
        """Simulates parameters from the prior

        Assumes a uniform prior in the log domain.
        """
        z = self.rng.rand(4) if num_sims == 1 else self.rng.rand(num_sims, 4)
        return np.exp((self.prior_max - self.prior_min) * z + self.prior_min)

    def run_sims_from_prior(self, num_sims=100000):
        """Runs several simulations with parameters sampled from the prior.

        Saves the parameters, normalized summary statistics
        and distances with the observed summary statistic. Intention is to use the data for rejection abc and to train mdns.
        """
        obs_stats = io.load(self.datadir + 'obs_stats.pkl')

        params = []
        stats = []
        dist = []

        for i in range(self.num_sims):

            prop_params = self.sim_prior()
            lv = mjp.LotkaVolterra(self.init, prop_params, seed=self.seed)

            try:
                states = lv.sim_time(self.dt, self.duration, max_n_steps=self.max_n_steps)
            except SimTooLongException:
                continue

            sum_stats = self.calc_summary_stats(states)

            params.append(prop_params)
            stats.append(sum_stats)
            dist.append(self.calc_dist(sum_stats, obs_stats))

            if self.verbose:
                print('simulation {0}, distance = {1}'.format(i, dist[-1]))

        params = np.array(params)
        stats = np.array(stats)
        dist = np.array(dist)

        filename = self.datadir + 'sims_from_prior_{0}.pkl'.format(time.time())
        io.save((params, stats, dist), filename)
