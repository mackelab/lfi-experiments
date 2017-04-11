from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lfmods.hh_bm as bm
#import lfmods.hh_bm_cython as bm
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

class HHSimulator(SimulatorBase):
    def __init__(self,
                 cached_pilot=True,
                 cached_sims=True,
                 dir_cache='results/hh/data/',
                 duration=120,
                 obs_sim=True,
                 pilot_samples=1000,
                 prior_uniform=True,
                 seed=None,
                 step_current=True,
                 summary_stats=1,
                 verbose=False):
        """Hodgkin-Huxley simulator

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
        obs_sim : bool
            If True, will use simulation data for x0
            If False, will use cell form AllenDB for x0
        pilot_samples : bool
            Number of pilot samples to generate, set to 0 to disable normalize
            of summary statistics
        prior_uniform : bool
            Flag to switch between Gaussian and uniform prior
        seed : int or None
            If set, randomness across runs is disabled
        step_current : bool
            Serves as a switch to change between step current and colored noise
        summary_stats : int
            Serves as a switch to change between different ways to calculate
            summary statistics:
                0 : no summary stats (returns identity)
                1 : moments
                2 : hand-crafted
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

        self.cached_pilot = cached_pilot
        self.cached_sims = cached_sims
        self.dir_cache = dir_cache
        self.duration = duration
        self.obs_sim = obs_sim
        self.pilot_samples = pilot_samples
        self.step_current = step_current
        self.summary_stats = summary_stats
        self.verbose = verbose

        if pilot_samples > 0:
            self.pilot_norm = True
        else:
            self.pilot_norm = False

        if self.seed is None:
            self.cached_pilot = False
            self.cached_sims = False

        # true parameters
        self.true_params = np.array([50., 5., 0.1, 50., 90., 70., 0.07, 6e2,
                                     0.5, 40., 60., 0.2])
        self.labels_params = ['g_Na', 'g_K', 'g_l', 'E_Na', '-E_K', '-E_l',
                              'g_M', 't_max', 'k_b_n1', 'k_b_n2', 'V_T', 'noise']
        self.labels_params = self.labels_params[0:len(self.true_params)]
        self.n_params = len(self.true_params)

        # parameters that globally govern the simulations
        self.init = [-70]  # =V0
        self.t_offset = 0.
        self.dt = 0.01
        self.t = np.arange(0, self.duration+self.dt, self.dt)
        self.t_on = 10
        self.t_off = self.duration - self.t_on

        # external current
        self.A_soma = np.pi*((70.*1e-4)**2)  # cm2
        self.curr_level = 5e-4  # uA
        step_current = np.zeros_like(self.t)
        step_current[int(np.round(self.t_on/self.dt)):int(np.round(self.t_off/self.dt))] = self.curr_level/self.A_soma
        if self.step_current:
            self.I = step_current
        else:
            times = np.linspace(0.0, self.duration, int(self.duration / self.dt) + 1)
            I_new = step_current*1.
            tau_n = 3.
            nois_mn = 0.2*step_current
            nois_fact = 2*step_current*np.sqrt(tau_n)
            for i in range(1, times.shape[0]):
                I_new[i] = I_new[i-1] + self.dt*(-I_new[i-1] + nois_mn[i-1] + nois_fact[i-1]*self.rng.normal(0)/(self.dt**0.5))/tau_n
            self.I = I_new
        self.I_obs = self.I.copy()

        self.max_n_steps = 10000
        self.eps_sumstat = 0. # std of jitter in summary statistics

        # summary statistics
        if self.summary_stats == 0:  # no summary (whole time-series)
            self.signal_ds = 10
            self.n_summary_stats = len(self.t[::self.signal_ds])  # quick hack: downsampling
            self.labels_sum_stats = []
            if self.pilot_norm:
                print('Warning: pilot_norm is True, while using identity summary stats')
        elif self.summary_stats == 1:  # moments
            self.n_xcorr = 10
            self.n_mom = 7
            self.n_summary_stats = self.n_xcorr + self.n_mom + 3
            self.labels_sum_stats = ['sp_t',
                                     'c0','c1','c2','c3', 'c4','c5','c6','c7','c8','c9',
                                     'r_pot',
                                     'mn','m2','m3','m4','m5','m6','m7','m8']
        elif self.summary_stats == 2:  # hand-crafted
            self.n_summary_stats = 7
            self.labels_sum_stats = ['sp_t',
                                     'c0','c1','c2','c3', 'c4','c5']
        else:
            raise ValueError('summary_stats invalid')

    @lazyprop
    def obs(self):
        if self.obs_sim:
            # generate observed data from simulation
            hh = bm.HH(self.init, self.true_params.reshape(1, -1),
                       seed=self.gen_newseed())
            states = hh.sim_time(self.dt, self.t, self.I)
            stats = self.calc_summary_stats(states)
        else:
            # use real data
            from allensdk.core.cell_types_cache import CellTypesCache
            from allensdk.api.queries.cell_types_api import CellTypesApi

            manifest_file = 'cell_types/cell_types_manifest.json'
            ephys_data = 464212183
            sweep_number = 33

            cta = CellTypesApi()
            ctc = CellTypesCache(manifest_file=manifest_file)
            data_set = ctc.get_ephys_data(ephys_data)

            sweeps = cta.get_ephys_sweeps(ephys_data)  # works, but different keys in dict

            #sweep_numbers = data_set.get_sweep_numbers()
            sweep_data = data_set.get_sweep(sweep_number)  # this fails

            index_range = sweep_data["index_range"]
            i = sweep_data["stimulus"][0:index_range[1]+1] # in A
            v = sweep_data["response"][0:index_range[1]+1] # in V
            sampling_rate = sweep_data["sampling_rate"] # in Hz

            i *= 1e12 # to pA
            v *= 1e3 # to mV

            v = v[int(self.t_offset*sampling_rate/1000):int((self.t_offset+self.duration)*sampling_rate/1000)]
            i = i[int(self.t_offset*sampling_rate/1000):int((self.t_offset+self.duration)*sampling_rate/1000)]

            states = np.array(v).reshape(-1,1)

            times = np.arange(0, len(v)) * (1000 / sampling_rate)

            # sub-sampling
            # states = v[:,0::200]
            # i = i[:,0::200]
            # times = np.arange(0, len(v)) * (200.0 / sampling_rate)

            stats = self.calc_summary_stats(states)
        return stats.reshape(1, -1)

    @lazyprop
    def prior(self):
        range_lower = self.param_transform(0.5*self.true_params)
        range_upper = self.param_transform(1.5*self.true_params)

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
            return 0.

    def _hash(self, *args):
        """Hashing function to generate key for shelve dicts

        Hashing will be based on args passed to function, plus a set
        of attributes describing global simulator settings.
        """
        key = [self.seed, self.init[0], self.dt, self.duration, self.t[-1], \
               np.sum(self.I), self.max_n_steps, self.summary_stats]

        for arg in args:
            key.append(arg)

        return str(hash(tuple(key)))

    def forward_model(self, prop_params, n_samples=1):
        """Runs the model

        Parameters
        ----------
        show_figure : bool

        Returns
        -------
        states : tracelength x features (=1)
        """
        assert n_samples == 1, 'n_samples > 1 not implemented'

        if self.cached_sims:
            cached_sims_path = self.dir_cache + 'cached_sims_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_sims_path)

        hh_seed = self.gen_newseed()
        key = self._hash(hh_seed, np.sum(prop_params))

        if self.cached_sims and key in d:
            return d[key]
        else:
            hh = bm.HH(self.init, prop_params.reshape(1, -1), seed=hh_seed)
            states = hh.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)

        if self.cached_sims:
            d[key] = states
            d.close()

        return states.reshape(n_samples, -1, 1)

    def calc_summary_stats(self,
                           states,
                           skip_norm=False):
        """Calculate summary stats

        Parameters
        ----------
        states : timesteps x features (=1)
        skip_norm : bool
            If True, will skip pilot run normalization disregarding self.pilot_norm setting

        Returns
        -------
        Array of summary stats

        Notes
        -----
        Output will be based on summary_stats property
        """
        assert states.ndim == 2, 'excepting 2d input'
        assert states.shape[1] == 1, 'expecting feature dimension to be 1'

        # no summary stats?
        if self.summary_stats == 0:
            return np.hstack((states[::self.signal_ds].reshape(-1,1),
                              self.I[::self.signal_ds].reshape(-1,1)))

        x = states.reshape(-1)
        len_x = x.shape[0]

        # initialise array of spike counts
        v = np.array(x)

        # put everything to -10 that is below -10 or has negative slope
        ind = np.where(v < -10)
        v[ind] = -10
        ind = np.where(np.diff(v) < 0)
        v[ind] = -10

        # remaining negative slopes are at spike peaks
        ind = np.where(np.diff(v) < 0)
        spike_times = np.array(self.t)[ind]
        spike_times_stim = spike_times[(spike_times > self.t_on) & (spike_times < self.t_off)]

        # moment based summary statistics
        if self.summary_stats == 1:
            # number of spikes
            if spike_times_stim.shape[0] > 0:
                spike_times_stim = spike_times_stim[np.append(1, np.diff(spike_times_stim))>0.5]

            # resting potential
            rest_pot = np.mean(x[self.t<self.t_on])

            # auto-correlations
            x_on_off = x[(self.t > self.t_on) & (self.t < self.t_off)]-np.mean(x[(self.t > self.t_on) & (self.t < self.t_off)])
            len_x_on_off = x_on_off.shape[0]
            x_corr1 = np.correlate(x_on_off,x_on_off,'full')[len_x_on_off:len_x_on_off+self.n_xcorr]/np.correlate(x_on_off,x_on_off,'valid')

            moments = spstats.moment(x[(self.t > self.t_on) & (self.t < self.t_off)], np.linspace(2,self.n_mom+1,self.n_mom))

            # concatenation of summary statistics
            sum_stats_vec = np.concatenate((
                    np.array([spike_times_stim.shape[0]]),
                    x_corr1,
                    np.array([rest_pot,np.mean(x[(self.t > self.t_on) & (self.t < self.t_off)])]),
                    moments
                ))

        elif self.summary_stats == 2:  # hand-crafted summary statistics
            if spike_times_stim.shape[0] == 0:
                firing_rate = 0
                time_1st_spike = 0
                AHD = np.min(x[(self.t > self.t_on) & (self.t < self.t_off)])
                A_ind = 1000
                spike_width = 0
            else:
                # choose one spike time within close spike times (window of 0.5 ms)
                ind1 = np.array(ind)
                ind_stim1 = ind1[0,(spike_times > self.t_on) & (spike_times < self.t_off)]
                ind_stim1 = ind_stim1[np.append(1,np.diff(spike_times_stim))>0.5]
                ind_stim = ind_stim1.astype(int)

                spike_times_stim = spike_times_stim[np.append(1,np.diff(spike_times_stim))>0.5]

                # firing rate
                firing_rate = np.absolute(spike_times_stim.shape[0]/(self.t_off-self.t_on)+self.eps_sumstat*self.rng.randn())

                time_1st_spike = spike_times_stim[spike_times_stim>self.t_on][0]

                # average spike width
                if spike_times_stim.shape[0] == 1:
                    delta_ind_spik = np.round(self.t[(self.t>self.t_on) & (self.t<self.t_off)].shape[0]/2).astype(int)
                else:
                    ISI = np.diff(spike_times_stim)
                    delta_ind_spik = np.round(np.min(ISI)/(2*self.dt)).astype(int)

                spike_width1 = np.zeros_like(spike_times_stim)
                for i_sp in range(spike_times_stim.shape[0]):
                    # voltages post-spike
                    x_isi = x[ind_stim[i_sp].astype(int):np.minimum(ind_stim[i_sp].astype(int)+delta_ind_spik,len_x)]
                    t_isi = t[ind_stim[i_sp].astype(int):np.minimum(ind_stim[i_sp].astype(int)+delta_ind_spik,len_x)]

                    x_post = x_isi[0:np.maximum(np.argmin(x_isi),2)]
                    t_post = t_isi[0:np.maximum(np.argmin(x_isi),2)]

                    if x_post.size == 0:
                        import pdb; pdb.set_trace()

                    # half-maximum voltage
                    x_half_max = 0.5*(x_post[-1]+x_post[0])

                    # voltages pre-spike
                    x_pre = x[np.maximum(ind_stim[i_sp].astype(int)-delta_ind_spik,0):ind_stim[i_sp].astype(int)]
                    t_pre = t[np.maximum(ind_stim[i_sp].astype(int)-delta_ind_spik,0):ind_stim[i_sp].astype(int)]

                    spike_width1[i_sp] = t_post[np.argmin(np.absolute(x_post - x_half_max))]-t_pre[np.argmin(np.absolute(x_pre - x_half_max))]

                spike_width = np.mean(spike_width1)


                # after-hyperpolarization depth and accomodation index
                if spike_times_stim.shape[0] < 3:
                    AHD_trace = x[(self.t > spike_times_stim[0]) & (self.t < self.t_off)]
                    if AHD_trace.shape[0] > 0:
                        AHD = np.min(AHD_trace)
                    else:
                        AHD = np.min(x[(self.t > self.t_on) & (self.t < self.t_off)])
                    A_ind = 1000
                else:
                    AHD = np.mean( [min(x[i_min:i_max]) for (i_min,i_max) in zip (ind_stim[0:-1].astype(int),ind_stim[1:].astype(int))] )
                    A_ind = np.mean( [ (ISI[i_min+1]-ISI[i_min])/(ISI[i_min+1]+ISI[i_min]) for i_min in range (0,ISI.shape[0]-1)] )

            t_begin = t[(self.t>self.t_on) & (self.t<time_1st_spike)]
            v_begin = x[(self.t>self.t_on) & (self.t<time_1st_spike)]

            # latency from stimulus onset to first spike and mean action potential overshoot
            if t_begin.shape[0] == 0:
                AP_latency = 10000
                AP_overshoot_mn = np.max(x[(self.t > t_on) & (self.t < self.t_off)])
            elif t_begin.shape[0] == 1 or t_begin.shape[0] == 2:
                AP_latency = np.absolute(t_begin[0]+self.eps_sumstat*self.rng.randn())
                AP_overshoot_mn = np.absolute(np.mean(x[ind_stim])+self.eps_sumstat*self.rng.randn())
            else:
                AP_latency = np.absolute(t_begin[np.argmax(np.diff(v_begin, n=2))]+self.eps_sumstat*self.rng.randn())
                AP_overshoot_mn = np.absolute(np.mean(x[ind_stim])+self.eps_sumstat*self.rng.randn())

            # resting potential
            rest_pot = np.mean(x[self.t<self.t_on])

            sum_stats_vec = np.array([
                    firing_rate,
                    AP_latency,
                    AP_overshoot_mn,
                    rest_pot,
                    AHD,
                    A_ind,
                    spike_width
                ])

        else:
            raise ValueError('summary_stats is invalid')

        if np.isnan(sum_stats_vec).any():
            print(sum_stats_vec)
            import pdb; pdb.set_trace()

        # pilot run normalization
        if not skip_norm and self.pilot_norm:
            sum_stats_vec -= self.pilot_means
            sum_stats_vec /= self.pilot_stds

        return sum_stats_vec.reshape(-1)


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
            hh_seed = self.gen_newseed()

            key = self._hash(hh_seed, np.sum(params))

            if self.cached_pilot and key in d:
                states = d[key]
            else:
                hh = bm.HH(self.init, params.reshape(1, -1), seed=hh_seed)
                states = hh.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)

            stats.append(self.calc_summary_stats(states, skip_norm=True))

            if self.cached_pilot and key not in d:
                d[key] = states

        if self.cached_pilot:
            d.close()

        stats = np.array(stats)
        means = np.mean(stats, axis=0)
        stds = np.std(stats, axis=0, ddof=1)

        return means, stds
