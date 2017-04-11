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

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from likelihoodfree.Simulator import lazyprop, SimulatorBase
from scipy import stats as spstats
from tqdm import tqdm

class HHSimulator(SimulatorBase):
    def __init__(self,
                 duration=120,
                 step_current=True,
                 summary_stats=1,
                 obs_sim=True,
                 cached_pilot=True,
                 cached_obs=True,
                 cached_sims=True,
                 pilot_norm=True,
                 prefix=None,
                 prior_uniform=True,
                 seed=None,
                 verbose=False):
        """Hodgkin-Huxley simulator

        Parameters
        ----------
        duration : int (default: 120)
            Duration of traces in ms
        step_current : bool (default: True)
            Serves as a switch to change between step current and colored noise
        summary_stats : int (default: 1)
            Serves as a switch to change between different ways to calculate
            summary statistics:
                0 : no summary stats (returns identity)
                1 : moments (default)
                2 : hand-crafted
        obs_sim : bool (default: True)
            If True, will use simulation data for x0
            If False, will use cell form AllenDB for x0
        cached_pilot : bool (default: True)
            If True, will try to use cached pilot data (only works if seed is set)
        cached_obs : bool (default: True)
            If True, will try to use cached observed data (only works if seed is set)
        cached_sims : bool (default: True)
            If True, and iff seed is specified, will cache simulations (only works if seed is set)
        pilot_norm : bool (default: True)
            If True, will normalize summary stats using a pilot run
        prefix : None or str (default: None)
            If set, will use this prefix for files that are loaded/saved
        prior_uniform : bool (default: True)
            Flag to switch between Gaussian and uniform prior
        seed : int or None (default: None)
            If set, randomness across runs is disabled
        verbose : bool (default: False)
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
        # call init of base class
        super().__init__(prior_uniform=prior_uniform, seed=seed)

        # bind args
        self.verbose = verbose
        self.pilot_norm = pilot_norm
        if prefix is not None:
            self.prefix = '{}_'.format(prefix)
        else:
            self.prefix = ''
        self.cached_pilot = cached_pilot
        self.cached_obs = cached_obs
        self.cached_sims = cached_sims
        if self.seed is None:
            self.cached_pilot = False
            self.cached_obs = False
            self.cached_sims = False
        self.obs_sim = obs_sim

        # parameters that globally govern the simulations
        self.init = [-70]  # =V0
        self.t_offset = 0.
        self.dt = 0.01
        self.duration = duration
        self.t = np.arange(0, self.duration+self.dt, self.dt)
        self.t_on = 10
        self.t_off = self.duration - self.t_on

        # external current
        self.step_current = step_current
        self.A_soma = np.pi*((70.*1e-4)**2)  # cm2
        self.curr_level = 5e-4  # uA
        self.set_current()
        self.I_obs = self.I.copy()

        # true parameters
        # gNa, gK, gleak, E_Na, -E_K, -E_leak, gbar_M, tau_max, k_beta_n1, k_beta_n2, V_T, nois_fact
        self.true_params = np.array([50., 5., 0.1, 50., 90., 70., 0.07, 6e2, 0.5, 40., 60., 0.2])
        self.labels_params = ['g_Na', 'g_K', 'g_l', 'E_Na', '-E_K', '-E_l', 'g_M', 't_max', 'k_b_n1', 'k_b_n2', 'V_T', 'noise']
        self.labels_params = self.labels_params[0:len(self.true_params)]
        self.n_params = len(self.true_params)

        # defining the prior
        self.prior_log = True
        if self.prior_log:
            self.param_transform = np.log
            self.param_invtransform = np.exp
            self.prior_min = np.log(0.5*self.true_params)
            self.prior_max = np.log(1.5*self.true_params)
        else:
            self.param_transform = lambda x: x
            self.param_invtransform = lambda x: x
            self.prior_min = 0.9*self.true_params
            self.prior_max = 1.1*self.true_params
        if not self.prior_uniform:
            prior_mn = self.param_transform(true_params)
            prior_cov = np.diag(0.01*(self.prior_max-self.prior_min))
            self.prior_gauss = pdf.Gaussian(m=prior_mn, S=prior_cov, seed=seed)
            self.prior_min = -np.inf  # to avoid rejections
            self.prior_max = +np.inf  # to avoid rejections

        self.max_n_steps = 10000
        self.eps_sumstat = 0. # std of jitter in summary statistics

        # summary statistics
        self.summary_stats = summary_stats
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

        # noise on summary statistics
        self.nois_stats = False
        if self.nois_stats:
            # TODO @PEDRO: need implementation for this
            vars_volt_feats = np.load(self.datadir + 'vars_volt_feats.npy')
            self.vars_volt_feats = vars_volt_feats[0:self.n_summary_stats]
        else:
            self.vars_volt_feats = np.zeros([1, self.n_summary_stats])

    @lazyprop
    def obs(self):
        if self.obs_sim:
            return self.get_obs_stats_sim()  # generate x0
        else:
            return self.get_obs_stats_data()  # generate x0

    @lazyprop
    def prior(self):
        return pdf.Uniform(lower=self.prior_min, upper=self.prior_max,
                           seed=self.gen_newseed())

    def set_current(self):
        # generate a step current
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

    def pilot_run(self, n_samples=1000):
        """Pilot run

        Runs a number of simulations, and it calculates and saves the mean and
        standard deviation of the summary statistics across simulations.

        Parameters
        ----------
        n_samples : int (default: 1000)
        """
        stats = []

        if self.cached_pilot:
            cached_pilot_path = self.datadir + 'cached_pilot_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_pilot_path)

        for i in tqdm(range(n_samples)):
            params = self.sim_prior()
            hh_seed = self.rng.randint(0, 2**31)

            key = self._hash(hh_seed, np.sum(params))

            if self.cached_pilot and key in d:
                states = d[key]
            else:
                hh = bm.HH(self.init, params.reshape(1, -1), seed=hh_seed)
                states = hh.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)

            stats.append(self.calc_summary_stats(states, skip_norm=True))
            if self.verbose:
                print('pilot simulation {0}'.format(i))

            if self.cached_pilot and key not in d:
                d[key] = states

        if self.cached_pilot:
            d.close()

        stats = np.array(stats)
        means = np.mean(stats, axis=0)
        stds = np.std(stats, axis=0, ddof=1)

        return means, stds

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
            cached_sims_path = self.datadir + 'cached_sims_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_sims_path)

        #self.set_current()

        hh_seed = self.rng.randint(0, 2**31)

        key = self._hash(hh_seed, np.sum(prop_params))

        if self.cached_sims and key in d:
            return d[key]
        else:
            hh = bm.HH(self.init, prop_params.reshape(1, -1), seed=hh_seed)
            states = hh.sim_time(self.dt, self.t, self.I, max_n_steps=self.max_n_steps)

        if self.cached_sims:
            d[key] = states.reshape(-1, 1)
            d.close()

        return states.reshape(n_samples, -1, 1)

    def get_obs_stats_sim(self):
        """Observed statistics from simulation

        Runs the model once with the true parameters and saves the observed
        summary statistics.
        """
        hh_seed = self.rng.randint(0, 2**31)

        if self.cached_obs:
            cached_obs_stats_sim_path = self.datadir + 'cached_obs_stats_sim_seed_{}.pkl'.format(self.seed)
            d = shelve.open(cached_obs_stats_sim_path)

        key = self._hash(hh_seed, np.sum(self.true_params))

        if self.cached_obs and key in d:
            return d[key]

        hh = bm.HH(self.init, self.true_params.reshape(1, -1), seed=hh_seed)
        states = hh.sim_time(self.dt, self.t, self.I)
        stats = self.calc_summary_stats(states)

        if self.cached_obs:
            d[key] = stats
            d.close()

        return stats

    def get_obs_stats_data(self,
                           manifest_file='cell_types/cell_types_manifest.json',
                           ephys_data=464212183,
                           sweep_number=33):
        """Observed statistics from data

        Parameters
        ----------
        manifest_file : str (default: 'cell_types/cell_types_manifest.json')
            Storage location of the manifest, which is a JSON file that tracks
            file paths
        ephys_data : int (default: 464212183)
            Identifier for NWB file to use
        sweep_number : int (default: 33)
            Sweep number
        show_figure : bool
        verbose : bool (default: True)
        """
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

        return stats

    def calc_summary_stats(self,
                           states,
                           skip_norm=False):
        """Calculate summary stats

        Parameters
        ----------
        states : timesteps x features (= timesteps x 1)
        skip_norm : bool (default: False)
            If True, will skip pilot run normalization disregarding self.pilot_norm setting

        Returns
        -------
        Array of summary stats

        Notes
        -----
        Output will be based on summary_stats property
        """
        # no summary stats?
        if self.summary_stats == 0:
            # quick hack for testing: trace, current, downsample
            return np.hstack((states[::self.signal_ds].reshape(-1,1), self.I[::self.signal_ds].reshape(-1,1)))
            #import pdb
            #pdb.set_trace()
            #return states.reshape(-1)

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

            # add gaussian noise
            sum_stats_vec = np.array([
                    firing_rate,
                    AP_latency,
                    AP_overshoot_mn,
                    rest_pot,
                    AHD,
                    A_ind,
                    spike_width
                ]) + vars_volt_feats*self.rng.randn(1, -1)

            # sum_stats_vec = sum_stats_vec*(np.sign(firing_rate))

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

    def sim_prior(self, n_samples=1):
        """Simulate from prior

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        n_samples x n_params
            If prior was in log domain, we inverse transform back to normal
        """
        return self.param_invtransform(self.prior.gen(n_samples).reshape(n_samples, -1))  # (n_samples, 12)

    @lazyprop
    def pilot_means(self):
        if self.pilot_norm:
            if not hasattr(self, '_pilot_means'):
                self._pilot_means, self._pilot_stds = self.pilot_run()
                return self._pilot_means
            else:
                return self._pilot_means
        else:
            return 0

    @lazyprop
    def pilot_stds(self):
        if self.pilot_norm:
            if not hasattr(self, '_pilot_stds'):
                self._pilot_means, self._pilot_stds = self.pilot_run()
                return self._pilot_stds
            else:
                return self._pilot_stds
        else:
            return 0
