from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bluepyopt as bpopt
import click
import likelihoodfree.io as io
import likelihoodfree.PDF as lfpdf
import numpy as np
import pdb
import os
import sys
import time

from bluepyopt.parameters import Parameter
from likelihoodfree.io import first, last, nth
from lfmods.hh import HHSimulator
from math import factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

class hh_evaluator(bpopt.evaluators.Evaluator):
    def __init__(self, I, dt, t, target_stats, labels_sum_stats, pilot_means, pilot_stds, params1):
        self.I = I
        self.dt = dt
        self.t = t
        self.target_stats = target_stats
        self.pilot_means = pilot_means
        self.pilot_stds = pilot_stds

        super(hh_evaluator, self).__init__(objectives=labels_sum_stats,params=params1)

    def evaluate_with_lists(self, param_list):
        A = param_list

        # simulation
        sim = HHSimulator(seed=101,pilot_samples=0.,cached_sims=False, cached_pilot=False)

        params = np.array(A)
        hh = sim.bm.HH(-70.,params.reshape(1,-1))
        states = hh.sim_time(self.dt, self.t, self.I).reshape(1, -1, 1)

        # summary statistics
        sum_stats = sim.calc_summary_stats(states)
        sum_stats -= self.pilot_means
        sum_stats /= self.pilot_stds

        return np.ndarray.tolist(np.abs(sum_stats - self.target_stats)[0])

def run_deap(I, dt, t, params1, obs_stats, labels_sum_stats,
             pilot_means, pilot_stds, optim_ibea=1, offspring_size=10, max_ngen=10, dir_genetic='', prefix=''):
    # choose and run genetic algorithm
    evaluator = hh_evaluator(I, dt, t, obs_stats, labels_sum_stats, pilot_means, pilot_stds, params1)

    # choose genetic algorithm
    if optim_ibea==1:
        algo = 'ibea'
        opt = bpopt.deapext.optimisations.IBEADEAPOptimisation(evaluator,offspring_size=offspring_size)
    else:
        algo = 'deap'
        opt = bpopt.deapext.optimisations.DEAPOptimisation(evaluator,offspring_size=offspring_size)

    final_pop, halloffame, log, hist = opt.run(max_ngen=max_ngen)

    io.save((final_pop,halloffame,log,hist), dir_genetic + str(prefix) + '_' + algo + '.pkl')

@click.command()
@click.argument('model', type=str)
@click.argument('prefix', type=str)
@click.option('--algo', type=str, default='ibea', show_default=True, help="Determines \
which genetic algorithm is run, so far only ibea and deap default are implemented.")
@click.option('--debug', default=True, is_flag=True, show_default=True,
              help='If provided, will enter debugger on error and show more \
info during runtime.')
@click.option('--seed', default=None, type=int, help='Seed')
def run(model, prefix, algo, debug, seed):
    """Runs genetic algorithm

    Call run_genetic.py together with a prefix and a model to run.

    See run_genetic.py --help for info on parameters.
    """
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_nets'] = 'results/'+model+'/nets/'
    dirs['dir_genetic'] = 'results/'+model+'/genetic/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    np.random.seed(seed)

    try:
        # load data from SNPE model
        dists, infos, losses, nets, posteriors, sims = io.load_prefix(dirs['dir_nets'], prefix)
        sim = io.last(sims)
        obs_stats = sim.obs
        y_obs = sim.obs_trace.reshape(-1,1)
        labels_sum_stats = sim.labels_sum_stats
        gt = sim.true_params
        t = sim.t
        I = sim.I_obs
        dt = sim.dt

        # un-logged prior over parameters
        unlog_prior_min = sim.param_invtransform(sim.prior_min)
        unlog_prior_max = sim.param_invtransform(sim.prior_max)

        n_params = sim.n_params
        labels_params = sim.labels_params

        params1 = []
        for i in range(n_params):
            params1.append(Parameter(labels_params[i], bounds=[unlog_prior_min[i], unlog_prior_max[i]]))

        pilot_means = sim.pilot_means
        pilot_stds = sim.pilot_stds

        if algo == 'ibea':
            print('IBEA for {}/{}'.format(model, prefix))
            optim_ibea = 1
        else:
            print('Default algorithm of DEAP for {}/{}'.format(model, prefix))
            optim_ibea = 0

        sim_step = io.last(infos)['n_samples']
        num_round = len(infos)

        offspring_size = int(sim_step)
        max_ngen = int(num_round)

        t_sim = time.time()
        run_deap(I, dt, t, params1, obs_stats, labels_sum_stats,
                 pilot_means, pilot_stds, optim_ibea, offspring_size, max_ngen, dirs['dir_genetic'], prefix)
    except:
        t, v, tb = sys.exc_info()
        if debug:
            print('')
            print('Exception')
            print(v.with_traceback(tb))
            pdb.post_mortem(tb)
        else:
            raise v.with_traceback(tb)

if __name__ == '__main__':
    run()
