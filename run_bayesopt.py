from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import likelihoodfree.PDF as lfpdf
import numpy as np
import pdb
import os
import sys

from GPyOpt.methods import BayesianOptimization
from likelihoodfree.io import first, last, nth
from lfmods.hh import HHSimulator


def run_bo(I, dt, t, bounds, obs_stats, pilot_means, pilot_stds,
                acqui, max_iter, dir_bayesopt='', prefix=''):

    # set up HH simulator
    sim = HHSimulator(seed=101,pilot_samples=0.,cached_sims=False, cached_pilot=False)

    def f_mod(params):
        # simulation
        params = np.array(params)
        hh = sim.bm.HH(-70.,params.reshape(1,-1))
        states = hh.sim_time(dt, t, I).reshape(1, -1, 1)

        # summary statistics
        sum_stats = sim.calc_summary_stats(states)
        sum_stats -= pilot_means
        sum_stats /= pilot_stds

        return sim.calc_dist(sum_stats, obs_stats)

    # GPyOpt object with model and acquisition function
    Bopt_mod = BayesianOptimization(f=f_mod,                  # function to optimize
                                  bounds=bounds,              # box-constraints of problem
                                  model_type = 'GP',
                                  acquisition_type=acqui,     # acquisition function
                                  exact_feval = True)
    # run GPyopt
    Bopt_mod.run_optimization(max_iter)

    io.save(Bopt_mod, dir_bayesopt + str(prefix) + '_' + acqui + '.pkl')

@click.command()
@click.argument('model', type=str)
@click.argument('prefix', type=str)
@click.option('--acqui', type=str, default='EI', show_default=True, help="Determines \
which acquisition function is run. Options are: EI, LCB... (see https://github.com/SheffieldML/GPyOpt)")
@click.option('--debug', default=True, is_flag=True, show_default=True,
              help='If provided, will enter debugger on error and show more \
info during runtime.')
@click.option('--seed', default=None, type=int, help='Seed')
def run(model, prefix, acqui, debug, seed):
    """Runs bayesian optimisation

    Call run_bayesopt.py together with a prefix and a model to run.

    See run_bayesopt.py --help for info on parameters.
    """
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_nets'] = 'results/'+model+'/nets/'
    dirs['dir_bayesopt'] = 'results/'+model+'/bayesopt/'
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

        bounds = []
        for i in range(n_params):
            bounds.append((unlog_prior_min[i],unlog_prior_max[i]))

        pilot_means = sim.pilot_means
        pilot_stds = sim.pilot_stds

        print('GPyopt for {}/{}'.format(model, prefix))

        sim_step = io.last(infos)['n_samples']
        num_round = len(infos)

        max_iter = int(num_round*sim_step/100)

        run_bo(I, dt, t, bounds, obs_stats, pilot_means, pilot_stds,
                    acqui, max_iter, dirs['dir_bayesopt'], prefix)
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
