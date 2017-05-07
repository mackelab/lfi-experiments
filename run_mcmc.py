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

from likelihoodfree.io import first, last, nth
from math import factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


@click.command()
@click.argument('model', type=str)
@click.argument('prefix', type=str)
@click.option('--algo', type=str, default='ess', show_default=True, help="Determines \
which MCMC algorithm is run, so far only ess is implemented.")
@click.option('--debug', default=True, is_flag=True, show_default=True,
              help='If provided, will enter debugger on error and show more \
info during runtime.')
@click.option('--seed-np', default=42, type=int, help='Seed for numpy')
@click.option('--seed-sampler', default=0, type=int, help='Seed for sampler')
def run(model, prefix, algo, debug, seed_np, seed_sampler):
    """Runs MCMC algorithm

    Call run_mcmc.py together with a prefix and a model to run.

    See run_mcmc.py --help for info on parameters.
    """
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_nets'] = 'results/'+model+'/nets/'
    dirs['dir_sampler'] = 'results/'+model+'/sampler/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    np.random.seed(seed_np)

    try:
        # load data from NSPE model
        dists, infos, losses, nets, posteriors, sims = io.load_prefix(dirs['dir_nets'], prefix)
        sim = io.last(sims)
        obs_stats = sim.obs
        y_obs = sim.obs_trace.reshape(-1,1)
        posterior = io.last(posteriors)
        gt = sim.true_params
        t = sim.t
        I = sim.I_obs.reshape(1,-1)

        if algo == 'ess':
            print('Elliptical slice sampling for {}/{}'.format(model, prefix))

            # seeding
            pg = PyPolyaGamma(seed=seed_sampler)

            # simulation protocol
            num_param_inf = len(gt)

            N = 1   # Number of trials
            M = num_param_inf-1   # Length of the filter

            # build covariate matrix X, such that X * h returns convolution of x with filter h
            X = np.zeros(shape=(len(t), M))
            for j in range(M):
                X[j:,j] = I[0,0:len(t)-j]

            # prior
            # smoothing prior on h; N(0, 1) on b0. Smoothness encouraged by penalyzing
            # 2nd order differences of elements of filter
            D = np.diag(np.ones(M)) - np.diag(np.ones(M-1), -1)
            F = np.dot(D, D)

            # Binv is block diagonal
            Binv1 = np.zeros(shape=(M+1,M+1))
            Binv1[0,0] = 1    # offset (b0)
            Binv1[1:,1:] = np.dot(F.T, F) # filter (h)
            prior_dist = lfpdf.Gaussian(m=gt*0., P=Binv1)

            # The sampler consists of two iterative Gibbs updates
            # 1) sample auxiliary variables: w ~ PG(N, psi)
            # 2) sample parameters: beta ~ N(m, V); V = inv(X'O X + Binv1), m = V*(X'k), k = y - N/2

            nsamp = 500000   # 500000 samples to evaluate the posterior

            # add a column of 1s to the covariate matrix X, in order to model the offset too
            X = np.concatenate((np.ones(shape=(len(t), 1)), X), axis=1)

            beta = gt*1.
            BETA = np.zeros((M+1,nsamp))

            for j in tqdm(range(1, nsamp)):
                psi = np.dot(X, beta)
                w = np.array([pg.pgdraw(N, b) for b in psi])
                O = np.diag(w)

                V = np.linalg.inv(np.dot(np.dot(X.T, O), X) + Binv1)
                m = np.dot(V, np.dot(X.T, y_obs - N * 0.5))

                beta = np.random.multivariate_normal(np.ravel(m), V)

                BETA[:,j] = beta

            # burn-in
            burn_in = 100000
            BETA_sub_samp = BETA[:, burn_in:nsamp:30]

            # save sampling results
            np.save(dirs['dir_sampler'] + '/' + prefix + '_ess.npy', BETA_sub_samp)

        elif algo == 'rejection':
            print('Rejection ABC for {}/{}'.format(model, prefix))

            pass
            """
            # from epsilonfree code, needs to be adopted
            # https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

            #  Runs mcmc abc inference. Saves the results for display later.
            tol=0.002, step=0.2, n_samples=50000

            n_sims = 0

            # load observed stats and simulated stats from prior
            _, obs_stats = helper.load(datadir + 'observed_data.pkl')
            prior_ps, prior_stats, prior_dist = load_sims_from_prior(n_files=1)
            n_dim = prior_ps.shape[1]

            # initialize markov chain with a parameter whose distance is within tolerance
            for ps, stats, dist in izip(prior_ps, prior_stats, prior_dist):
                if dist < tol:
                    cur_ps = ps
                    cur_stats = stats
                    cur_dist = dist
                    break
            else:
                raise ValueError('No parameter was found with distance within tolerance.')

            # simulate markov chain
            ps = [cur_ps.copy()]
            stats = [cur_stats.copy()]
            dist = [cur_dist]
            n_accepted = 0

            for i in xrange(n_samples):

                prop_ps = cur_ps + step * rng.randn(n_dim)
                _, _, _, idts, _ = sim_likelihood(*prop_ps)
                prop_stats = calc_summary_stats(idts)
                prop_dist = calc_dist(prop_stats, obs_stats)
                n_sims += 1

                # acceptance / rejection step
                if prop_dist < tol and eval_prior(*prop_ps) > 0.0:
                    cur_ps = prop_ps
                    cur_stats = prop_stats
                    cur_dist = prop_dist
                    n_accepted += 1

                ps.append(cur_ps.copy())
                stats.append(cur_stats.copy())
                dist.append(cur_dist)

                print 'simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(n_accepted) / (i+1))

            ps = np.array(ps)
            stats = np.array(stats)
            dist = np.array(dist)
            acc_rate = float(n_accepted) / n_samples

            filename = datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step)
            helper.save((ps, stats, dist, acc_rate, n_sims), filename)
            """
        elif algo == 'smc':
            print('Sequential Monte Carlo for {}/{}'.format(model, prefix))

            pass
            """
            # from epsilonfree code, needs to be adopted
            # https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

            # Runs sequential monte carlo abc and saves results

            # set parameters
            n_particles = 1000
            eps_init = 0.1
            eps_last = 0.001
            eps_decay = 0.9
            ess_min = 0.5

            # load observed data
            _, obs_stats = helper.load(datadir + 'observed_data.pkl')
            n_dim = 3

            all_ps = []
            all_logweights = []
            all_eps = []
            all_nsims = []

            # sample initial population
            ps = np.empty([n_particles, n_dim])
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)
            eps = eps_init
            iter = 0
            nsims = 0

            for i in xrange(n_particles):

                dist = float('inf')

                while dist > eps:
                    ps[i] = sim_prior()
                    _, _, _, idts, _ = sim_likelihood(*ps[i])
                    stats = calc_summary_stats(idts)
                    dist = calc_dist(stats, obs_stats)
                    nsims += 1

            all_ps.append(ps)
            all_logweights.append(logweights)
            all_eps.append(eps)
            all_nsims.append(nsims)

            print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, 1.0)

            while eps > eps_last:

                iter += 1
                eps *= eps_decay

                # calculate population covariance
                mean = np.mean(ps, axis=0)
                cov = 2.0 * (np.dot(ps.T, ps) / n_particles - np.outer(mean, mean))
                std = np.linalg.cholesky(cov)

                # perturb particles
                new_ps = np.empty_like(ps)
                new_logweights = np.empty_like(logweights)

                for i in xrange(n_particles):

                    dist = float('inf')

                    while dist > eps:
                        idx = helper.discrete_sample(weights)[0]
                        new_ps[i] = ps[idx] + np.dot(std, rng.randn(n_dim))
                        _, _, _, idts, _ = sim_likelihood(*new_ps[i])
                        stats = calc_summary_stats(idts)
                        dist = calc_dist(stats, obs_stats)
                        nsims += 1

                    logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_ps[i] - ps).T) ** 2, axis=0)
                    new_logweights[i] = -float('inf') if eval_prior(*new_ps[i]) < 0.5 else -scipy.misc.logsumexp(logweights + logkernel)

                ps = new_ps
                logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
                weights = np.exp(logweights)

                # calculate effective sample size
                ess = 1.0 / (np.sum(weights ** 2) * n_particles)
                print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, ess)

                if ess < ess_min:

                    # resample particles
                    new_ps = np.empty_like(ps)

                    for i in xrange(n_particles):
                        idx = helper.discrete_sample(weights)[0]
                        new_ps[i] = ps[idx]

                    ps = new_ps
                    weights = np.ones(n_particles, dtype=float) / n_particles
                    logweights = np.log(weights)

                all_ps.append(ps)
                all_logweights.append(logweights)
                all_eps.append(eps)
                all_nsims.append(nsims)

                # save results
                filename = datadir + 'smc_abc_results.pkl'
                helper.save((all_ps, all_logweights, all_eps, all_nsims), filename)

            """
        else:
            raise ValueError('{} not implemented'.format(algo))
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
