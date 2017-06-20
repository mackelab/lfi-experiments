from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import likelihoodfree.PDF as lfpdf
import numpy as np
import pdb
import os
import scipy.misc
import sys

from likelihoodfree.io import first, last, nth
from math import factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


@click.command()
@click.argument('model', type=str)
@click.argument('prefix', type=str)
@click.option('--algo', type=str, default='mcmc', show_default=True, help="Determines \
which MCMC algorithm is run: so far ess (for the model `glm`), smc(-abc) and mcmc(-abc) \
are implemented.")
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
    dirs['dir_smc'] = 'results/'+model+'/smc/'
    dirs['dir_mcmc'] = 'results/'+model+'/mcmc/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    np.random.seed(seed_np)

    try:
        # load data from SNPE model
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

        elif algo == 'mcmc':
            print('MCMC-ABC for {}/{}'.format(model, prefix))

            # Adapted from epsilonfree code
            # https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

            #  Runs MCMC-ABC inference. Saves the results for display later.
            tol = .5
            step = 0.2
            n_samples = 10000

            n_sims = 0
            n_dim = len(gt)

            # initialize markov chain with a parameter whose distance is within tolerance
            num_init_sims = 100000
            for i in range(num_init_sims):
                ps = sim.sim_prior(n_samples=1)[0]
                states = sim.forward_model(ps, n_samples=1)
                stats = sim.calc_summary_stats(states)
                dist = sim.calc_dist(stats, obs_stats)
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

            for i in range(n_samples):

                cur_ps_transf = sim.param_transform(cur_ps)
                prop_ps = sim.param_invtransform(cur_ps_transf + step * np.random.randn(n_dim))
                states = sim.forward_model(prop_ps, n_samples=1)
                prop_stats = sim.calc_summary_stats(states)
                prop_dist = sim.calc_dist(prop_stats, obs_stats)
                n_sims += 1

                # acceptance / rejection step
                prop_ps_transf = sim.param_transform(prop_ps)
                if prop_dist < tol and np.all(prop_ps_transf <= sim.prior_max) and np.all(prop_ps_transf >= sim.prior_min):
                    cur_ps = prop_ps
                    cur_stats = prop_stats
                    cur_dist = prop_dist
                    n_accepted += 1

                ps.append(cur_ps.copy())
                stats.append(cur_stats.copy())
                dist.append(cur_dist)

                print('simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(n_accepted) / (i+1)))

            ps = np.array(ps)
            stats = np.array(stats)
            dist = np.array(dist)
            acc_rate = float(n_accepted) / n_samples

            # save results
            io.save((ps, stats, dist, acc_rate, n_sims), dirs['dir_mcmc'] + prefix + '_mcmc_abc.pkl')

        elif algo == 'smc':
            print('Sequential Monte Carlo for {}/{}'.format(model, prefix))

            # Adapted from epsilonfree code
            # https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

            # Runs Sequential Monte Carlo ABC and saves results

            # set parameters
            n_particles = 1000
            eps_init = 10
            eps_last = 0.1
            eps_decay = 0.9
            ess_min = 0.5

            n_dim = len(gt)

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

            for i in range(n_particles):

                dist = float('inf')

                while dist > eps:
                    ps[i] = sim.sim_prior(n_samples=1)[0]
                    states = sim.forward_model(ps[i], n_samples=1)
                    stats = sim.calc_summary_stats(states)
                    dist = sim.calc_dist(stats, obs_stats)
                    nsims += 1

            all_ps.append(ps)
            all_logweights.append(logweights)
            all_eps.append(eps)
            all_nsims.append(nsims)

            print('iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, float(eps), 1.0))

            while eps > eps_last:

                iter += 1
                eps *= eps_decay

                # calculate population covariance
                ps_transf = sim.param_transform(ps)
                mean = np.mean(ps_transf, axis=0)
                cov = 2.0 * (np.dot(ps_transf.T, ps_transf) / n_particles - np.outer(mean, mean))
                std = np.linalg.cholesky(cov)

                # perturb particles
                new_ps = np.empty_like(ps)
                new_logweights = np.empty_like(logweights)

                for i in range(n_particles):

                    dist = float('inf')

                    while dist > eps:
                        idx = lfpdf.discrete_sample(weights)[0]
                        new_ps[i] = sim.param_invtransform(ps_transf[idx] + np.dot(std, np.random.randn(n_dim)))
                        states = sim.forward_model(new_ps[i], n_samples=1)
                        stats = sim.calc_summary_stats(states)
                        dist = sim.calc_dist(stats, obs_stats)
                        nsims += 1

                    new_ps_transf_i = sim.param_transform(new_ps[i])
                    logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_ps_transf_i - ps_transf).T) ** 2, axis=0)
                    new_logweights[i] = -float('inf') if np.any(new_ps_transf_i > sim.prior_max) or np.any(new_ps_transf_i < sim.prior_min) else -scipy.misc.logsumexp(logweights + logkernel)

                ps = new_ps
                logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
                weights = np.exp(logweights)

                # calculate effective sample size
                ess = 1.0 / (np.sum(weights ** 2) * n_particles)
                print('iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, float(eps), ess))

                if ess < ess_min:

                    # resample particles
                    new_ps = np.empty_like(ps)

                    for i in range(n_particles):
                        idx = lfpdf.discrete_sample(weights)[0]
                        new_ps[i] = ps[idx]

                    ps = new_ps
                    weights = np.ones(n_particles, dtype=float) / n_particles
                    logweights = np.log(weights)

                all_ps.append(ps)
                all_logweights.append(logweights)
                all_eps.append(eps)
                all_nsims.append(nsims)

                # save results
                io.save((all_ps, all_logweights, all_eps, all_nsims), dirs['dir_smc'] + prefix + '_smc_abc.pkl')


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
