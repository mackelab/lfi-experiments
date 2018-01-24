import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.utils.io as io
import delfi.summarystats as ds
import lfimodels.glm.utils as utils
import numpy as np
import os
import scipy.misc
import sys


def run_mcmc(model, prior, summary, obs_stats, n_params, seed=None, tol=2,step=0.1,n_samples=5e7):
    """Runs MCMC-ABC algorithm.
    Adapted from epsilonfree code https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

    Parameters
    ----------
    model : 
         Model
    prior :
         Prior
    summary :
         Function to compute summary statistics
    obs_stats: 
         Observed summary statistics
    n_params : 
         Number of parameters
    seed : int or None
        If set, randomness in sampling is disabled
    tol : float
        Tolerance for MCMC-ABC
    step : float
        Step for MCMC-ABC
    n_samples : int
        Number of simulations for MCMC-ABC
    """
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_abc'] = './results/abc/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)
            
    prefix = str(n_params)+'params'
    
    #####################
    np.random.seed(seed)
    
    n_sims = 0
    n_samples = int(n_samples)

    # initialize markov chain with a parameter whose distance is within tolerance
    num_init_sims = int(1e5)
    for i in range(num_init_sims):
        ps = prior.gen(n_samples=1)[0]
        states = model.gen_single(ps)
        stats = summary.calc([states])
        dist = calc_dist(stats, obs_stats)
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

        prop_ps = cur_ps + step * np.random.randn(n_params)
        states = model.gen_single(prop_ps)
        prop_stats = summary.calc([states])
        prop_dist = calc_dist(prop_stats, obs_stats)
        n_sims += 1

        # acceptance / rejection step
        #if prop_dist < tol and np.all(prop_ps <= prior_max) and np.all(prop_ps >= prior_min):
        if prop_dist < tol and np.random.rand() < np.exp(prior.eval(prop_ps[np.newaxis, :], log=True) - prior.eval(cur_ps[np.newaxis, :], log=True)):
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

    return ps, stats, dist, acc_rate, n_sims

    
def run_smc(model, prior, summary, obs_stats, n_params, seed=None,n_particles=1e3,eps_init=2,maxsim=5e7):
    """Runs Sequential Monte Carlo ABC algorithm.
    Adapted from epsilonfree code https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

    Parameters
    ----------
    model : 
         Model
    prior :
         Prior
    summary :
         Function to compute summary statistics
    obs_stats: 
         Observed summary statistics
    n_params : 
         Number of parameters
    seed : int or None
        If set, randomness in sampling is disabled
    n_particles : int
        Number of particles for SMC-ABC
    eps_init : Float
        Initial tolerance for SMC-ABC
    maxsim : int
        Maximum number of simulations for SMC-ABC
    """
    n_particles = int(n_particles)
    
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_abc'] = './results/abc/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    prefix = str(n_params)+'params'
    
    #####################
    np.random.seed(seed)

    # set parameters
    eps_last = 0.01
    eps_decay = 0.9
    ess_min = 0.5
    maxsim = int(maxsim)

    all_ps = []
    all_logweights = []
    all_eps = []
    all_nsims = []

    # sample initial population
    ps = np.empty([n_particles, n_params])
    weights = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0

    for i in range(n_particles):

        dist = float('inf')

        while dist > eps:
            ps[i] = prior.gen(n_samples=1)[0]
            states = model.gen_single(ps[i])
            stats = summary.calc([states])
            dist = calc_dist(stats, obs_stats)
            nsims += 1

    all_ps.append(ps)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    break_flag = False

    print('iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, float(eps), 1.0))

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

        for i in range(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = discrete_sample(weights)[0]
                new_ps[i] = ps[idx] + np.dot(std, np.random.randn(n_params))
                states = model.gen_single(new_ps[i])
                stats = summary.calc([states])
                dist = calc_dist(stats, obs_stats)
                nsims += 1
                if nsims>=maxsim:
                    raise Warning('Maximum number of simulations reached.')
                    break_flag = True
                    break
            #new_ps_i = new_ps[i]
            logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_ps[i] - ps).T) ** 2, axis=0)
            #new_logweights[i] = -float('inf') if np.any(new_ps_i > prior_max) or np.any(new_ps_i < prior_min) else -scipy.misc.logsumexp(logweights + logkernel)
            new_logweights[i] = prior.eval(new_ps[i, np.newaxis], log=True)[0] - scipy.misc.logsumexp(logweights + logkernel)

            if break_flag:
                break

        if break_flag:
            break

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
                idx = discrete_sample(weights)[0]
                new_ps[i] = ps[idx]

            ps = new_ps
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        all_ps.append(ps)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(nsims)

    return all_ps, all_logweights, all_eps, all_nsims

        
def calc_dist(stats_1, stats_2):
    """Euclidian distance between summary statistics"""
    return np.sqrt(np.sum((stats_1 - stats_2) ** 2))

def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = np.random.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)