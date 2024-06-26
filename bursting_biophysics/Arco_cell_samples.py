from __future__ import division

import multiprocessing
import multiprocessing.pool
import numpy as np
import os
import sys
import time
import utils

from parameter_setup import load_ground_truth_params

################################################################################
# seeding by time stamp
seed1 = time.time()
seed = int((seed1 % 1)*1e7)
rng = np.random.RandomState(seed=seed)


################################################################################
# ground-truth parameters
gt = load_ground_truth_params()


################################################################################
# sample from prior and run simulations
n_sims = 20000

# sample from prior
seed_p = rng.randint(1e7)
params = utils.prior_around_gt(gt=gt, fraction_of_full_prior=0.1, num_samples=n_sims, seed=seed_p)

# number of parallel processes (smaller or equal to n_sims)
n_processes = 28

# simulate
seeds_model = rng.randint(1e7,size=n_sims)
def sim_f(param):
    x = utils.simulator_wrapper(param[1:],seed=int(param[0]))
    return utils.summary_stats(x, n_xcorr=0, n_mom=4)

data = []
params_seed = np.concatenate((seeds_model.reshape(-1,1),params),axis=1)
pool = multiprocessing.pool.Pool(n_processes)
data.append(pool.map(sim_f, params_seed))
pool.close()
pool.join()

# save parameters and respective simulations
outfile = 'arco_cell_samples_seed_p'+str(seed_p)+'.npz'
np.savez_compressed(outfile, seeds=seeds_model, params=params, data=data)