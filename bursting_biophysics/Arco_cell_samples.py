import model.utils as utils
import multiprocessing
import multiprocessing.pool
import numpy as np
import os
import sys
import time

from __future__ import division

################################################################################
# seeding by time stamp
seed1 = time.time()
seed = int((seed1 % 1)*1e7)
rng = np.random.RandomState(seed=seed)

################################################################################
# directory for simulation
dir_path = '$HOME/in_silico_framework/'
sys.path.append(dir_path)
import Interface as I

from biophysics_fitting import hay_complete_default_setup, L5tt_parameter_setup

################################################################################
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

################################################################################
true_params = utils.obs_params()

# define prior
seed_p = rng.randint(1e7)
p = utils.prior(true_params=true_params,seed=seed_p)

################################################################################
# sample from prior and run simulations
n_sims = 1000

# sample from prior
params = p.gen(n_sims)

# number of parallel processes (smaller or equal to n_sims)
n_processes = 10

# simulate
seeds_model = rng.randint(1e7,size=n_sims)
def sim_f(param):
    x = utils.simulator_wrapper(param[1:],seed=int(param[0]))
    return utils.summary_stats(x)

data = []
params_seed = np.concatenate((seeds_model.reshape(-1,1),params),axis=1)
pool = MyPool(n_processes)
data.append(pool.map(sim_f, params_seed))
pool.close()
pool.join()

# save parameters and respective simulations
outfile = 'arco_cell_samples_seed_p'+str(seed_p)+'.npz'
np.savez_compressed(outfile, seeds=seeds_model, params=params, data=data)