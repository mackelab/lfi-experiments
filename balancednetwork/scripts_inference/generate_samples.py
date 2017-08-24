import delfi.distribution as dd
import delfi.inference as infer
import os
import pickle
import time

from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 1
n_cores_to_use = 4
nsampels = 10
save_data = True

m = BalancedNetwork(dim=n_params, first_port=8010,
                    verbose=True, n_servers=n_cores_to_use, duration=3.)
p = dd.Uniform(lower=[1.], upper=[5.])
s = BalancedNetworkStats(n_workers=n_cores_to_use)
g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

params, stats = g.gen(nsampels)

# set up a dict for saving the results
path_to_save_folder = '../data/'  # has to exist on your local path

if save_data and os.path.exists(path_to_save_folder):
    nrounds = 1
    result_dict = dict(params=params, stats=stats)

    filename = os.path.join(path_to_save_folder,
                            '{}_params_stats_data_n{}'.format(time.time(), nsampels).replace('.', '') + '.p')
    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename)