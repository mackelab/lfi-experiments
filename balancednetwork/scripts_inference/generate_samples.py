import delfi.distribution as dd
import os
import pickle
import time

try:
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator
except:
    import sys
    sys.path.append('../../../lfimodels')
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 1
n_cores_to_use = 4
nsampels = 200
save_data = True

param_name = 'wee'

m = BalancedNetwork(param_name, dim=n_params, first_port=8010,
                    verbose=True, n_servers=n_cores_to_use, duration=5.)
p = dd.Uniform(lower=[0.01], upper=[0.05])
s = BalancedNetworkStats(n_workers=n_cores_to_use)

g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

params, stats = g.gen(nsampels)

# set up a dict for saving the results
path_to_save_folder = '../data/'  # has to exist on your local path

if save_data and os.path.exists(path_to_save_folder):
    nrounds = 1
    result_dict = dict(params=params, stats=stats)

    filename = os.path.join(path_to_save_folder,
                            '{}_params_stats_data_{}_n{}'.format(time.time(),
                                                                 param_name,
                                                                 nsampels).replace('.', '') + '.p')
    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename)
