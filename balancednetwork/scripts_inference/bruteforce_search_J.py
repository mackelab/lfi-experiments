import delfi.distribution as dd
import delfi.inference as infer
import itertools
import os
import pickle
import time


try:
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats, Identity
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator
    from utils import save_results
except:
    import sys
    sys.path.append('../../../lfi-models')
    sys.path.append('../')
    from utils import save_results
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats, Identity
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

import numpy as np

save_data = True
path_to_save_folder = '../results/'  # has to exist on your local path

true_params = [0.024, 0.045, 0.014, 0.057]  # params from the paper

n_steps = 9
n_cores_to_use = 32

jees = np.linspace(0.009, 0.049, n_steps)
jeis = np.linspace(0.02, 0.06, n_steps)
jies = np.linspace(0.009, 0.049, n_steps)
jiis = np.linspace(0.032, 0.072, n_steps)

n_simulations = n_steps ** 4

print('N simulations:', n_simulations)

# setup simulator to run the simulations
simulator = BalancedNetwork(inference_params=['wxy'], dim=4, first_port=9000,
                            verbose=True, n_servers=n_cores_to_use, duration=3.,
                            estimate_time=True, calculate_stats=True)

calculator = Identity()

# set up the list of parameters: a list of lists with 4 entries
product_iterator = itertools.product(jees, jeis, jies, jiis)
params_list = [p for p in product_iterator]

# simulate
data = simulator.gen(params_list, verbose=True)

# calculate summary stats
stats = calculator.calc_all(data)

result_dict = dict(true_params=true_params, stats=stats, data=data,
                   params=params_list)


simulation_name = '{}_bruteforce_n{}'.format(time.time(), n_simulations).replace('.', '')

# save the results
if save_data:
    path_to_file = save_results(result_dict, simulation_name, path_to_save_folder)
    print(path_to_file)
