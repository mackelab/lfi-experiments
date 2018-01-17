import delfi.distribution as dd
import delfi.inference as infer
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

# set the seed
seed = None

m = BalancedNetwork(inference_params=['wie'], dim=1, first_port=8100,
                    verbose=True, n_servers=2, duration=1., parallel=True,
                    estimate_time=False, calculate_stats=True, seed=seed)

s = Identity(seed=seed)

wie = 0.014
# create a list of three identical params
params = [wie] * 2

# generate data
data = m.gen(params)

# generate stats
stats = s.calc_all(data)

print(stats[0])
print(stats[1])

m.stop_server()
