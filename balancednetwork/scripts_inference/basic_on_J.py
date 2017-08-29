import delfi.distribution as dd
import delfi.inference as infer
import os
import pickle
import scipy.stats as st
import time

from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 4
n_cores_to_use = 4
ntrain = 10
save_data = True
path_to_save_folder = '../data/'  # has to exist on your local path

m = BalancedNetwork(dim=n_params, first_port=8010,
                    verbose=True, n_servers=n_cores_to_use, duration=3.)
p = dd.Uniform(lower=[0.01] * n_params, upper=[0.1] * n_params)
s = BalancedNetworkStats(n_workers=n_cores_to_use)
g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# here we set the true params
true_params = [[0.024, 0.045, 0.014, 0.057]]  # params from the paper
# run forward model
data = m.gen(true_params)
# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.Basic(g, n_components=3, pilot_samples=50)

# run with N samples
out, trn_data = res.run(ntrain, epochs=1000, minibatch=10)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

# set up a dict for saving the results
if save_data and os.path.exists(path_to_save_folder):
    nrounds = 1
    result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                       posterior=posterior, out=out, trn_data=trn_data)

    filename = os.path.join(path_to_save_folder,
                            '{}_basic_J_ntrain{}'.format(time.time(), ntrain).replace('.', '') + '.p')
    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename)