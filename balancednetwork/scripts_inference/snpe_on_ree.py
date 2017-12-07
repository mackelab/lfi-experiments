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

n_params = 1
n_cores_to_use = 8
ntrain = 1000
n_minibatch = 100
n_pilot_samples = 100

nrounds = 5
round_cl = 3

save_data = False

true_ree = 2.5

m = BalancedNetwork(['ree'], dim=n_params, first_port=8000,
                    verbose=True, n_servers=n_cores_to_use, duration=.5, parallel=True)
p = dd.Uniform(lower=[0.5 * true_ree], upper=[1.5 * true_ree])
s = BalancedNetworkStats(n_workers=n_cores_to_use)
g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# here we set the true params
true_params = [[true_ree]]
# run forward model
data = m.gen(true_params)
# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.SNPE(g, obs=stats_obs, n_components=1, pilot_samples=n_pilot_samples, svi=True)

# run with N samples
out, trn_data, posteriors = res.run(ntrain, nrounds, epochs=500, minibatch=n_minibatch, round_cl=3)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

# set up a dict for saving the results
path_to_save_folder = '../results/'  # has to exist on your local path
result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                   posterior=posterior, out=out, trn_data=trn_data, prior=p, posterior_list=posteriors)

simulation_name = '{}_snpe_ree_r{}_n{}_rcl{}'.format(time.time(), nrounds, ntrain, round_cl).replace('.', '')

# save the results
if save_data:
    path_to_file = save_results(result_dict, simulation_name, path_to_save_folder)
    print(path_to_file)

# extract the posterior
n_components = len(posterior.a)
means = [posterior.xs[c].m for c in range(n_components)]
Ss = [posterior.xs[c].S for c in range(n_components)]

print('Predicited: {} +- {}'.format(means, Ss))
print('True: {}'.format(true_params))
