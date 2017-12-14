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
n_cores_to_use = 30

ntrain = 500
n_minibatch = 100
n_pilot_samples = 100

nrounds = 10
round_cl = 5

save_data = True

# if True, calculates the summary stats on the fly to save memory
stats_onthefly = True

path_to_save_folder = '../results/'  # has to exist on your local path

j_index = 2
true_params = [0.024, 0.045, 0.014, 0.057]  # params from the paper
j_label = ['ee', 'ei', 'ie', 'ii'][j_index]
true_param = [true_params[j_index]]
param_name = 'w' + j_label

s = Identity() if stats_onthefly else BalancedNetworkStats(n_workers=n_cores_to_use)

m = BalancedNetwork(inference_params=[param_name], dim=n_params, first_port=8700,
                    verbose=True, estimate_time=False, n_servers=n_cores_to_use, duration=3.0, parallel=True, calculate_stats=stats_onthefly)

p = dd.Uniform(lower=[0.8 * true_param[0]], upper=[1.4 * true_param[0]])

g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# run forward model
data = m.gen(true_param)

# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.SNPE(g, obs=stats_obs, n_components=1, pilot_samples=n_pilot_samples)

# run with N samples
out, trn_data, posteriors = res.run(ntrain, nrounds, epochs=500, minibatch=n_minibatch, round_cl=round_cl)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

result_dict = dict(true_params=true_param, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                   posterior=posterior, out=out, trn_data=trn_data, prior=p, posterior_list=posteriors)

simulation_name = '{}_snpe_J{}_r{}_n{}_rcl{}'.format(time.time(), j_label, nrounds, ntrain, round_cl).replace('.', '')

# save the results
if save_data:
    path_to_file = save_results(result_dict, simulation_name, path_to_save_folder)
    print(path_to_file)

# extract the posterior
n_components = len(posterior.a)
means = [posterior.xs[c].m for c in range(n_components)]
Ss = [posterior.xs[c].S for c in range(n_components)]

print('Predicited: {} +- {}'.format(means, Ss))
print('True: {}'.format(true_param))
