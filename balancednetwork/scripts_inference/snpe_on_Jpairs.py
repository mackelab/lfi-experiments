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

seed = 1
ree = 2.5
n_params = 2
n_cores_to_use = 32

ntrain = 3000
n_minibatch = 100
n_pilot_samples = 100

nrounds = 7
round_cl = 7

save_data = True

# if True, calculates the summary stats on the fly to save memory
stats_onthefly = True

path_to_save_folder = '../results/'  # has to exist on your local path

true_params = [0.024, 0.045, 0.014, 0.057]  # params from the paper

j_indices = [0, 1]
j_labels = [['ee', 'ei', 'ie', 'ii'][j_index] for j_index in j_indices]

true_params = [true_params[j_index] for j_index in j_indices]
param_names = ''
param_names = [param_names + 'w' + j_label for j_label in j_labels]

s = Identity() if stats_onthefly else BalancedNetworkStats(n_workers=n_cores_to_use)

m = BalancedNetwork(inference_params=param_names, dim=n_params, first_port=8500, seed=seed, ree=ree,
                    verbose=True, n_servers=n_cores_to_use, duration=3.0, parallel=True, estimate_time=True,
                    calculate_stats=stats_onthefly)

p = dd.Uniform(lower=[0.8 * true_param for true_param in true_params],
               upper=[1.3 * true_param for true_param in true_params], seed=seed)

g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# run forward model
data = m.gen([true_params])

# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.SNPE(g, obs=stats_obs, n_components=1, prior_norm=True, pilot_samples=n_pilot_samples, seed=seed)

# run with N samples
out, trn_data, posteriors = res.run(ntrain, nrounds, epochs=500, minibatch=n_minibatch, round_cl=round_cl)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                   seed=seed, posterior=posterior, out=out, trn_data=trn_data, prior=p, posterior_list=posteriors)

simulation_name = '{}_snpe_cJ{}_r{}_n{}_rcl{}'.format(time.time(), ''.join(j_labels), nrounds,
                                                     ntrain, round_cl).replace('.', '')

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
