import delfi.distribution as dd
import delfi.inference as infer
import os
import pickle
import time

try:
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats, Identity
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator
except:
    import sys
    sys.path.append('../../../lfi-models')
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats, Identity
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 2
n_cores_to_use = 4

ntrain = 4
n_minibatch = 2
n_pilot_samples = 0

nrounds = 1
round_cl = 3

save_data = False

# if True, calculates the summary stats on the fly to save memory
stats_onthefly = True

path_to_save_folder = '../data/'  # has to exist on your local path

true_params = [0.024, 0.045, 0.014, 0.057]  # params from the paper

j_indices = [0, 3]
j_labels = [['ee', 'ei', 'ie', 'ii'][j_index] for j_index in j_indices]

true_params = [true_params[j_index] for j_index in j_indices]
param_names = ''
param_names = [param_names + 'w' + j_label for j_label in j_labels]
print(param_names)



s = Identity() if stats_onthefly else BalancedNetworkStats(n_workers=n_cores_to_use)

m = BalancedNetwork(inference_params=param_names, dim=n_params, first_port=9000,
                    verbose=True, n_servers=n_cores_to_use, duration=.5, parallel=True, calculate_stats=stats_onthefly)

p = dd.Uniform(lower=[0.5 * true_param for true_param in true_params],
               upper=[1.3 * true_param for true_param in true_params])

g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# run forward model
data = m.gen([true_params])

# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.SNPE(g, obs=stats_obs, n_components=1, pilot_samples=n_pilot_samples)

# run with N samples
out, trn_data, posteriors = res.run(ntrain, nrounds, epochs=500, minibatch=n_minibatch, round_cl=round_cl)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                   posterior=posterior, out=out, trn_data=trn_data, prior=p, posterior_list=posteriors)

filename = '{}_snpe_J{}_r{}_ntrain{}'.format(time.time(), ''.join(j_labels), nrounds, ntrain).replace('.', '') + '.p'
print(filename)

# set up a dict for saving the results
if save_data and os.path.exists(path_to_save_folder):

    filepath = os.path.join(path_to_save_folder, filename)

    with open(filepath, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif save_data:
    print('Path does not exist: {} saving in .'.format(path_to_save_folder))

    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
