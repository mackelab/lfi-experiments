import delfi.distribution as dd
import delfi.inference as infer
import os
import pickle
import time

try:
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator
except:
    import sys
    sys.path.append('../../../lfi-models')
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 1
n_cores_to_use = 8

ntrain = 500
n_minibatch = 100
n_pilot_samples = 50

nrounds = 5
round_cl = 3

save_data = True
path_to_save_folder = '../data/'  # has to exist on your local path

j_index = 0
true_params = [0.024, 0.045, 0.014, 0.057]  # params from the paper
j_label = ['ee', 'ei', 'ie', 'ii'][j_index]
true_param = [true_params[j_index]]
param_name = 'w' + j_label

m = BalancedNetwork(inference_param=param_name, dim=n_params, first_port=9000,
                    verbose=True, n_servers=n_cores_to_use, duration=3., parallel=True)
p = dd.Uniform(lower=[0.5 * true_param[0]], upper=[1.3 * true_param[0]])
s = BalancedNetworkStats(n_workers=n_cores_to_use)
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

filename = '{}_snpe_J{}_r{}_ntrain{}'.format(time.time(), j_label, nrounds, ntrain).replace('.', '') + '.p'
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
