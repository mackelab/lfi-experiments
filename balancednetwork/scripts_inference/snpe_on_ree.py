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
n_cores_to_use = 1
ntrain = 100
nrounds = 2
n_pilot_samples = 0
save_data = True

true_ree = 2.5

m = BalancedNetwork('ree', dim=n_params, first_port=9000,
                    verbose=True, n_servers=n_cores_to_use, duration=3., parallel=True)
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
out, trn_data, posteriors = res.run(ntrain, nrounds, epochs=400, minibatch=10)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

# set up a dict for saving the results
path_to_save_folder = '../data/'  # has to exist on your local path
result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                   posterior=posterior, out=out, trn_data=trn_data, prior=p, posterior_list=posteriors)

filename = '{}_snpe_ree_r{}_ntrain{}'.format(time.time(), nrounds, ntrain).replace('.', '') + '.p'
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

# extract the posterior
n_components = len(posterior.a)
means = [posterior.xs[c].m for c in range(n_components)]
Ss = [posterior.xs[c].S for c in range(n_components)]

print('Predicited: {} +- {}'.format(means, Ss))
print('True: {}'.format(true_params))
