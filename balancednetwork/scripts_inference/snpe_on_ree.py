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
    sys.path.append('../../../lfimodels')
    from lfimodels.balancednetwork.BalancedNetworkSimulator import BalancedNetwork
    from lfimodels.balancednetwork.BalancedNetworkStats import BalancedNetworkStats
    from lfimodels.balancednetwork.BalancedNetworkGenerator import BalancedNetworkGenerator

n_params = 1
n_cores_to_use = 8
ntrain = 200
nrounds = 3
save_data = True


m = BalancedNetwork('ree', dim=n_params, first_port=9000,
                    verbose=True, n_servers=n_cores_to_use, duration=3.)
p = dd.Uniform(lower=[1.], upper=[5.])
s = BalancedNetworkStats(n_workers=n_cores_to_use)
g = BalancedNetworkGenerator(model=m, prior=p, summary=s)

# here we set the true params
true_params = [[4.5]]
# run forward model
data = m.gen(true_params)
# get summary stats
stats_obs = s.calc(data[0])

# set up inference
res = infer.SNPE(g, obs=stats_obs, n_components=1, pilot_samples=50)

# run with N samples, for N=100 this will take 2000s = 0.6h
out, trn_data = res.run(ntrain, nrounds, epochs=1000, minibatch=100)

# evaluate the posterior at the observed data
posterior = res.predict(stats_obs)

# set up a dict for saving the results
path_to_save_folder = '../data/'  # has to exist on your local path

if save_data and os.path.exists(path_to_save_folder):
    result_dict = dict(true_params=true_params, stats_obs=stats_obs, nrouns=nrounds, ntrain=ntrain,
                       posterior=posterior, out=out, trn_data=trn_data)

    filename = os.path.join(path_to_save_folder,
                            '{}_snpe_ree_r{}_ntrain{}'.format(time.time(), nrounds, ntrain).replace('.', '') + '.p')
    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename)
elif save_data:
    print('Path does not exist: {}'.format(path_to_save_folder))

# extract the posterior
n_components = len(posterior.a)
means = [posterior.xs[c].m for c in range(n_components)]
Ss = [posterior.xs[c].S for c in range(n_components)]

print('Predicited: {} +- {}'.format(means, Ss))
print('True: {}'.format(true_params))
