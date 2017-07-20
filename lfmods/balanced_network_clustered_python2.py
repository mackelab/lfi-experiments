from brian import *
from lfmods.balanced_network_utils import *
import time

n_rounds = 1
n_trials = 3

# create simulation network
net = Network()
NE = 4000
NI = 1000
N = NE + NI

simulation_time = 3 * second
vt = 1
vr = 0

# cluster parameters
n_cluster = 50
# cluster coef
ree = 1.0
# average ee sparseness
p_ee = 0.2
cluster_weight_factor = 1.9
p_in, p_out = get_cluster_connection_probs(ree, n_cluster, p_ee)
Nc = int(NE / n_cluster)

tau_e = 15*ms
tau_i = 10*ms
tau1 = 1 * ms
tau2_e = 4 * ms
tau2_i = 2 * ms
tau_scale = 1 * ms

# weights
wee = 0.024
wei = 0.045
wie = 0.014
wii = 0.057

# sparseness
p_ie = 0.5
p_ei = 0.5
p_ii = 0.5

eqs = '''
dv/dt = (mu-v)/tau + (I_e - I_i)/tau_scale : 1
dI_e/dt = -(I_e - x_e)/tau2_e : 1
dI_i/dt = -(I_i - x_i)/tau2_i : 1
dx_e/dt = -x_e / tau1 : 1
dx_i/dt = -x_i / tau1 : 1
mu : 1
tau : second
'''

print('Setting up...')
params = dict(n_rounds=n_rounds, n_trials=n_trials, NE=NE, NI=NI, simulation_time=simulation_time)

P = NeuronGroup(N, eqs, threshold='v>vt', reset='v=vr', refractory=5 * ms)
net.add(P)

Pe = P[:NE]
Pi = P[NE:]

Pe.tau = tau_e
Pi.tau = tau_i

example_neuron = int(NE / 2)

Mv = StateMonitor(P, 'v', record=example_neuron)
MIe = StateMonitor(P, 'I_e', record=example_neuron)
MIi = StateMonitor(P, 'I_i', record=example_neuron)

sme = SpikeMonitor(Pe)
smi = SpikeMonitor(Pi)
monitors = [Mv, MIe, MIi, sme, smi]
net.add(monitors)

# build a dict to be save to disc: holds each trial in a sub dict. each sub dict holds the spike times
round_dict = dict(params=params)

for r in range(n_rounds):

    print('building connections...')
    # create clusters
    PeCluster = [Pe[i * Nc:(i + 1) * Nc] for i in range(n_cluster)]
    connection_objects = []

    # establish connections
    Cii = Connection(Pi, Pi, 'x_i', sparseness=p_ii, weight=wii)
    Cei = Connection(Pi, Pe, 'x_i', sparseness=p_ei, weight=wei)
    Cie = Connection(Pe, Pi, 'x_e', sparseness=p_ie, weight=wie)
    connection_objects.append(Cii)
    connection_objects.append(Cei)
    connection_objects.append(Cie)

    if ree == 1.:
        Cee = Connection(Pe, Pe, 'x_e', sparseness=p_ee, weight=wee)  # uniform only
        connection_objects.append(Cee)  # uniform only
        print('uniform connectivity is used...')
    else:
        CeeIn = [None] * n_cluster
        CeeOut = [None] * n_cluster * (n_cluster - 1)
        for i in range(n_cluster):
            for j in range(n_cluster):
                # cluster-internal excitatory connections (cluster only)
                if i == j:
                    CeeIn[i] = Connection(PeCluster[i], PeCluster[i], 'x_e', sparseness=p_in,
                                          weight=wee * cluster_weight_factor)
                    connection_objects.append(CeeIn[i])

                # cluster-external excitatory connections (cluster only)
                else:
                    connection_objects.append(Connection(PeCluster[i], PeCluster[j], 'x_e', sparseness=p_out, weight=wee))

    net.add(connection_objects)

    for trial in range(n_trials):
        tic = time.time()

        # reset state variables
        net.reinit()

        # set the random initial conditions
        Pe.mu = np.random.uniform(1.1, 1.2, NE) * (vt - vr) + vr
        Pi.mu = np.random.uniform(1.0, 1.05, NI) * (vt - vr) + vr

        Pe.v = np.random.rand(NE) * (vt - vr) + vr
        Pi.v = np.random.rand(NI) * (vt - vr) + vr

        print('Running ... trial {} / {} in round {} / {}'.format(trial + 1, n_trials, r + 1, n_rounds))
        net.run(simulation_time, report='text')
        print('Done...')

        toc = time.time() - tic
        print('time elapsed this trial in min: ', toc / 60.)

        # save the result of the current trial
        trial_dict = dict()
        trial_dict['spikes_E'] = sme.getspiketimes()
        trial_dict['spikes_I'] = smi.getspiketimes()
        round_dict['trial{}'.format(trial)] = trial_dict

    # remove the connections so that they can be formed again for a new round
    net.remove(connection_objects)

# save results to disk
save_data(data=round_dict, filename='{}r{}t{}ree{}'.format(time.time(), n_rounds, n_trials, 1),
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
sme.plot()
plt.savefig('spiketrain_uniform.pdf')
plt.show()
