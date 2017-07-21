from brian2 import *
from lfmods.balanced_network_utils import *
from brian2tools import *
import time
from sklearn.model_selection import KFold


n_realizations = 1
n_trials = 1


full_time = 10 * second
vt = 1
vr = 0
# seed the random number generator
np.random.seed(11)

tau_e = 15 * ms
tau_i = 10 * ms
tau1 = 1 * ms
tau2_e = 3 * ms
tau2_i = 2 * ms
tau_scale = 1 * ms

# weights
wee = 0.024
wei = 0.045
wie = 0.014
wii = 0.057

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

NE = 4000
NI = 1000
N = NE + NI

n_clusters = 50
# cluster coef
ree = 2.5
# average ee sparseness
p_ee = 0.2
cluster_weight_factor = 1.9
p_in, p_out = get_cluster_connection_probs(ree, n_clusters, p_ee)
Nc = int(NE / n_clusters)

params = dict(n_rounds=n_realizations, n_trials=n_trials, NE=NE, NI=NI, simulation_time=full_time)

# create simulation network
net = Network()

example_neuron = int(NE / 2)

# store the network before the connectivity
net.store('initial_state')

# build a dict to be save to disc: holds each trial in a sub dict. each sub dict holds the spike times
round_dict = dict(params=params)

for realization in range(n_realizations):

    net.restore('initial_state')

    P = NeuronGroup(N, eqs, threshold='v>vt', reset='v=vr', method='euler', refractory=5 * ms)
    Pe = P[:NE]
    Pi = P[NE:]
    net.add(P)

    Pe.tau = tau_e
    Pi.tau = tau_i

    print('building connections...')
    PeCluster = [Pe[i * Nc:(i + 1) * Nc] for i in range(n_clusters)]
    connection_objects = []

    tic = time.time()
    if ree == 1.:
        See = Synapses(Pe, Pe, 'w : 1', on_pre='''x_e += w''')
        See.connect(p=0.2)
        See.w = wee
        connection_objects.append(See)
    else:
        print('connecting the clusters...')
        # list of synapse objects
        # # do the cluster connection like cross validation: cluster neuron := test idx; other neurons := train idx
        # kf = KFold(n_splits=n_clusters)
        # loop_idx = 1
        # for other_idx, cluster_idx in kf.split(range(NE)):
        #
        #     # connect current cluster to itself
        #     See.connect(condition='{} <= i and i <= {} and {} <= j and j <= {}'.format(cluster_idx[0], cluster_idx[-1],
        #                                                                                cluster_idx[0], cluster_idx[-1]),
        #                 p=p_in)
        #     See.w[cluster_idx, cluster_idx] = wee * cluster_weight_factor
        #
        #     # connect current cluster to itself
        #     See.connect(condition='{} <= i and i <= {} and {} <= j and j <= {}'.format(cluster_idx[0], cluster_idx[-1],
        #                                                                                other_idx[0], other_idx[-1]),
        #                 p=p_out)
        #     See.w[cluster_idx, other_idx] = wee
        #     print('{} / {} clusters connected'.format(loop_idx, n_clusters))
        #     loop_idx += 1

        loop_idx = 1
        for i in range(n_clusters):
            for j in range(n_clusters):
                # cluster-internal excitatory connections (cluster only)
                if i == j:
                    SeeIn = Synapses(PeCluster[i], PeCluster[j], 'w : 1', on_pre='''x_e += w''')
                    SeeIn.connect(p=p_in)
                    SeeIn.w = wee * cluster_weight_factor
                    connection_objects.append(SeeIn)

                # cluster-external excitatory connections (cluster only)
                else:
                    SeeOut = Synapses(PeCluster[i], PeCluster[j], 'w : 1', on_pre='''x_e += w''')
                    SeeOut.connect(p=p_out)
                    SeeOut.w = wee
                    connection_objects.append(SeeOut)
            print('{} / {} clusters connected'.format(loop_idx, n_clusters))
            loop_idx += 1

    Sii = Synapses(Pi, Pi, 'w : 1', on_pre='''x_i += w''')
    Sii.connect(p=0.5)
    Sii.w = wii
    connection_objects.append(Sii)

    Sei = Synapses(Pi, Pe, 'w : 1', on_pre='''x_i += w''')
    Sei.connect(p=0.5)
    Sei.w = wei
    connection_objects.append(Sei)

    Sie = Synapses(Pe, Pi, 'w : 1', on_pre='''x_e += w''')
    Sie.connect(p=0.5)
    Sie.w = wie
    connection_objects.append(Sie)

    net.add(connection_objects)

    toc = time.time() - tic
    print('time elapsed in min: ', toc / 60.)

    # set the random initial conditions
    Pe.mu = np.random.uniform(1.1, 1.2, NE) * (vt - vr) + vr
    Pi.mu = np.random.uniform(1.0, 1.05, NI) * (vt - vr) + vr

    # monitor only the trial activity
    Mv = StateMonitor(P, 'v', record=example_neuron)
    MIe = StateMonitor(P, 'I_e', record=example_neuron)
    MIi = StateMonitor(P, 'I_i', record=example_neuron)

    sme = SpikeMonitor(Pe)
    smi = SpikeMonitor(Pi)

    net.add([Mv, MIe, MIi, sme, smi])

    # store the network when connected
    net.store('connected')

    for trial in range(n_trials):
        tic = time.time()

        # restore the connected state
        net.restore('connected')

        Pe.v = np.random.rand(NE) * (vt - vr) + vr
        Pi.v = np.random.rand(NI) * (vt - vr) + vr

        print('Running ... trial {} / {} in realization {} / {}'.format(trial + 1, n_trials,
                                                                        realization + 1, n_realizations))
        net.run(full_time, report='text')
        print('Done...')

        toc = time.time() - tic
        print('time elapsed this trial in min: ', toc / 60.)

        # save the result of the current trial
        trial_dict = dict()
        trial_dict['spikes_E'] = sme.spike_trains()
        trial_dict['spikes_I'] = smi.spike_trains()
        round_dict['trial{}'.format(trial)] = trial_dict

# save results to disk
save_data(data=round_dict, filename='{}r{}t{}ree{}dur{}_brain2'.format(time.time(), n_realizations, n_trials, ree,
                                                                       full_time),
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
brian_plot(sme, markersize=1.)
save_figure(filename='spiketrain_uniform_b2.pdf')
plt.show()
