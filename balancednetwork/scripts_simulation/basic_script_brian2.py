import time

from brian2 import *
from brian2tools import *
from sklearn.model_selection import KFold

from balanced_network.balanced_network_utils import *

n_realizations = 1
n_trials = 1


full_time = 2 * second
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
    tic = time.time()
    #PeCluster = [Pe[i * Nc:(i + 1) * Nc] for i in range(n_clusters)]
    connection_objects = []
    See = Synapses(Pe, Pe, 'w : 1', on_pre='''x_e += w''')

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

        #  loop_idx = 1
        # for i in range(n_clusters):
        #     for j in range(n_clusters):
        #         # cluster-internal excitatory connections (cluster only)
        #         if i == j:
        #             SeeIn = Synapses(PeCluster[i], PeCluster[j], 'w : 1', on_pre='''x_e += w''')
        #             SeeIn.connect(p=p_in)
        #             SeeIn.w = wee * cluster_weight_factor
        #             connection_objects.append(SeeIn)
        #
        #         # cluster-external excitatory connections (cluster only)
        #         else:
        #             SeeOut = Synapses(PeCluster[i], PeCluster[j], 'w : 1', on_pre='''x_e += w''')
        #             SeeOut.connect(p=p_out)
        #             SeeOut.w = wee
        #             connection_objects.append(SeeOut)
        #     print('{} / {} clusters connected'.format(loop_idx, n_clusters))
        #     loop_idx += 1

        # loop_idx = 1
        # index_template = np.arange(0, Nc)
        # for k in range(n_clusters):
        #     idx_k = index_template + k * Nc
        #     for j in range(n_clusters):
        #         idx_j = index_template + j * Nc
        #
        #         # cluster-internal excitatory connections (cluster only)
        #         if k == j:
        #             # draw one probs:
        #             # there are k**2 possible connections. every neuron k can connect to every neuron in j
        #             k_neurons = np.repeat(idx_k, idx_k.size)
        #             j_neurons = np.tile(idx_j, idx_j.size)
        #             # now we make a single mask over all these neurons
        #             conn_mask = np.random.rand(idx_k.size ** 2) < p_in
        #             if conn_mask.sum() > 0:
        #                 See.connect(i=k_neurons[conn_mask], j=j_neurons[conn_mask])
        #                 See.w[k_neurons[conn_mask], j_neurons[conn_mask]] = wee * cluster_weight_factor
        #
        #         # cluster-external excitatory connections (cluster only)
        #         else:
        #             k_neurons = np.repeat(idx_k, idx_k.size)
        #             j_neurons = np.tile(idx_j, idx_j.size)
        #             # now we make a single mask over all these neurons
        #             conn_mask = np.random.rand(idx_k.size ** 2) < p_out
        #             if conn_mask.sum() > 0:
        #                 See.connect(i=k_neurons[conn_mask], j=j_neurons[conn_mask])
        #                 See.w[k_neurons[conn_mask], j_neurons[conn_mask]] = wee
        #     print('{} / {} clusters connected'.format(loop_idx, n_clusters))
        #     loop_idx += 1

        kf = KFold(n_splits=n_clusters)
        loop_idx = 1
        for other_idx, cluster_idx in kf.split(range(NE)):

            # draw one probs:
            # there are k**2 possible connections. every neuron k can connect to every neuron in j
            repeats = max(cluster_idx.size, cluster_idx.size)
            k_neurons = np.repeat(cluster_idx, repeats)  # cluster neurons
            j_neurons = np.tile(cluster_idx, repeats)  # also cluster neurons

            # now we make a single mask over all these neurons
            conn_mask = np.random.rand(cluster_idx.size * cluster_idx.size) < p_in
            if conn_mask.sum() > 0:
                See.connect(i=k_neurons[conn_mask], j=j_neurons[conn_mask])
                See.w[k_neurons[conn_mask], j_neurons[conn_mask]] = wee * cluster_weight_factor

            k_neurons = np.repeat(cluster_idx, other_idx.size)  # cluster neurons
            j_neurons = np.tile(other_idx, cluster_idx.size)  # other neurons

            # now we make a single mask over all these neurons
            conn_mask = np.random.rand(cluster_idx.size * other_idx.size) < p_out
            if conn_mask.sum() > 0:
                See.connect(i=k_neurons[conn_mask], j=j_neurons[conn_mask])
                See.w[k_neurons[conn_mask], j_neurons[conn_mask]] = wee
            print('{} / {} clusters connected'.format(loop_idx, n_clusters))
            loop_idx += 1

    connection_objects.append(See)

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

time_str = time.time()
data_filename = '{}ree{}_dur{}_brain2'.format(time_str, ree, full_time).replace('.', '')

# save results to disk
save_data(data=round_dict, filename=data_filename,
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
brian_plot(sme, markersize=1.0)
plt.title('Spike trains of E neurons')
spiketrain_filename = '{}_spiketrain_ree{}_b2'.format(time_str, ree).replace('.', '') + '.pdf'
save_figure(filename=spiketrain_filename)