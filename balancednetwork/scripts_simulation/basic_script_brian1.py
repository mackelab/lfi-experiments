import sys
import time

from brian import *
from balancednetwork.utils import *

n_realizations = 1
n_trials = 1

np.random.seed(11)

# create simulation network
net = Network()
n = 1000
NE = 4 * n
NI = 1 * n
N = NE + NI

# get the scaling factor for weights in case the network size is different
alpha = get_scaling_factor_for_weights(NE, NI)

simulation_time = 5 * second
vt = 1
vr = 0

# cluster parameters
C = 80
n_clusters = int(NE / C)
# cluster coef
ree = 2.5

# average ee sparseness
p_ee = 0.2
cluster_weight_factor = 1.9
p_in, p_out = get_cluster_connection_probs(ree, n_clusters, p_ee)

tau_e = 15 * ms
tau_i = 10 * ms
tau1 = 1 * ms
tau2_e = 3 * ms
tau2_i = 2 * ms
tau_scale = 1 * ms
tau_refrac = 5 * ms

# weights
wee = 0.024 * alpha
wei = 0.045 * alpha
wie = 0.014 * alpha
wii = 0.057 * alpha

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
params = dict(n_rounds=n_realizations, n_trials=n_trials, NE=NE, NI=NI, simulation_time=simulation_time,
              n_clusters=n_clusters)

# build a dict to be save to disc: holds each trial in a sub dict. each sub dict holds the spike times
round_dict = dict(params=params)

# define monitor place holder for plotting
sme = None

for realization in range(n_realizations):

    # clear workspace to make sure that is a new realization of the network
    net.reinit()

    P = NeuronGroup(N, eqs, threshold='v>vt', reset='v=vr', refractory=tau_refrac, method='Euler')
    net.add(P)

    Pe = P[:NE]
    Pi = P[NE:]

    Pe.tau = tau_e
    Pi.tau = tau_i

    # set the resting potentials
    Pe.mu = np.random.uniform(1.1, 1.2, NE) * (vt - vr) + vr
    Pi.mu = np.random.uniform(1.0, 1.05, NI) * (vt - vr) + vr

    example_neuron = int(NE / 2)

    print('building connections...')
    tic = time.time()
    # create clusters
    PeCluster = [Pe[i * C:(i + 1) * C] for i in range(n_clusters)]

    # establish connections
    Cii = Connection(Pi, Pi, 'x_i', sparseness=p_ii, weight=wii)
    Cei = Connection(Pi, Pe, 'x_i', sparseness=p_ei, weight=wei)
    Cie = Connection(Pe, Pi, 'x_e', sparseness=p_ie, weight=wie)
    net.add(Cii)
    net.add(Cei)
    net.add(Cie)

    # prelocate the connection object, specify connection weights later
    Cee = Connection(Pe, Pe, state='x_e')

    if ree == 1.:
        Cee = Connection(Pe, Pe, 'x_e', sparseness=p_ee, weight=wee)  # uniform only
        print('uniform connectivity is used...')
    else:
        print('connecting the clusters...')
        for i in range(n_clusters):
            for j in range(n_clusters):
                # cluster-internal excitatory connections
                if i == j:
                    Cee.connect_random(PeCluster[i], PeCluster[j], p=p_in, weight=wee * cluster_weight_factor)

                # cluster-external excitatory connections
                else:
                    Cee.connect_random(PeCluster[i], PeCluster[j], p=p_out, weight=wee)

        # kf = KFold(n_splits=n_clusters)
        # for other_idx, cluster_idx in kf.split(range(n_clusters)):
        #
        #     Pin = Pe[cluster_idx[0]:cluster_idx[-1]]
        #     Pout = Pe[other_idx[0]:other_idx[-1]]
        #
        #     # within cluster
        #     Cee.connect_random(Pin, Pin, p=p_in, weight=wee * cluster_weight_factor)
        #
        #     # out of cluster
        #     Cee.connect_random(Pin, Pout, p=p_out, weight=wee)

    net.add(Cee)
    toc = time.time() - tic
    print('time elapsed for connections in sec: ', np.round(toc, 2))

    Mv = StateMonitor(P, 'v', record=example_neuron)
    MIe = StateMonitor(P, 'I_e', record=example_neuron)
    MIi = StateMonitor(P, 'I_i', record=example_neuron)

    sme = SpikeMonitor(Pe[:400])
    smi = SpikeMonitor(Pi)

    spiketimedict_e = sme.getspiketimes()
    spiketimedict_i = smi.getspiketimes()
    spiketimedict = {'{}'.format(k): v.tolist() for k, v in spiketimedict_e.items()}
    

    net.add([Mv, MIe, MIi, sme, smi])

    for trial in range(n_trials):
        tic = time.time()

        # set random initial conditions
        Pe.v = np.random.rand(NE) * (vt - vr) + vr
        Pi.v = np.random.rand(NI) * (vt - vr) + vr

        print('Running ... trial {} / {} in realization {} / {}'.format(trial + 1, n_trials, realization + 1, n_realizations))
        net.run(simulation_time, report='text')
        print('Done...')

        toc = time.time() - tic
        print('time elapsed this trial in sec: ', np.round(toc, 2))

        # save the result of the current trial
        trial_dict = dict()
        trial_dict['spikes_E'] = sme.getspiketimes()
        trial_dict['spikes_I'] = smi.getspiketimes()
        round_dict['trial{}'.format(trial)] = trial_dict

time_str = time.time()
data_filename = '{}ree{}_dur{}_brain1'.format(time_str, ree, simulation_time).replace('.', '')

# save results to disk
save_data(data=round_dict, filename=data_filename,
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
raster_plot(sme, markersize=4)
#raster_plot(smi, markersize=2)
plt.title('Spike trains of E neurons, $R_{ee}$=' + '{}'.format(ree))
spiketrain_filename = '{}_spiketrain_ree{}_dur{}_b1'.format(time_str, ree, simulation_time).replace('.', '') + '.pdf'
plt.tight_layout()
save_figure(filename=spiketrain_filename)
plt.show()
