from brian import *
from lfmods.balanced_network_utils import *
import time
from sklearn.model_selection import KFold

n_realizations = 1
n_trials = 1

np.random.seed(11)

# create simulation network
net = Network()
NE = 4000
NI = 1000
N = NE + NI

simulation_time = 10 * second
vt = 1
vr = 0

# cluster parameters
n_clusters = 50
# cluster coef
ree = 2.5
# average ee sparseness
p_ee = 0.2
cluster_weight_factor = 1.9
p_in, p_out = get_cluster_connection_probs(ree, n_clusters, p_ee)
Nc = int(NE / n_clusters)

tau_e = 15 * ms
tau_i = 10 * ms
tau1 = 1 * ms
tau2_e = 3 * ms
tau2_i = 2 * ms
tau_scale = 1 * ms
tau_refrac = 5 * ms

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
params = dict(n_rounds=n_realizations, n_trials=n_trials, NE=NE, NI=NI, simulation_time=simulation_time)

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
    # create clusters
    PeCluster = [Pe[i * Nc:(i + 1) * Nc] for i in range(n_clusters)]

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
        connection_objects.append(Cee)  # uniform only
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

    net.add(Cee)
    net.add(connection_objects)

    Mv = StateMonitor(P, 'v', record=example_neuron)
    MIe = StateMonitor(P, 'I_e', record=example_neuron)
    MIi = StateMonitor(P, 'I_i', record=example_neuron)

    sme = SpikeMonitor(Pe)
    smi = SpikeMonitor(Pi)

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
        print('time elapsed this trial in min: ', toc / 60.)

        # save the result of the current trial
        trial_dict = dict()
        trial_dict['spikes_E'] = sme.getspiketimes()
        trial_dict['spikes_I'] = smi.getspiketimes()
        round_dict['trial{}'.format(trial)] = trial_dict

    # remove the connections so that they can be formed again for a new round
    net.remove(connection_objects)

time_str = time.time()
data_filename = '{}ree{}_dur{}_brain1'.format(time_str, ree, simulation_time).replace('.', '')

# save results to disk
save_data(data=round_dict, filename=data_filename,
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
raster_plot(sme, markersize=1.0)
plt.title('Spike trains of E neurons')
spiketrain_filename = '{}_spiketrain_ree{}_b1'.format(time_str, ree).replace('.', '') + '.pdf'
save_figure(filename=spiketrain_filename)
