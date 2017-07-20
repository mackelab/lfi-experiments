from brian2 import *
from lfmods.balanced_network_utils import *
from brian2tools import *
import time

n_rounds = 1
n_trials = 10

simulation_time = 3 * second
vt = 1
vr = 0

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
# seed the random number generator
np.random.seed(10)
# create simulation network
net = Network()
NE = 4000
NI = 1000
N = NE + NI

params = dict(n_rounds=n_rounds, n_trials=n_trials, NE=NE, NI=NI, simulation_time=simulation_time)

P = NeuronGroup(N, eqs, threshold='v>vt', reset='v=vr', method='euler', refractory=5 * ms)
Pe = P[:NE]
Pi = P[NE:]
net.add(P)

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

# store the network before the connectivity
net.store('before_connectivity')

# build a dict to be save to disc: holds each trial in a sub dict. each sub dict holds the spike times
round_dict = dict(params=params)

for r in range(n_rounds):

    print('building connections...')
    See = Synapses(Pe, Pe, 'w : 1', on_pre='''x_e += w''')
    See.connect(p=0.2)
    See.w = wee

    Sii = Synapses(Pi, Pi, 'w : 1', on_pre='''x_i += w''')
    Sii.connect(p=0.5)
    Sii.w = wii

    Sei = Synapses(Pi, Pe, 'w : 1', on_pre='''x_i += w''')
    Sei.connect(p=0.5)
    Sei.w = wei

    Sie = Synapses(Pe, Pi, 'w : 1', on_pre='''x_e += w''')
    Sie.connect(p=0.5)
    Sie.w = wie

    net.add([See, Sii, Sei, Sie])

    # store the network when connected
    net.store('connected')

    for trial in range(n_trials):
        tic = time.time()

        # restore the connected state
        net.restore('connected')

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
        trial_dict['spikes_E'] = sme.spike_trains()
        trial_dict['spikes_I'] = smi.spike_trains()
        round_dict['trial{}'.format(trial)] = trial_dict

# save results to disk
save_data(data=round_dict, filename='{}r{}t{}ree{}'.format(time.time(), n_rounds, n_trials, 1),
          folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/')

# #
plt.figure(figsize=(15, 5))
brian_plot(sme, markersize=1.)
plt.savefig('spiketrain_uniform.pdf')
plt.show()
