from lfmods.balanced_network_utils import *

filename = '1500558782.44r1t3ree1.p'
folder = '/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/'

round_dict = load_data(filename, folder)

params = round_dict['params']
n_rounds = params['n_rounds']
n_trials = params['n_trials']
NE = params['NE']
NI = params['NI']

t = 1.5  # in sec
delta_t = 1.5  # in sec
recordings_length = delta_t

# define time windows for spike counts
# in 50 ms time window
window_length = 0.05  # in s
# ignore overlap for now
n_time_windows = int(recordings_length / window_length)

# also make a matrix that collects all the spike counts, for correlations and ff: rounds x trials x neurons
spike_count_mat_E = np.zeros((n_rounds, n_trials, NE))
spike_count_mat_I= np.zeros((n_rounds, n_trials, NI))

spike_count_window_mat_E = np.zeros((n_rounds, n_trials, NE, n_time_windows))
spike_count_window_mat_I = np.zeros((n_rounds, n_trials, NI, n_time_windows))
correlations_E = []
correlations_I = []
fano_factors_E = []

for r in range(n_rounds):

    for trial in range(n_trials):
        trial_dict = round_dict['trial{}'.format(trial)]

        # get spike counts in matrix
        spike_count_mat_E[r, trial, :] = get_spikecounts_for_time_window(trial_dict['spikes_E'],
                                                                         t, delta_t) # for E neurons
        spike_count_mat_I[r, trial, :] = get_spikecounts_for_time_window(trial_dict['spikes_I'],
                                                                         t, delta_t) # for I neurons

        # get spike counts for sliding time windows
        spike_count_window_mat_E[r, trial, :, :] = calculate_spike_counts_over_windows(trial_dict['spikes_E'], t,
                                                                                       delta_t, window_length)


    # calculate correlations
    correlations_E.append(calculate_correlation_matrix(spike_count_window_mat_E[r, ...]))
    fano_factors_E.append(calculate_fano_factor(spike_count_window_mat_E[r, ...]))

# rate histogram
rates = np.mean(spike_count_mat_E, axis=1).squeeze() / delta_t  # mean over trials
plt.figure(figsize=(10, 5))
plt.hist(rates, bins=40, range=[0, 15], alpha=.7)
plt.title('Firing rates in spikes / sec')
plt.axvline(np.mean(rates), linestyle='--', label='mean={}'.format(np.round(np.mean(rates), 2)), color='C1')
plt.legend()
plt.savefig('rate_hist_' + filename + 'df')

# correlation histogram
rho = correlations_E[0]
plt.figure(figsize=(10, 5))
plt.hist(rho, bins=40, range=[-.5, .5], alpha=.7)
plt.axvline(np.mean(rho), linestyle='--', label='mean={}'.format(np.round(np.mean(rho), 2)), color='C1')
plt.legend()
plt.savefig('rho_hist_' + filename + 'df')

# fano factor histogram
ff = fano_factors_E[0]
plt.figure(figsize=(10, 5))
plt.hist(ff, bins=40, range=[0, 3], alpha=.7)
plt.title('Fano factors over trials and {}s windows'.format(window_length))
plt.axvline(np.mean(ff), linestyle='--', label='mean={}'.format(np.round(np.mean(ff), 2)), color='C1')
plt.legend()
plt.savefig('ff_hist_' + filename + 'df')
plt.show()
