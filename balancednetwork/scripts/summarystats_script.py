from utils import *

# uniform
filenames = [ 'data_ree1.9.p', 'data_ree2.5.p', 'data_ree3.1.p']
# clustered
# filenames = ['150117497016ree32_dur50_brain1.p']

for filename in filenames:
    folder = '/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/data/'

    full_path = os.path.join(folder, filename)

    round_dict = pickle.load(open(full_path, 'rb'), encoding='latin1')

    # params = round_dict['params']
    params = dict(n_rounds=1, n_trials=1, simulation_time=2., NE=4000, NI=1000)
    n_rounds = params['n_rounds']
    n_trials = params['n_trials']
    simulation_time = np.asarray(params['simulation_time'])  # remove unit
    NE = params['NE']
    NI = params['NI']

    time_offset = 1.  # in sec
    delta_t = simulation_time - time_offset  # in sec
    recordings_length = delta_t

    # define time windows for spike counts
    # in 50 ms time window
    window_length = 0.1 # in s
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
            trial_dict = round_dict

            # get spike counts in matrix
            spike_count_mat_E[r, trial, :] = get_spikecounts_fixed_time_window(trial_dict['spikes_E'],
                                                                               time_offset, delta_t) # for E neurons
            # spike_count_mat_I[r, trial, :] = get_spikecounts_fixed_time_window(trial_dict['spikes_I'],
            #                                                                    time_offset, delta_t) # for I neurons

            # get spike counts for sliding time windows
            spike_count_window_mat_E[r, trial, :, :] = get_spike_counts_over_time_windows(trial_dict['spikes_E'], time_offset,
                                                                                          delta_t, window_length)

        # calculate correlations
        correlations_E.append(calculate_correlation_matrix(spike_count_window_mat_E[r, ...]))
        fano_factors_E.append(calculate_fano_factor(spike_count_window_mat_E[r, ...]))

    # rate histogram
    rates = spike_count_mat_E.flatten() / delta_t  # mean over trials? no --> axis=0
    plt.figure(figsize=(10, 5))
    plt.hist(rates, bins=40, range=[0, 100], alpha=.7)
    plt.title('Firing rates in spikes / sec')
    plt.axvline(np.mean(rates), linestyle='--', label='mean={}'.format(np.round(np.mean(rates), 2)), color='C1')
    plt.legend()
    save_figure(filename=filename[:-2] + '_rate_hist.pdf')

    # correlation histogram
    rho = correlations_E[0]
    plt.figure(figsize=(10, 5))
    plt.hist(rho, bins=40, range=[-1, 1], alpha=.7)
    plt.axvline(np.mean(rho), linestyle='--', label='mean={}'.format(np.round(np.mean(rho), 2)), color='C1')
    plt.legend()
    save_figure(filename=filename[:-2] + '_rho_hist.pdf')

    # fano factor histogram
    ff = fano_factors_E[0]
    plt.figure(figsize=(10, 5))
    plt.hist(ff, bins=40, range=[0, 6], alpha=.7)
    plt.title('Fano factors over trials and {}s windows'.format(window_length))
    plt.axvline(np.mean(ff), linestyle='--', label='mean={}'.format(np.round(np.mean(ff), 2)), color='C1')
    plt.legend()
    save_figure(filename=filename[:-2] + '_ff_hist.pdf')
    plt.show()
