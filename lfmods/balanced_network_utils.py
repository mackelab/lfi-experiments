import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def get_cluster_connection_probs(REE, k, p_ee):
    """
    When using clustered connectivity in the E population the avergae sparseness should still be p_ee.
    For a given clustering coef REE this method calculates the sparseness within, p_in, and between, p_out, clusters.
    For the uniform case, REE=1, p_ee = p_in = p_out.
    :param REE:
    :param k:
    :param p_ee:
    :return:
    """
    p_out = p_ee * k / (REE + k - 1)
    p_in = REE * p_out
    return p_in, p_out


def plot_spike_trains(spikemonitor, time_setoff=0.):
    spiketimes = np.asarray(spikemonitor.spike_trains())
    for idx in range(len(spiketimes)):
        times = spiketimes[idx][spiketimes[idx] > time_setoff]
        plt.plot(times, idx * np.ones_like(times), '.b', markersize=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')


def plot_traces(statemonitor, time_setoff=0.):
    time_mask = np.asarray(statemonitor.t) > time_setoff

    plt.subplot(211)
    plt.plot(statemonitor.t[time_mask], statemonitor.v[0, time_mask], label='vE')
    plt.legend()
    plt.subplot(212)
    plt.plot(statemonitor.t[time_mask], statemonitor.I_e[0, time_mask], label='Ie')
    plt.plot(statemonitor.t[time_mask], statemonitor.I_i[0, time_mask], label='Ii')
    plt.legend(loc='best')
    plt.xlabel('Time (ms)')


def select_spiketimes(spiketimes, t, delta_t):
    spiketimes = np.asarray(spiketimes)
    timemask = np.logical_and(spiketimes >= t, spiketimes <= t + delta_t)
    return spiketimes[timemask]


def get_spiketimes_for_time_window(spiketime_dict, t, delta_t):
    return {k: select_spiketimes(v, t, delta_t) for k, v in spiketime_dict.items()}


def get_spikecounts_for_time_window(spiketime_dict, t, delta_t):
    # get spike times for
    spiketime_dict_window = get_spiketimes_for_time_window(spiketime_dict, t, delta_t)
    spikecounts = [spiketime_array.size for spiketime_array in spiketime_dict_window.values()]
    return np.array(spikecounts)


def calculate_spike_counts_over_windows(spiketime_dict, t, delta_t, window_length):
    n_neurons = len(spiketime_dict.keys())
    length_of_recording = delta_t
    n_time_windows = int(length_of_recording / window_length)
    spike_counts_windows = np.zeros((n_neurons, n_time_windows))

    for window_idx in range(n_time_windows):
        wt = t + window_idx * window_length
        spike_counts_windows[:, window_idx] = get_spikecounts_for_time_window(spiketime_dict,
                                                                              t=wt,
                                                                              delta_t=window_length)
    return spike_counts_windows


def calculate_correlation_matrix(spikecount_matrix_windows):
    n_trials, n_neurons, n_time_windows = spikecount_matrix_windows.shape

    # prelocate the cov matrix
    cov = np.zeros((n_neurons, n_neurons))
    for trial in range(n_trials):
        # just add them up over trials
        cov += np.cov(spikecount_matrix_windows[trial, ...])
    # average across trials
    cov /= n_trials

    # get the mask of spiking neurons idx
    spiking_mask = np.logical_not(np.diag(cov).copy() == 0)

    # remove silent neurons from the analysis
    temp_cov = cov[spiking_mask, :]
    new_cov = temp_cov[:, spiking_mask]
    var = np.diag(new_cov).copy()

    # use the outer product over the variance vector to do it vectorized
    rho = new_cov / np.sqrt(np.outer(var, var))
    assert(np.sum(np.isnan(rho)) == 0), 'there are nan in the correlation matrix'

    # remove the diagonal elements and the lower triangle because we are interested cross correlations only
    matrix_mask = np.tril_indices(rho.shape[0])
    rho[matrix_mask] = np.inf
    reduced_rho = rho[np.isfinite(rho)]
    return reduced_rho


def calculate_fano_factor(spike_counts):
    """
    Calculates ff over trials and time windows. Assumes the spike_counts matrix to have shape
    n_trials x n_neurons x n_time_windows
    """
    count_variance = np.var(spike_counts, axis=(2)).flatten()  # over trials or windows or both?
    count_mean = np.mean(spike_counts, axis=(2)).flatten()

    # we have to exclude silent neurons
    spiking_mask = np.logical_not(count_mean == 0)
    ff = count_variance[spiking_mask] / count_mean[spiking_mask]
    return ff


def save_data(data, filename, folder):
    full_path_to_file = os.path.join(folder, filename + '.p')
    with open(full_path_to_file, 'wb') as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename, folder):
    full_path = os.path.join(folder, filename)
    return pickle.load(open(full_path, 'rb'))


def save_figure(filename, folder='/Users/Jan/Dropbox/Master/mackelab/code/balanced_clustered_network/figures/'):
    plt.savefig(os.path.join(folder, filename))