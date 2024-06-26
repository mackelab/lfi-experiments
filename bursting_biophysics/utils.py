import os
import socket

path1 = os.getcwd()
hostname = socket.gethostname()
if 'ga24sot2' in path1:  # michael LRZ account
    dir_path = '/dss/dsshome1/lxc09/ga24sot2/biophysics/in_silico_framework/'
#     dir_path = '../../in_silico_framework/'
    dir_data_path = '/dss/dsshome1/lxc09/ga24sot2/biophysics/Data_arco/results/'
#     dir_data_path = '../../Data_arco/results/'
elif 'ge57buf2' in path1:  # pedro LRZ account
    dir_path = '/dss/dsshome1/lxc0B/ge57buf2/in_silico_framework/'
    dir_data_path = '/dss/dsshome1/lxc0B/ge57buf2/Data_arco/results/'
elif hostname == 'dgkarl':  # michael workstation
    dir_path = '/home/michael/Documents/in_silico_framework/'
    dir_data_path = '/home/michael/Documents/Data_arco8/results/'
elif hostname == 'nsa3010':  # pedro workstation
    dir_path = '/home/pedro/repos/in_silico_framework/'
    dir_data_path = '/home/pedro/Documents/Data_arco/results/'
else:
    raise ValueError('Unknown hostname {}, add in if-else block'.format(hostname))

import sys
sys.path.append(dir_path)
import Interface as I
import numpy as np

from biophysics_fitting import hay_complete_default_setup, L5tt_parameter_setup
from parameter_setup import (load_param_names, load_ground_truth_params,
                            load_prior_min, load_prior_max)
from scipy import stats as spstats


def get_params(parameter_set):
    """
    Given a numpy array or list parameter_set, this returns the dictionary
    containing parameters.
    """

    keys = load_param_names()

    theta = {}
    for key, param in zip(keys, parameter_set):
        theta[key] = param

    return theta

def simulator_ss_wrapper(new_params, seed=None):
    """Wrapper for Arco's simulator from https://github.com/abast/in_silico_framework
    Also directly computes the hay summstats

    Parameters
    ----------
    new_params : dictionary
        Parameters for simulation
    Returns
    -------
    voltage_traces : dictionary
    """

    new_params = get_params(new_params)
    
    mdb = I.ModelDataBase(dir_data_path+'20190117_fitting_CDK_morphologies_Kv3_1_slope_variable_dend_scale')

    def get_template():
        param = L5tt_parameter_setup.get_L5tt_template()
        param.ApicalDendrite.mechanisms.range.SKv3_1 = I.scp.NTParameterSet({
                'slope': None,
                'distance': 'relative',
                'gSKv3_1bar': None,
                'offset': None,
                'spatial': 'linear'})
        param['cell_modify_functions'] = I.scp.NTParameterSet({'scale_apical': {'scale': None}})
        return param


    # params: a pandas dataframe with the parameternames as index and the columns min_ and max_
    # specifying the parameter boundaries
    params = hay_complete_default_setup.get_feasible_model_params().drop('x', axis = 1)
    params.index = 'ephys.' + params.index
    params = params.append(I.pd.DataFrame({'ephys.SKv3_1.apic.slope': {'min': -3, 'max': 0},
                                           'ephys.SKv3_1.apic.offset': {'min': 0, 'max': 1}}).T)
    params = params.append(I.pd.DataFrame({'min': .333, 'max': 3}, index = ['scale_apical.scale']))
    params = params.sort_index()


    def scale_apical(cell_param, params):
        assert(len(params) == 1)
        cell_param.cell_modify_functions.scale_apical.scale = params['scale']
        return cell_param


    from biophysics_fitting.parameters import param_to_kwargs
    def get_Simulator(mdb_setup, step = False):
        fixed_params = mdb_setup['get_fixed_params'](mdb_setup)
        s = hay_complete_default_setup.get_Simulator(I.pd.Series(fixed_params), step = step)
        s.setup.cell_param_generator = get_template
        s.setup.cell_param_modify_funs.append(('scale_apical', scale_apical))
        return s

    # morphology 91
    if not '91' in mdb.keys():
        mdb.create_sub_mdb('91')

    if not 'morphology' in mdb['91'].keys():
        mdb['91'].create_managed_folder('morphology')

    I.shutil.copy(dir_path+'MOEA_EH_model_visualization/morphology/CDK_morphologies_from_daniel/91_L5_CDK20050815_nr8L5B_dend_PC_neuron_transform_registered_C2.hoc', mdb['91']['morphology'].join('91_L5_CDK20050815_nr8L5B_dend_PC_neuron_transform_registered_C2.hoc'))

    mdb['91']['fixed_params'] = {'BAC.hay_measure.recSite': 734,
        'BAC.stim.dist': 734,
        'bAP.hay_measure.recSite1':734,
        'bAP.hay_measure.recSite2':914,
        'hot_zone.min_': 799,
        'hot_zone.max_': 999,
        'morphology.filename': None}

    def get_fixed_params(mdb_setup):
        fixed_params = mdb_setup['fixed_params']
        fixed_params['morphology.filename'] = mdb_setup['morphology'].get_file('hoc')
        return fixed_params


    mdb['91']['get_fixed_params'] = get_fixed_params
    mdb['91'].setitem('params', params, dumper = I.dumper_pandas_to_pickle)

    mdb['91'].setitem('get_Simulator', I.partial(get_Simulator, step = False), dumper = I.dumper_to_cloudpickle)

    s = mdb['91']['get_Simulator'](mdb['91'])

    with I.silence_stdout:
        voltage_traces = s.run(I.pd.Series(new_params))

    # add observation noise to voltage traces
    nois_fact_obs = 0 #1.5
    protocols = voltage_traces.keys()
    for protocol_name in protocols:
        num_tstep = len(voltage_traces[protocol_name]['tVec'])
        voltage_traces[protocol_name]['vList'] = list(voltage_traces[protocol_name]['vList'])
        for i in range(np.shape(voltage_traces[protocol_name]['vList'])[0]):
            voltage_traces[protocol_name]['vList'][i] = voltage_traces[protocol_name]['vList'][i] + nois_fact_obs*np.random.randn(num_tstep)
            
            
    ########################
    ########################
    ########################
    ########################
    summstats = mdb['91']['get_Evaluator'](mdb['91'])

    # compute summary stats
    stats = summstats.evaluate(voltage_traces)
    
    return stats



def simulator_wrapper(new_params, seed=None):
    """Wrapper for Arco's simulator from https://github.com/abast/in_silico_framework

    Parameters
    ----------
    new_params : dictionary
        Parameters for simulation
    Returns
    -------
    voltage_traces : dictionary
    """

    new_params = get_params(new_params)
    
    mdb = I.ModelDataBase(dir_data_path+'20190117_fitting_CDK_morphologies_Kv3_1_slope_variable_dend_scale')

    def get_template():
        param = L5tt_parameter_setup.get_L5tt_template()
        param.ApicalDendrite.mechanisms.range.SKv3_1 = I.scp.NTParameterSet({
                'slope': None,
                'distance': 'relative',
                'gSKv3_1bar': None,
                'offset': None,
                'spatial': 'linear'})
        param['cell_modify_functions'] = I.scp.NTParameterSet({'scale_apical': {'scale': None}})
        return param


    # params: a pandas dataframe with the parameternames as index and the columns min_ and max_
    # specifying the parameter boundaries
    params = hay_complete_default_setup.get_feasible_model_params().drop('x', axis = 1)
    params.index = 'ephys.' + params.index
    params = params.append(I.pd.DataFrame({'ephys.SKv3_1.apic.slope': {'min': -3, 'max': 0},
                                           'ephys.SKv3_1.apic.offset': {'min': 0, 'max': 1}}).T)
    params = params.append(I.pd.DataFrame({'min': .333, 'max': 3}, index = ['scale_apical.scale']))
    params = params.sort_index()


    def scale_apical(cell_param, params):
        assert(len(params) == 1)
        cell_param.cell_modify_functions.scale_apical.scale = params['scale']
        return cell_param


    from biophysics_fitting.parameters import param_to_kwargs
    def get_Simulator(mdb_setup, step = False):
        fixed_params = mdb_setup['get_fixed_params'](mdb_setup)
        s = hay_complete_default_setup.get_Simulator(I.pd.Series(fixed_params), step = step)
        s.setup.cell_param_generator = get_template
        s.setup.cell_param_modify_funs.append(('scale_apical', scale_apical))
        return s

    # morphology 91
    if not '91' in mdb.keys():
        mdb.create_sub_mdb('91')

    if not 'morphology' in mdb['91'].keys():
        mdb['91'].create_managed_folder('morphology')

    I.shutil.copy(dir_path+'MOEA_EH_model_visualization/morphology/CDK_morphologies_from_daniel/91_L5_CDK20050815_nr8L5B_dend_PC_neuron_transform_registered_C2.hoc', mdb['91']['morphology'].join('91_L5_CDK20050815_nr8L5B_dend_PC_neuron_transform_registered_C2.hoc'))

    mdb['91']['fixed_params'] = {'BAC.hay_measure.recSite': 734,
        'BAC.stim.dist': 734,
        'bAP.hay_measure.recSite1':734,
        'bAP.hay_measure.recSite2':914,
        'hot_zone.min_': 799,
        'hot_zone.max_': 999,
        'morphology.filename': None}

    def get_fixed_params(mdb_setup):
        fixed_params = mdb_setup['fixed_params']
        fixed_params['morphology.filename'] = mdb_setup['morphology'].get_file('hoc')
        return fixed_params


    mdb['91']['get_fixed_params'] = get_fixed_params
    mdb['91'].setitem('params', params, dumper = I.dumper_pandas_to_pickle)

    mdb['91'].setitem('get_Simulator', I.partial(get_Simulator, step = False), dumper = I.dumper_to_cloudpickle)

    s = mdb['91']['get_Simulator'](mdb['91'])

    with I.silence_stdout:
        voltage_traces = s.run(I.pd.Series(new_params))

    # add observation noise to voltage traces
    nois_fact_obs = 0 #1.5
    protocols = voltage_traces.keys()
    for protocol_name in protocols:
        num_tstep = len(voltage_traces[protocol_name]['tVec'])
        voltage_traces[protocol_name]['vList'] = list(voltage_traces[protocol_name]['vList'])
        for i in range(np.shape(voltage_traces[protocol_name]['vList'])[0]):
            voltage_traces[protocol_name]['vList'][i] = voltage_traces[protocol_name]['vList'][i] + nois_fact_obs*np.random.randn(num_tstep)
            
    return voltage_traces


def summary_stats(x, n_xcorr=0, n_mom=4):
    """Summary statistics

    Parameters
    ----------
    x : dictionary
        Output of simulation
    Returns
    -------
    stats : array
    """

    protocols = x.keys()
    t_on = 295

    sum_stats = []
    for protocol_name in protocols:
        for i in range(np.shape(x[protocol_name]['vList'])[0]):
            if i == 1:
                # soma
                threshold = -10
            else:
                # dendrite
                threshold = -40

            if protocol_name == 'BAC.hay_measure':
                t_off = t_on + 100
                spike_delay = 0.5
            elif protocol_name == 'bAP.hay_measure':
                t_off = x[protocol_name]['tVec'][-1]
                spike_delay = 1.

            # prepare trace data
            trace = {}
            trace['T'] = x[protocol_name]['tVec']
            trace['V'] = x[protocol_name]['vList'][i]

            # calculate summary statistics
            sum_stats1 = stats_calc(trace, t_on, t_off, n_xcorr, n_mom, threshold, spike_delay)
            sum_stats.append(sum_stats1)

    stats = np.concatenate(sum_stats).ravel()

    return np.asarray(stats)


def stats_calc(trace, t_on, t_off, n_xcorr, n_mom, threshold, spike_delay):
    """Calculates summary statistics
    Parameters
    ----------
    trace : dictionary
    Returns
    -------
    np.array, 1d with dimension n_mom+n_xcorr+3
    """
    t_original = trace['T']
    x_original = trace['V']
    
    # interpolate voltage (because numerical solver has adaptive timestep)
    t, x = np.arange(0,t_original[-1],0.025), np.interp(np.arange(0,t_original[-1],0.025), t_original, x_original)

    # initialise array of spike counts
    v = np.array(x)

    # put everything to threshold that is below threshold or has negative slope
    ind = np.where(v < threshold)
    v[ind] = threshold
    ind = np.where(np.diff(v) < 0)
    v[ind] = threshold

    # remaining negative slopes are at spike peaks
    ind = np.where(np.diff(v) < 0)
    spike_times = np.array(t)[ind]
    spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

    # number of spikes
    if spike_times_stim.shape[0] > 0:
        spike_times_stim = spike_times_stim[np.append(2*spike_delay, np.diff(spike_times_stim))>spike_delay]

    # resting potential and std
    rest_pot = np.mean(x[t<t_on])
    rest_pot_std = np.std(x[(t > .9*t_on) & (t < t_on)])

    # auto-correlations
    x_on_off = x[(t > t_on) & (t < t_off)]-np.mean(x[(t > t_on) & (t < t_off)])
    x_corr_val = np.dot(x_on_off,x_on_off)

    xcorr_steps = np.linspace(1.,n_xcorr*1.,n_xcorr).astype(int)
    x_corr_full = np.zeros(n_xcorr)
    for ii in range(n_xcorr):
        x_on_off_part = np.concatenate((x_on_off[xcorr_steps[ii]:],np.zeros(xcorr_steps[ii])))
        x_corr_full[ii] = np.dot(x_on_off,x_on_off_part)

    x_corr1 = x_corr_full/x_corr_val

    std_pw = np.power(np.std(x[(t > t_on) & (t < t_off)]), np.linspace(3,n_mom,n_mom-2))
    std_pw = np.concatenate((np.ones(1),std_pw))
    moments = spstats.moment(x[(t > t_on) & (t < t_off)], np.linspace(2,n_mom,n_mom-1))/std_pw

    # concatenation of summary statistics
    stats = np.concatenate((
            np.array([spike_times_stim.shape[0]]),
            x_corr1,
            np.array([rest_pot,rest_pot_std,np.mean(x[(t > t_on) & (t < t_off)])]),
            moments
        ))

    return np.asarray(stats)


def prior_around_gt(gt, fraction_of_full_prior, num_samples, seed=None):
    """
    Returns samples from a uniform prior around the np.array gt
    fraction_of_full_prior: float, what fraction of Arco's prior to use

    Note that, if gt is not exactlyt the mean of lower and upper bound, then
    fraction_of_full_prior will not be exaxtly the fraction of Arco's prior.
    Instead, one side will be capped by Arco's max (or min) value and the new
    prior will only extend in the other direction by fraction_of_full_prior/2

    returns: samples from new prior
    """

    # load Arco's prior bounds
    prior_min = np.asarray(load_prior_min())
    prior_max = np.asarray(load_prior_max())

    # get the width of the new prior
    prior_range = (prior_max - prior_min) * fraction_of_full_prior

    # find new prior bounds
    lower_bound = np.maximum(gt - prior_range / 2, prior_min)
    upper_bound = np.minimum(gt + prior_range / 2, prior_max)

    # seed prior
    np.random.seed(seed)

    # build numpy distribution
    samples = np.random.rand(num_samples, len(prior_min)) *\
                (upper_bound - lower_bound) + lower_bound

    return samples


# # load ground truth parameters that Arco gave us
# gt = load_ground_truth_params()
# # draw samples from prior around ground truth
# parameter_sets = prior_around_gt(gt, fraction_of_full_prior=0.1, num_samples=2)

# for parameter_set in parameter_sets:
#     # build dictionary
#     theta = get_params(parameter_set)
#     # simulate
#     output_trace = simulator_wrapper(theta)
#     print('===== Successfully finished =====')
