import sys

dir_path = '/dss/dsshome1/lxc09/ga24sot2/biophysics/in_silico_framework/'
sys.path.append(dir_path)
import Interface as I

from biophysics_fitting import hay_complete_default_setup, L5tt_parameter_setup


def obs_params():
    """Parameters for x_o

    Returns
    -------
    true_params : dictionary of labels and floats
    """

    true_params = {'ephys.CaDynamics_E2.apic.decay': 137.86213603423801,
                     'ephys.CaDynamics_E2.apic.gamma': 0.0005793520824526776,
                     'ephys.CaDynamics_E2.axon.decay': 199.12980481497891,
                     'ephys.CaDynamics_E2.axon.gamma': 0.00061080490759830625,
                     'ephys.CaDynamics_E2.soma.decay': 152.16474193930151,
                     'ephys.CaDynamics_E2.soma.gamma': 0.0074243065368466803,
                     'ephys.Ca_HVA.apic.gCa_HVAbar': 0.0010965218089651857,
                     'ephys.Ca_HVA.axon.gCa_HVAbar': 0.00081867706020657862,
                     'ephys.Ca_HVA.soma.gCa_HVAbar': 0.00011435310571497434,
                     'ephys.Ca_LVAst.apic.gCa_LVAstbar': 0.0022763084379226854,
                     'ephys.Ca_LVAst.axon.gCa_LVAstbar': 0.0036986082079423594,
                     'ephys.Ca_LVAst.soma.gCa_LVAstbar': 0.00013883334761566004,
                     'ephys.Im.apic.gImbar': 3.2474860530531394e-06,
                     'ephys.K_Pst.axon.gK_Pstbar': 0.0054268374162654382,
                     'ephys.K_Pst.soma.gK_Pstbar': 0.10568666421909532,
                     'ephys.K_Tst.axon.gK_Tstbar': 0.048126766921039982,
                     'ephys.K_Tst.soma.gK_Tstbar': 0.094826660872338001,
                     'ephys.NaTa_t.apic.gNaTa_tbar': 0.013854989311151315,
                     'ephys.NaTa_t.axon.gNaTa_tbar': 3.9010342040060975,
                     'ephys.NaTa_t.soma.gNaTa_tbar': 3.8851157748263354,
                     'ephys.Nap_Et2.axon.gNap_Et2bar': 0.009964343408409574,
                     'ephys.Nap_Et2.soma.gNap_Et2bar': 0.0060004974488750964,
                     'ephys.SK_E2.apic.gSK_E2bar': 0.0012602755616811401,
                     'ephys.SK_E2.axon.gSK_E2bar': 0.013922406480998821,
                     'ephys.SK_E2.soma.gSK_E2bar': 0.062837104215625134,
                     'ephys.SKv3_1.apic.gSKv3_1bar': 6.6838213839617901e-05,
                     'ephys.SKv3_1.apic.offset': 0.08311048073340864,
                     'ephys.SKv3_1.apic.slope': -2.9836949894223825,
                     'ephys.SKv3_1.axon.gSKv3_1bar': 1.9642986130169147,
                     'ephys.SKv3_1.soma.gSKv3_1bar': 1.2999358521956366,
                     'ephys.none.apic.g_pas': 4.4931548434199036e-05,
                     'ephys.none.axon.g_pas': 2.062212836678345e-05,
                     'ephys.none.dend.g_pas': 4.2205984329741197e-05,
                     'ephys.none.soma.g_pas': 2.2409802171891654e-05,
                     'scale_apical.scale': 1.7109080877160283}

    return true_params


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


    mdb = I.ModelDataBase('/dss/dsshome1/lxc09/ga24sot2/biopysics/Data_arco/results/20190117_fitting_CDK_morphologies_Kv3_1_slope_variable_dend_scale')


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

    return voltage_traces


def summary_stats():
    """Summary statistics

    Parameters
    ----------
    x : dictionary
        Output of simulation
    Returns
    -------
    stats : array
    """

    return


def prior(true_params,seed=None):
    """Prior"""
    range_lower = np.array([.5,1e-4,1e-4,1e-4,50.,40.,1e-4,35.])
    range_upper = np.array([80.,15.,.6,.6,3000.,90.,.15,100.])

    prior_min = range_lower
    prior_max = range_upper

    return dd.Uniform(lower=prior_min, upper=prior_max, seed=seed)


def convert_np_to_series(x):
    """Function to convert numpy to pandas"""
    index = ['ephys.CaDynamics_E2.apic.decay',
     'ephys.CaDynamics_E2.apic.gamma',
     'ephys.CaDynamics_E2.axon.decay',
     'ephys.CaDynamics_E2.axon.gamma',
     'ephys.CaDynamics_E2.soma.decay',
     'ephys.CaDynamics_E2.soma.gamma',
     'ephys.Ca_HVA.apic.gCa_HVAbar',
     'ephys.Ca_HVA.axon.gCa_HVAbar',
     'ephys.Ca_HVA.soma.gCa_HVAbar',
     'ephys.Ca_LVAst.apic.gCa_LVAstbar',
     'ephys.Ca_LVAst.axon.gCa_LVAstbar',
     'ephys.Ca_LVAst.soma.gCa_LVAstbar',
     'ephys.Im.apic.gImbar',
     'ephys.K_Pst.axon.gK_Pstbar',
     'ephys.K_Pst.soma.gK_Pstbar',
     'ephys.K_Tst.axon.gK_Tstbar',
     'ephys.K_Tst.soma.gK_Tstbar',
     'ephys.NaTa_t.apic.gNaTa_tbar',
     'ephys.NaTa_t.axon.gNaTa_tbar',
     'ephys.NaTa_t.soma.gNaTa_tbar',
     'ephys.Nap_Et2.axon.gNap_Et2bar',
     'ephys.Nap_Et2.soma.gNap_Et2bar',
     'ephys.SK_E2.apic.gSK_E2bar',
     'ephys.SK_E2.axon.gSK_E2bar',
     'ephys.SK_E2.soma.gSK_E2bar',
     'ephys.SKv3_1.apic.gSKv3_1bar',
     'ephys.SKv3_1.apic.offset', #new: # potassium channel desnity on apical dendrite can be parameterized in a more complex way. In the paper, it was just constant. Now, it can increase or decrease and have a max oir min
     'ephys.SKv3_1.apic.slope', #new: # potassium channel desnity on apical dendrite can be parameterized in a more complex way. In the paper, it was just constant. Now, it can increase or decrease and have a max oir min
     'ephys.SKv3_1.axon.gSKv3_1bar',
     'ephys.SKv3_1.soma.gSKv3_1bar',
     'ephys.none.apic.g_pas',
     'ephys.none.axon.g_pas',
     'ephys.none.dend.g_pas',
     'ephys.none.soma.g_pas',
     'scale_apical.scale'] # this is an extension to the published version

    dict = {i: xx for i, xx in zip(index, x)}

    return I.pd.Series(dict)
