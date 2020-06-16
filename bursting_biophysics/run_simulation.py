import sys
sys.path.append('../../')

import Interface as I
from biophysics_fitting import hay_complete_default_setup, L5tt_parameter_setup

mdb=I.ModelDataBase('/home/michael/Documents/Data_arco8/results/20190117_fitting_CDK_morphologies_Kv3_1_slope_variable_dend_scale')

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

### if you run 'params' (without the ' in a jupyter notebook cell, it will print the min and max of the prior)

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
    #s.setup.cell_modify_funs.append(('scale_apical', param_to_kwargs(scale_apical)))    
    return s

def get_Evaluator(mdb_setup, step = False):
    return hay_complete_default_setup.get_Evaluator(step = step)
    
def get_Combiner(mdb_setup, step = False):
    return hay_complete_default_setup.get_Combiner(step = step)

##############################################################################################################
##############################################################################################################
##############################################################################################################
# Morphology 91
##############################################################################################################
##############################################################################################################
##############################################################################################################

if not '91' in mdb.keys():
    mdb.create_sub_mdb('91')
    
if not 'morphology' in mdb['91'].keys():
    mdb['91'].create_managed_folder('morphology')

mdb['91']['morphology'].get_file('.hoc')


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
mdb['91'].setitem('get_Evaluator', I.partial(get_Evaluator, step = False), dumper = I.dumper_to_cloudpickle)
mdb['91'].setitem('get_Combiner', I.partial(get_Combiner, step = False), dumper = I.dumper_to_cloudpickle)

s = mdb['91']['get_Simulator'](mdb['91'])


# show all the keys
mdb['91'].keys()

s = mdb['91']['get_Simulator'](mdb['91'])
# last thing is just that you have to provide the simulator data again (not yet the parameters)
# s is now an instance of the simulator class

with I.silence_stdout:
    voltage_trace = s.run(params['max'])
       
def convert_np_to_series(x):
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
    
import numpy as np
# create a dummy parameter vector
x = np.random.rand(35)

# make a pandas dataframe from a numpy array
theta = convert_np_to_series(x)


# params is a pandas dataframe
theta = params['max']


with I.silence_stdout:
    voltage_trace = s.run(theta)

##################################################################################################################
# Plotting
##################################################################################################################
import matplotlib.pyplot as plt

#I.plt.plot(voltage_trace['BAC.hay_measure']['tVec'], voltage_trace['BAC.hay_measure']['vList'][2])
# first elemetn is somatic recording
# second is dendritic recording

# when you look at bAP, you have to pipets at the dendrite and therefore the list has 3 elements
# for BAC, there are only 2

##################################################################################################################
# Summstats
##################################################################################################################

summstats = mdb['91']['get_Evaluator'](mdb['91'])


# compute summary stats
stats = summstats.evaluate(voltage_trace)
print('Summary stats are', stats)

##################################################################################################################
# Combiner
##################################################################################################################
c = mdb['91']['get_Combiner'](mdb['91'])

# combiner takes the max of a group of summary stats
c.combine(summstats.evaluate(voltage_trace))


































