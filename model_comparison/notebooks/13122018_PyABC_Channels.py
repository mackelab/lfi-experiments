# coding: utf-8

# ## SMC ABC on channel comparison

# In[1]:

import os
import tempfile
import numpy as np
import pickle
import scipy.stats as st

from pyabc import (ABCSMC, RV,
                   PercentileDistanceFunction, sampler)
from pyabc import Distribution as abcdis

import sys
sys.path.append('../../')
sys.path.append('../../../lfi-models/')

from lfimodels.channelomics.ChannelSingle import ChannelSingle
from lfimodels.channelomics.ChannelStats import ChannelStats
from matplotlib import pyplot as plt

from model_comparison.utils import *
from model_comparison.mdns import *
import warnings
warnings.filterwarnings('ignore')

# ## Define the channel model generators

# In[2]:

GT = {'kd': np.array([[4, -63, 0.032, 15, 5, 0.5, 10, 40]]),
      'kslow': np.array([[1, 35, 10, 3.3, 20]])}

LP = {'kd': ['power',r'$V_T$',r'$R_{\alpha}$',r'$th_{\alpha}$', r'$q_{\alpha}$', r'$R_{\beta}$', r'$th_{\beta}$',
             r'$q_{\beta}$'],
      'kslow': ['power', r'$V_T$', r'$q_p$', r'$R_{\tau}$', r'$q_{\tau}$']}

E_channel = {'kd': -90.0, 'kslow': -90.0}
fact_inward = {'kd': 1, 'kslow': 1}

prior_lims_kd = np.sort(np.concatenate((0.3 * GT['kd'].reshape(-1, 1), 1.3 * GT['kd'].reshape(-1, 1)), axis=1))
prior_lims_ks = np.sort(np.concatenate((0.3 * GT['kslow'].reshape(-1, 1), 1.3 * GT['kslow'].reshape(-1, 1)), axis=1))

cython = True
seed = 2


# In[3]:

n_params = 2
m_obs = ChannelSingle(channel_type='kd', n_params=8, cython=cython)
s = ChannelStats(channel_type='kd')

xo = m_obs.gen(GT['kd'].reshape(1,-1))
sxo = s.calc(xo[0])[:, :n_params]


# In[4]:


mkd = ChannelSingle(channel_type='kd', n_params=8, cython=cython, seed=seed)
skd = ChannelStats(channel_type='kd', seed=seed)

mks = ChannelSingle(channel_type='kslow', n_params=5, cython=cython, seed=seed)
sks = ChannelStats(channel_type='kslow', seed=seed)


# ## Define PyABC SMC models and priors

# In[5]:


# Define models oin pyabc style 
def model_1(parameters):
    params = np.array([parameters.p1, parameters.p2, parameters.p3, parameters.p4, 
                       parameters.p5, parameters.p6, parameters.p7, parameters.p8])
    x = mkd.gen(params.reshape(1, -1))
    sx = skd.calc(x[0])
    sxdict = dict()
    for i, sxi in enumerate(sx.T):
        sxdict['y{}'.format(i)] = sxi[0]
    return sxdict


def model_2(parameters):
    params = np.array([parameters.p1, parameters.p2, parameters.p3, parameters.p4, parameters.p5])
    x = mks.gen(params.reshape(1, -1))
    sx = skd.calc(x[0])
    sxdict = dict()
    for i, sxi in enumerate(sx.T):
        sxdict['y{}'.format(i)] = sxi[0]
    return sxdict

# priors
prior_dict_kd = dict()
for i in range(8): 
    prior_dict_kd['p{}'.format(i + 1)] = dict(type='uniform', 
                                              kwargs=dict(loc=prior_lims_kd[i, 0], 
                                                         scale=prior_lims_kd[i, 1] - prior_lims_kd[i, 0]))
    
prior1 = abcdis.from_dictionary_of_dictionaries(prior_dict_kd)

prior_dict_ks = dict()
for i in range(5): 
    prior_dict_ks['p{}'.format(i + 1)] = dict(type='uniform', 
                                              kwargs=dict(loc=prior_lims_ks[i, 0], 
                                                          scale=prior_lims_ks[i, 1] - prior_lims_ks[i, 0]))

prior2 = abcdis.from_dictionary_of_dictionaries(prior_dict_ks)

models = [model_1, model_2]
parameter_priors = [prior1, prior2]


# For model selection we usually have more than one model.
# These are assembled in a list. We
# require a Bayesian prior over the models.
# The default is to have a uniform prior over the model classes.
# This concludes the model definition.

# ### Configuring the ABCSMC run
# 
# Having the models defined, we can plug together the `ABCSMC` class.
# We need a distance function,
# to measure the distance of obtained samples.

# In[6]:


# We plug all the ABC options together
n_measure_to_use = 1
ss_measures = ['y{}'.format(i) for i in range(n_measure_to_use)]
abc = ABCSMC(
    models, parameter_priors,
    PercentileDistanceFunction(measures_to_use=ss_measures), sampler=sampler.SingleCoreSampler())


# ### Setting the observed data
# 
# Actually measured data can now be passed to the ABCSMC.
# This is set via the `new` method, indicating that we start
# a new run as opposed to resuming a stored run (see the "resume stored run" example).
# Moreover, we have to set the output database where the ABC-SMC run
# is logged.

# In[7]:


sx_t = sxo


# ## Run a loop over all test data points

# In[8]:


n_simulations = np.zeros(sx_t.shape[0])
phat = np.zeros((2, sx_t.shape[0]))

for idx, y_observed in enumerate(sx_t): 
    # y_observed is the important piece here: our actual observation.
    # and we define where to store the results
    db_path = ("sqlite:///" +
               os.path.join(tempfile.gettempdir(), "test.db"))
    abc_id = abc.new(db_path, {key: y_observed[i] for i, key in enumerate(ss_measures)})

    # We run the ABC until either criterion is met
    history = abc.run(minimum_epsilon=0.1, max_nr_populations=3)
    
    n_simulations[idx] = history.total_nr_simulations
    
    phat[:, idx] = history.get_model_probabilities().values[-1, :]


# In[ ]:


n_simulations.mean()


# In[ ]:


dd = dict(phat=phat, nsims=n_simulations, ptrue=ptrue)

import time
time_stamp = time.strftime('%Y%m%d%H%M_')
fn = os.path.join('../data/', time_stamp + '_SMCABC_results_PoissonNB_Ntest{}.p'.format(sx_t.shape[0]))

with open(fn, 'wb') as outfile: 
    pickle.dump(dd, outfile, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




