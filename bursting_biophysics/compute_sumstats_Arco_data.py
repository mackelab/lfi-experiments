from __future__ import division

import copy
import multiprocessing
import multiprocessing.pool
import numpy as np
import os
import pickle
import sys
import time
import utils

from os import listdir


##########################################################################################
# add observation noise to voltage traces
def add_observation_noise(voltage_traces):
    nois_fact_obs = 1.5 #1.5
    protocols = voltage_traces.keys()
    for protocol_name in protocols:
        num_tstep = len(voltage_traces[protocol_name]['tVec'])
        voltage_traces[protocol_name]['vList'] = list(voltage_traces[protocol_name]['vList'])
        for i in range(np.shape(voltage_traces[protocol_name]['vList'])[0]):
            voltage_traces[protocol_name]['vList'][i] = voltage_traces[protocol_name]['vList'][i] + nois_fact_obs*np.random.randn(num_tstep)
    return voltage_traces


##########################################################################################
# load each file, compute and merge respective summary statistics (for memory efficiency)

# number of parallel processes (smaller or equal to n_sims)
n_processes = 10

# compute summary statistics
def calc_sum_stats(x):
    x = add_observation_noise(x)
    return utils.summary_stats(x, n_xcorr=0, n_mom=4)


# load files and compute summary statistics
dir_samples = '/media/pedro/3f73e907-0de1-4e3d-8c7b-fadc5398dbdc/Arco_data/'
file_ls = [f for f in listdir(dir_samples) if f.endswith('.pickle')]

file_exclude_ls = []
params_ls = []
stats_ls = []
for f in file_ls:
    with open(dir_samples+f,'rb') as data1:
#         data = pickle.load(data1,encoding='latin1')
        data = pickle.load(data1)
        try:
            params = np.asarray([data[i][0] for i in range(len(data))])
            params_ls.append(params)
            x = [data[i][1] for i in range(len(data))]           
            pool = multiprocessing.pool.Pool(n_processes)
            stats_ls.append(pool.map(calc_sum_stats, x))
            pool.close()
            pool.join()
        except:
            file_exclude_ls.append(f)
            continue

            
n_params = np.shape(data[0][0])[0]
n_summary_stats = np.shape(stats_ls)[2]

params_mat = np.reshape(params_ls,(-1,n_params))
stats_mat = np.reshape(stats_ls,(-1,n_summary_stats))
file_mat = np.asarray(file_ls)

# save computed summary statistics
outfile = 'Arco_data_our_sumstats_v1.npz'
np.savez_compressed(outfile, files=file_mat, params=params_mat, data=stats_mat)