import numpy as np
from model import LotkaVolterraModel
from summstats import LotkaVolterraStats

import matplotlib.pyplot as plt

import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
from delfi.utils.viz import plot_pdf

from parameters import ParameterSet

params = ParameterSet({})

params.seed = 42

params.model = ParameterSet({})
params.model.dt = 0.2
params.model.T = 30

params.logparam_limits = [(-5,-2), (-5,1), (-5,1), (-5,-2)]

params.true_params = np.log([0.01, 0.5, 1, 0.01])

params.res = ParameterSet({})
params.res.n_hiddens = [20,20]
params.res.convert_to_T = None#3
params.res.pilot_samples = 1000
params.res.svi = False

params.train = ParameterSet({})
params.train.n_train = 5000
params.train.n_rounds = 2

params.n_cores = 2

mlist = [ LotkaVolterraModel(dt=params.model.dt, T=params.model.T, seed=params.seed + i) for i in range(params.n_cores) ]
m = mlist[0]
param_limits = np.array(params.logparam_limits).T

p = dd.Uniform(*param_limits)
s = LotkaVolterraStats()

sample = m.gen_single(params.true_params)
params.obs_stats = s.calc([sample])[0]

g = dg.MPGenerator(models=mlist, prior=p, summary=s)

params.save(url='setup.prm')

import run_abc

print("Starting")

samples, lweights, _, _ = run_abc.run_smc(mlist, p, s, params.obs_stats, n_params=p.ndim, 
                                          eps_last=0.01, eps_decay=0.9, n_particles=1000)
