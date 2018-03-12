import numpy as np
from model import LotkaVolterraModel
from summstats import LotkaVolterraStats

import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer

from parameters import ParameterSet

import dill
import sys

####

params = ParameterSet(sys.argv[1])

params.train.n_train = 100000
params.train.n_rounds = 1

n_cores = 6 

####

mlist = [ LotkaVolterraModel(dt=params.model.dt, T=params.model.T, seed=params.seed + i) for i in range(n_cores) ]
m = mlist[0]

param_limits = np.reshape(params.logparam_limits, (2,1)) * np.ones((1,len(params.true_params)))

p = dd.Uniform(*param_limits)
s = LotkaVolterraStats()

sample = m.gen_single(params.true_params)
params.obs_stats = s.calc([sample])[0]

g = dg.MPGenerator(models=mlist, prior=p, summary=s)

### 

res = infer.SNPE(generator=g, 
                 obs=[params.obs_stats],
                 seed=params.seed, 
                 **params.res)

ret = res.run(**params.train)

with open("gt.pkl", "wb") as of:
    dill.dump(ret, of)
