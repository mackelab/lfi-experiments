import numpy as np

import BiophysModel_cython_comp

class HH:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

	def sim_time(self, dt, t, I, max_n_steps=float('inf')):
		"""Simulates the model for a specified time duration."""

		hh1comp.setparams(self.params)
		tstep = float(dt)

		V = np.zeros_like(t) # baseline voltage
		n = np.zeros_like(t)
		m = np.zeros_like(t)
		h = np.zeros_like(t)
		p = np.zeros_like(t)

		BiophysModel_cython_comp.computelf(t,I,V,m,n,h,p,tstep)
		return np.array(V).reshape(-1,1) + nois_fact_obs*self.rng.randn(t.shape[0],1)
