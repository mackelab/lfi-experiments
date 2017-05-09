import numpy as np

from . import hh_bm_cython_comp

solver = hh_bm_cython_comp.forwardeuler
# solver = hh1comp.backwardeuler
# solver = hh1comp.hinesmethod

# def solver(t, I, V, m, n, h, p, dt, fineness)
# t: array of time steps
# I: array of I values
# V: array of V values (OUTPUT)
# m, n, h, p: buffers for gating variables
# dt: time step
# fineness: nr of iterations per time step (simulation dt = dt / fineness)
#
# The arrays must have the same size. The simulation runs until V is exhausted.

class HH:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        self.seed = seed
        if seed is not None:
            hh_bm_cython_comp.seed(seed)
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I, fineness=1, max_n_steps=float('inf')):
        """Simulates the model for a specified time duration."""

        hh_bm_cython_comp.setparams(self.params)
        tstep = float(dt)

        # explictly cast everything to double precision
        t = t.astype(np.float64)
        I = t.astype(np.float64)
        V = np.zeros_like(t).astype(np.float64)  # baseline voltage
        n = np.zeros_like(t).astype(np.float64)
        m = np.zeros_like(t).astype(np.float64)
        h = np.zeros_like(t).astype(np.float64)
        p = np.zeros_like(t).astype(np.float64)
        
        solver(t, I, V, m, n, h, p, tstep, fineness)

        return np.array(V).reshape(-1,1)
