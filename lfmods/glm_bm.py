import numpy as np
import scipy.signal as ss

class GLM:
    def __init__(self, params, seed=None):
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, t, I, max_n_steps=float('inf')):
        """Simulates the model for a specified time duration."""

        b0 = self.params[0,0]
        b0.astype(float)
        h = self.params[0,1:]
        h.astype(float)

        ########################################################################
        # simulation
        N = 1   # Number of trials

        psi = b0 + ss.lfilter(h, 1, I.reshape(-1,1), axis=0)

        # psi goes through a sigmoid non-linearity, returning a firing probability
        z = 1 /(1 + np.exp(-psi))

        # sample the spikes
        y = self.rng.uniform(size=(len(t), N)) < z
        y = np.sum(y, axis=1)

        return np.array(y).reshape(-1,1)
