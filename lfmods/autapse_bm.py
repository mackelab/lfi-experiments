import numpy as np

class Autapse:
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

        J = self.params[0,0]
        J.astype(float)
        tau_x = self.params[0,1]
        tau_x.astype(float)
        tstep = float(dt)

        ####################################
        # fixed parameters
        # tau_x = 1         # time constant; 10 originally; 1 might blow up to nans / 0.1
        nois_fact = 15    # dynamics noise
        nois_fact_obs = 0.25 # observation noise

        ####################################
        # simulation from initial point
        x = np.zeros_like(t) # baseline activity

        for i in range(1, t.shape[0]):
            x[i] = x[i-1] + tstep*((J-1)*x[i-1] + I[i-1] + nois_fact*self.rng.randn()/(tstep**0.5))/tau_x

        return np.array(x).reshape(-1,1) + nois_fact_obs*self.rng.randn(t.shape[0],1)
