from delfi.simulator import BaseSimulator
import MarkovJumpProcess as mjp
import numpy as np

class LotkaVolterraModel(BaseSimulator):
    def __init__(self, dt=0.2, T=30, init=[50,100], seed=None):
        super().__init__(dim_param=4, seed=seed)
        self.dt = dt
        self.T = T
        self.init = init

    def gen_single(self, params):
        lv = mjp.LotkaVolterra(self.init, params)

        states = lv.sim_time(self.dt, self.T)

        return { 'data' : np.asarray(states) }
