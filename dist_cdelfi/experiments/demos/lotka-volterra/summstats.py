from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
import numpy as np

class LotkaVolterraStats(BaseSummaryStats):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.n_summary = 9

    def calc(self, data):
        """
        Parameters
        ----------
        data: array (...x2N)

        Result
        ------
        summ: array (...xn_summary)
        """

        data = np.atleast_2d(data)
        data_shape = data.shape[:-2] + (2,data.shape[-1] // 2) 
        data = data.reshape(data_shape, order='F')

        ms = np.mean(data, axis=-1, keepdims=True)
        s2s = np.var(data, axis=-1, keepdims=True, ddof=1)

        data_norm = (data - ms) / s2s

        lags = [1,2]

        ac = np.tensordot(data[...,:-lags], data[...,lags:].T, axes=1) / data.shape[-1]

        cc = np.tensordot(data[...,0,:], data[...,1,:].T, axes=1) / data.shape[-1]

        return np.concatenate((ms.squeeze(-1),  # Undo keepdims
                               s2s.squeeze(-1), # Dito
                               ac.reshape(*ac.shape[:-2], 2 * len(lag)),
                               cc), axis=-1)
