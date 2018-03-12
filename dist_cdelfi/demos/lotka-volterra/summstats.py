from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
import numpy as np

class LotkaVolterraStats(BaseSummaryStats):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.n_summary = 9

    def calc(self, output):
        """
        Parameters
        ----------
        data: dict (entry 'data':array (...x2xN))

        Result
        ------
        summ: array (...xn_summary)
        """

        data = [ x['data'] for x in output ]
        data = np.asarray(data)
        if len(data.shape) <= 2:
            data = data.reshape((-1,*data.shape))

        assert(data.ndim >= 3)

        ms = np.mean(data, axis=-1, keepdims=True)
        s2s = np.var(data, axis=-1, keepdims=True, ddof=1)

        data_norm = (data - ms) / s2s

        lags = [1,2]

        ac = [ np.einsum('...j,...j->...', data[...,:-lag], data[...,lag:])  for lag in lags ]

        ac = np.einsum('i...->...i', ac) / data.shape[-1]

        cc = np.atleast_2d(np.einsum('...i,...i->...', data[...,0,:], data[...,1,:])) / data.shape[-1]

        return np.concatenate((ms.squeeze(-1),  # Undo keepdims
                               s2s.squeeze(-1), # Dito
                               ac.reshape((*ac.shape[:-2], 2 * len(lags))),
                               cc), axis=-1)
