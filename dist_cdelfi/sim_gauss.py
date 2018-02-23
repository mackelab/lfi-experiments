import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class Gauss(BaseSimulator):
    def __init__(self, S, n=1, seed=None):
        """Gauss simulator

        Toy model that draws data from a distribution centered on theta with
        fixed noise.

        Parameters
        ----------
        S : float
            Covariance of noise on observations
        n : int
            Number of samples to draw
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=S.shape[0], seed=seed)
        self.S = S
        self.n = n

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        sample = dd.Gaussian(m=param, S=self.S,
                             seed=self.gen_newseed()).gen(self.n)

        return {'data': sample}
