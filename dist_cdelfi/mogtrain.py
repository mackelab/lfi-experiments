import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import delfi.distribution as dd

from tqdm import tqdm

from pdf_mog import multivariate_mog_pdf

dtype = torch.FloatTensor

class MoG(nn.Module):
    def __init__(self, n_components, dim, dtype=dtype):
        """ Trainable MoG using torch.autograd 
        
        Parameters
        ---------
        n_components : int
            Number of components in the mixture
        dim : int
            Dimension of the distribution
        dtype : torch.dtype
            dtype for the variables
        """
        super().__init__()

        self.n_components = n_components
        self.dim = dim
        self.dtype = dtype
        
        self.softmax = nn.Softmax(dim=0)
        
        self.weights = nn.Parameter(
            torch.ones(n_components).type(self.dtype))
        self.mus = nn.Parameter(
            torch.Tensor(n_components, dim).normal_().type(self.dtype))
        self.Lvecs = nn.Parameter(
            torch.Tensor(n_components * (dim**2+dim) // 2).uniform_(0, 0.1).type(self.dtype))
    
    def eval(self, data):
        """Evaluate MoG at data points
        
        Parameters
        ---------
        data : array of size n x dim
            Data points at which the MoG is to be evaluated
        
        Returns
        ---------
        lprobs : array of size n
            Log probabilities of data points
        """
            
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
        data = Variable(data.type(self.dtype), requires_grad=False)
        
        # weights
        weights = self.softmax(self.weights)
                
        Ls = self.get_Ls()

        lprobs = multivariate_mog_pdf(data, weights, self.mus, Ls, log=True)
        
        return lprobs, (weights, self.mus, Ls)
        
    def get_Ls(self):
        """Compute Cholesky factors of the covariance matrices
        
        Returns
        -------
        Ls : array (n_components x n_dim x n_dim)
            Cholesky factors of the components
        """
        
        # covariance factors
        Ls = Variable(torch.zeros(self.n_components, self.dim, self.dim)).type(self.dtype)
        
        # assign vector to lower triangle of Ls
        (idx1, idx2) = np.tril_indices(self.dim)
        Ls[:, idx1, idx2] = self.Lvecs

        # apply exponential to get positive diagonal
        (idx1, idx2) = np.diag_indices(self.dim)
        Ls[:, idx1, idx2] = torch.exp(Ls[:, idx1, idx2])
        
        return Ls
        
    def forward(self, inp):
        """Evaluate the mean log probabilities of the data points 
        
        Parameters
        ----------
        inp : array (n x dim)
            Data points at which the MoG is to be evaluated
            
        Returns
        -------
        lprobs : dtype
            Mean log probability of the data points
        (weights, mus, Ls) : torch.Variable
            Parameters of the MoG
        """
        
        lprobs, (weights, mus, Ls) = self.eval(inp)
        return -torch.mean(lprobs), (weights, mus, Ls)
    
    def get_distribution(self):
        """Convert MoG to delfi.distribution.GaussianMixture
        
        Returns
        -------
        dist : delfi.distribution.GaussianMixture
            Converted MoG
        """
        
        a = self.softmax(self.weights).data.numpy()
        ms = self.mus.data.numpy()
        Ls = self.get_Ls().data.numpy()
        
        Us = np.linalg.inv(Ls)
        
        return dd.MoG(a=a, ms=ms, Us=Us)

    def set_distribution(self, target):
        if isinstance(target, dd.Gaussian):
            target = dd.MoG(a=[1], ms=[target.m], Ss=[target.S])
            
        if not isinstance(target, dd.MoG):
            raise TypeError("Cannot initialise trainable MoG to non-MoG target distribution")
            
        assert self.dim == target.ndim, "Cannot initialise trainable MoG to target distribution of wrong dimension"
            
        if target.ncomp != self.n_components:
            raise TypeError("Cannot initialise trainable MoG to MoG target distribution with different number of components")
        
        self.weights.data = dtype(np.log(target.a))
        self.mus.data = dtype([ np.copy(x.m) for x in target.xs ])
        
        Ls = np.copy([ np.linalg.cholesky(x.S) for x in target.xs ])
        (idx1, idx2) = np.diag_indices(self.dim)
        Ls[:,idx1,idx2] = np.log(Ls[:,idx1,idx2])
        
        # assign vector to lower triangle of Ls
        (idx1, idx2) = np.tril_indices(self.dim)
        self.Lvecs.data = dtype(Ls[:, idx1, idx2])
        
        
class MoGTrainer:
    def __init__(self, prop, prior, qphi, n_components, nsamples, lr=0.01, es_rounds=1000, es_thresh=0, dtype=dtype, init_to_qphi=True):
        """ Train a MoG to fit uncorrected posterior
        
        Parameters
        ----------
        prop : delfi.distribution.BaseDistribution
            Proposal prior
        prior : delfi.distribution.BaseDistribution
            True prior
        qphi : delfi.distribution.BaseDistribution
            Uncorrected posterior
        ncomponents : int
            Number of mixture components
        nsamples : int
            Number of samples drawn from proposal prior for training
        lr : float
            Learning rate of the optimiser
        es_rounds : int
            Number of previous rounds considered for early stopping (es_rounds=0 disables early stopping)
        es_thresh : float
            If mean decrease in loss over the last `es_rounds` rounds is less than this threshold, stop training
        dtype : dtype
            dtype of MoG 
        """
        
        self.prop = prop
        self.prior = prior
        self.qphi = qphi
        
        self.ndim = prior.ndim
        
        assert self.ndim == prop.ndim and self.ndim == qphi.ndim
        
        self.dtype = dtype
        
        self.mog = MoG(n_components=n_components, dim=self.ndim, dtype=dtype)
        
        self.redraw_samples(nsamples)
        
        self.lr = lr
        self.es_rounds = es_rounds
        self.es_thresh = es_thresh
        
        if init_to_qphi:
            try:
                self.mog.set_distribution(self.qphi)
            except TypeError:
                pass
        
    def redraw_samples(self, nsamples):
        """ Draw samples from proposal prior for Monte-Carlo simulations
        
        Parameters
        ----------
        nsamples : int
            Number of samples to be drawn
        """
        
        self.samples = self.prop.gen(n_samples=nsamples)
                     
        self.llsamples_prior = self.prior.eval(self.samples, log=True)
        self.llsamples_prop = self.prop.eval(self.samples, log=True)
        self.llsamples_qphi = self.qphi.eval(self.samples, log=True)
    
        logterm_fixed_ = torch.from_numpy(self.llsamples_prop - self.llsamples_prior - self.llsamples_qphi)
        self.logterm_fixed = Variable(logterm_fixed_.type(self.dtype), requires_grad=False)
        
        samples = self.qphi.gen(n_samples=nsamples)
        Z_prime = np.log(np.mean(self.prior.eval(self.samples, log=False)))
        self.Z_prime = Variable(torch.from_numpy(np.array([Z_prime])).type(self.dtype), requires_grad=False)
        
    def get_loss(self):
        """ Compute the loss function
        
        Returns
        -------
        loss : torch.Variable
            Loss of the MoG
        """
        
        llsamples_pbeta, _ = self.mog.eval(self.samples)
        logterm = llsamples_pbeta + self.logterm_fixed
                         
        lsamples_pbeta = torch.exp(llsamples_pbeta)
        Z_beta = torch.mean(lsamples_pbeta)

        eterm = torch.mean(lsamples_pbeta * logterm)

        loss = -torch.log(Z_beta) + eterm / Z_beta + self.Z_prime
        return loss
        
    def train(self, nsteps=-1, lr=None, es_rounds=None, es_thresh=None):
        """ Train MoG
        
        Parameters
        ----------
        nsteps : int
            Number of steps. If nsteps == -1, train until convergence.
        lr : float (default self.lr)
            Learning rate
        es_rounds : int (default self.es_rounds)
            Number of rounds to be considered for early stopping
        es_thresh : float (default self.es_thresh)
            Threshold for loss decrease in early stopping
            
        Returns
        -------
        losses : np.ndarray (1 dimensional)
            Value of the loss after each training step
        """
            
        lr = lr or self.lr
        if es_rounds == None:
            es_rounds = self.es_rounds
            
        if es_thresh == None:
            es_thresh = self.es_thresh
               
        if nsteps == -1 and es_rounds == 0:
            raise ValueError("Training MoG with indefinite number of steps and no convergence criterion")
            
        optim = torch.optim.Adam(self.mog.parameters(), lr=lr)
        
        # Progress bars cause bugs
        if nsteps == -1:
            progress = tqdm(iter(int, 1))
        else:
            progress = tqdm(range(nsteps))
        with progress:
            losses = []

            #for step in range(nsteps):
            for step in progress:
                optim.zero_grad()
                loss = self.get_loss()
                
                if len(losses) != 0 and np.abs(np.asscalar(loss.data.numpy()) - losses[-1]) > 1e3:
                    import pdb
                    pdb.set_trace()
                if np.any(np.isnan(loss.data.numpy())):
                    import pdb
                    pdb.set_trace()
                    
                loss.backward()      
                optim.step()

                progress.set_description("loss={}".format(loss.data.numpy()))
                losses.append(np.asscalar(loss.data.numpy()))

                # If early stopping is enabled and the average reduction in the loss over the last `es_rounds` rounds 
                # is less than `es_thresh`, stop training.
                if es_rounds and len(losses) > es_rounds:
                    if not np.mean(np.diff(losses[-es_rounds:])) < -es_thresh:
                        break

        return losses

    def get_mog(self):
        """Convert into delfi.distribution.MoG
        
        Returns
        -------
        mog : delfi.distribution.MoG
            Converted MoG
        """
        
        return self.mog.get_distribution()

class DefensiveDistribution(dd.BaseDistribution.BaseDistribution):
    """Defensive distribution
    """
    def __init__(self, prop, prior, alpha, seed=None):
        super().__init__(prop.ndim, seed=seed)

        self.prop = prop
        self.prior = prior
        self.alpha = alpha
        
        assert 0 <= alpha and alpha <= 1, "Alpha in DefensiveDistribution must be between 0 and 1"

    def eval(self, x, ii=None, log=True):
        eval_prop = self.prop.eval(x, ii=ii, log=False)
        eval_prior = self.prior.eval(x, ii=ii, log=False)
        
        ret = (1 - self.alpha) * eval_prop + self.alpha * eval_prior

        if log:
            return np.log(ret)
        else:
            return ret

    def gen(self, n_samples=1):
        prior_mask = self.rng.binomial(n=1, p=self.alpha, size=(n_samples,)).astype(bool)
        n_prior_draws = np.count_nonzero(prior_mask)
        ret = np.empty((n_samples, self.ndim))
        ret[prior_mask] = self.prior.gen(n_samples=n_prior_draws)
        ret[~prior_mask] = self.prop.gen(n_samples=n_samples-n_prior_draws)
        
        return ret
    
class DividedPdf:
    def __init__(self, a, b, norm_region):
        self.a = a
        self.b = b
        
        self.ndim = self.a.ndim
        self.Z = 1
        
        mass = self.get_mass(norm_region)
        self.Z = mass
        
        
    def get_mass(self, norm_region):
        dim = self.a.ndim

        unif = dd.Uniform(norm_region[0] * np.ones(dim), norm_region[1] * np.ones(dim))
        mgrid = unif.gen(5000000)

        N = (norm_region[1] - norm_region[0]) ** dim
        samples = self.eval(mgrid, log=False)

        mass = np.mean(samples) * N
        return mass
        
    def eval(self, samples, log=True):
        ret = self.a.eval(samples, log=log) / self.b.eval(samples, log=log)
        return ret / self.Z
    
def divide_dists(a, b, norm_region):
    if isinstance(a, dd.Gaussian) and isinstance(b, dd.Gaussian):
        return a / b
    
    return DividedPdf(a, b, norm_region)

class CroppedDistribution(dd.BaseDistribution.BaseDistribution):
    def __init__(self, base_dist, ref_dist, nsamples=10000):
        super().__init__(base_dist.ndim)
        self.base_dist = base_dist
        self.ref_dist = ref_dist

        self.Z = 1
        samples = self.ref_dist.gen(nsamples)
        self.Z = np.mean(self.eval(samples, log=False) / self.ref_dist.eval(samples, log=False))
    
    def eval(self, samples, log=True):
        if log:
            return self.base_dist.eval(samples, log=True) + np.log(self.mask(samples)) - np.log(self.Z)
        else:
            return self.base_dist.eval(samples, log=False) * self.mask(samples) / self.Z

    def gen(self, n_samples, n_reps=1):
        ret = self.base_dist.gen(n_samples)
        mask = ~self.mask(ret)

        while np.any(mask):
            ret[mask] = self.base_dist.gen(np.count_nonzero(mask))
            mask = ~self.mask(ret)

        return ret
        
    def mask(self, samples):
        return self.ref_dist.eval(samples, log=False) != 0
      