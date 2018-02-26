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
            torch.Tensor(int(n_components*((dim**2+dim)/2))).uniform_(0, 0.1).type(self.dtype))
    
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

class MoGTrainer:
    def __init__(self, prop, prior, qphi, ncomponents, nsamples, lr=0.01, es_rounds=0, es_thresh=0, dtype=dtype):
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
        self.dtype = dtype
        
        self.mog = MoG(n_components=ncomponents, dim=self.ndim, dtype=dtype)
        
        self.redraw_samples(nsamples)
        
        self.lr = lr
        self.es_rounds = es_rounds
        self.es_thresh = es_thresh
        
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

        eterm = torch.dot(lsamples_pbeta, logterm) / len(self.samples)

        loss = -torch.log(Z_beta) + eterm / Z_beta  
        return loss
        
    def train(self, nsteps, lr=None, es_rounds=None, es_thresh=None):
        """ Train MoG
        
        Parameters
        ----------
        nsteps : int
            Number of steps
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
               
        optim = torch.optim.Adam(self.mog.parameters(), lr=lr)
        
        # Progress bars cause bugs
        with tqdm(range(nsteps)) as progress:
            losses = []

            #for step in range(nsteps):
            for step in progress:
                optim.zero_grad()
                loss = self.get_loss()
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