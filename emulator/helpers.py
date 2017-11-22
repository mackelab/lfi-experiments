import math
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle


def batch_generator(dataset, batch_size=5):
    shuffle(dataset)
    N_full_batches = len(dataset) // batch_size
    for i in range(N_full_batches):
        idx_from = batch_size * i
        idx_to = batch_size * (i + 1)
        xs, ys = zip(*[(x, y) for x, y in dataset[idx_from:idx_to]])
        yield xs, ys


def gauss_pdf(y, mu, sigma, log=False):
    # TODO: rethink torch.log
    result = -0.5*torch.log(2*np.pi*sigma**2*Variable(torch.ones(1))) - 1/(2*sigma**2) * (y - mu)**2
    if log:
        return result
    else: 
        return torch.exp(result)

def mdn_logloss(out_alpha, out_sigma, out_mu, y):
    # TODO: learn sigma
    result = (gauss_pdf(y, out_mu, Variable(torch.ones((1)))) * out_alpha).squeeze()
    result = torch.log(result)
    return result

def mdn_kl(mps, sps, wdecay):
    assert wdecay > 0.0
    n_params = sum([mp.nelement() for mp in mps])
    L1 = 0.5 * wdecay * (sum([torch.sum(mp**2) for mp in mps]) +
                         sum([torch.sum(torch.exp(sp * 2)) for sp in sps]))
    L2 = sum([torch.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))
    L = L1 - L2 - Lc
    return L


def log_posterior(model, prior_mean, prior_var, theta, obs, frozen=True):
    # log L(θ)p(θ)
    x_param = nn.Parameter(torch.Tensor(theta))
    y_var = Variable(torch.Tensor(obs))
    (out_alpha, out_sigma, out_mu) = model(x_param, frozen=frozen)
    lp  = mdn_logloss(out_alpha, out_sigma, out_mu, y_var)
    lp += gauss_pdf(x_param, prior_mean, math.sqrt(prior_var), 
                   log=True).squeeze()
    return x_param, lp

def al_loss(model, prior_mean, prior_var, theta, obs, beta=1., frozen=True):
    # C(θ) = β E[log L(θ)p(θ)] + (1-β) VAR[log L(θ)p(θ)]
    x_param, lp = log_posterior(model, prior_mean, prior_var, theta, obs, frozen)   
    Ef = lp.mean()
    Ef2 = (lp**2).mean()
    E2f = Ef**2
    C = - (1-beta) * Ef - beta * (Ef2 - E2f)
    C.backward()
    return C.data.numpy(), x_param.grad.data.numpy(), lp.data.numpy()


class FullyConnectedLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_samples=9, 
                 sigma_prior=0.1, svi=True, activation=nn.Tanh()):
        # TODO: make sure svi=False works
        super(FullyConnectedLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.sigma_prior = sigma_prior
        self.svi = svi
        self.activation = activation
        
        self.W_mu = nn.Parameter(torch.Tensor(n_inputs, n_outputs).normal_(0, 1/np.sqrt(n_inputs)))
        self.b_mu = nn.Parameter(torch.Tensor(n_outputs).uniform_(-0.01, 0.01))
        
        if self.svi:
            self.W_logsigma = nn.Parameter(torch.Tensor(n_inputs, n_outputs).uniform_(-4., -6.))
            self.b_logsigma = nn.Parameter(torch.Tensor(n_outputs).uniform_(-4., -6.))
            
        self.ua = None
            
    def forward(self, X, sample=True, frozen=False, debug=False):
        """
        Input  : (n_batch, n_input) or (n_samples, n_batch, n_input)
        Output : (n_samples, n_batch, n_outputs)
        """       
        # TODO: make sure sample=False works
        if sample:
            X = X[None, :, :] if X.dim() == 2 else X
        if debug: print('In  : {}'.format(X.size()))

        # mean activation
        ma = torch.matmul(X, self.W_mu) + self.b_mu
        
        # stochastic activation
        if self.svi and sample:
            sa = torch.matmul(X**2, torch.exp(2*self.W_logsigma)) + torch.exp(2*self.b_logsigma)
            if not frozen or self.ua is None:
                self.ua = Variable(torch.Tensor(self.n_samples, 1, self.n_outputs).normal_())
            output = torch.sqrt(sa)*self.ua + ma
        else:
            output = ma
        if debug: print('Out : {}'.format(output.size()))

        if self.activation is not None:
            return self.activation(output)
        else:
            return output

class MDN(nn.Module):
    def __init__(self, ndim_input=1, ndim_output=1, n_hiddens=[5], n_components=1, 
                 n_samples=1, svi=True):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.n_samples = n_samples
        self.svi = svi
        
        # convert n_hiddens if needed
        if type(n_hiddens) == list:
            self.n_hiddens = n_hiddens
        elif type(n_hiddens) == int:
            self.n_hiddens = [n_hiddens]
        else:
            raise ValueError
            
        # shared keyword arguments
        skwargs = {'n_samples': n_samples, 'svi': svi}

        n_ci = ndim_input
        self.layers = {}
        for idx, n_hidden in enumerate(self.n_hiddens):
            self.layers['fc_{}'.format(idx+1)] = FullyConnectedLayer(n_ci, n_hidden, **skwargs)
            n_ci = n_hidden
        
        self.alpha_out = FullyConnectedLayer(n_ci, n_components, activation=None, **skwargs)
        self.logsigma_out = FullyConnectedLayer(n_ci, n_components, activation=None, **skwargs)
        self.mu_out = FullyConnectedLayer(n_ci, n_components, activation=None, **skwargs)
        
        # activation for alpha output layer
        self.alpha_act = nn.Softmax()
    
    def forward(self, x, sample=True, frozen=False):
        out = x
        for idx in range(len(self.n_hiddens)):
            out = self.layers['fc_{}'.format(idx+1)](out, sample=sample, frozen=frozen)
        
        n_batch = out.size()[1]
        out_alpha = self.alpha_act(self.alpha_out(out, sample=sample, frozen=frozen).view(-1, self.n_components))
        out_sigma = torch.exp(self.logsigma_out(out, sample=sample, frozen=frozen).view(-1, self.n_components))
        out_mu = self.mu_out(out, sample=sample, frozen=frozen).view(-1, self.n_components)
        return (out_alpha, out_sigma, out_mu)
    
    @property
    def logsigma(self):
        output = []
        for k, v in self.named_parameters():
            if '_logsigma' in k:
                output.append(v)
        return output

    @property
    def mu(self):
        output = []
        for k, v in self.named_parameters():
            if '_mu' in k:
                output.append(v)
        return output