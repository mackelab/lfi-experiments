import numpy as np
import torch

from torch.autograd import Variable
from pdf_mvn import MultivariateNormal


def multivariate_mog_pdf(X, weights, mus, Ls, log=False):
    """Mixture of Gaussians

    Calculate pdf values for a batch under a MoG.

    Parameters
    ----------
    X : Variable of shape (batch_size, ndims)
        Data
    weights : Variable of shape (ncomponents)
        Mixture weights
    mus : Variable of shape (ncomponents, ndims)
        Mixture means
    Ls: Variable of shape (ncomponents, ndims, ndims)
        Mixture precision factors
    log: bool
        If True, log probs are returned

    Returns
    -------
    result :  Variable containing a Tensor with shape (batch_size, 1)
        batch of density values, if log=True log probs
    """
    n_batch, _ = X.data.size()
    n_components, n_dims = mus.size()

    if type(X.data) == torch.cuda.FloatTensor:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    log_probs_mat = Variable(torch.zeros(n_batch, n_components).type(dtype), requires_grad=False)

    # take weighted sum over components to get log probs
    for k in range(n_components):
        mvn = MultivariateNormal(loc=mus[k, :], scale_tril=Ls[k, :, :])
        log_probs_mat[:, k] = mvn.log_prob(X)

    # log sum_k alpha_k * N(Y|mu, sigma)
    log_probs_batch = logsumexp(torch.log(weights) + log_probs_mat, dim=1)

    return log_probs_batch if log else torch.exp(log_probs_batch)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp

    Parameters
    ----------
    inputs : Variable
    dim : int
    keepdim : bool

    Returns
    -------
    Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim))
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
