import math
from numbers import Number

import torch
from torch.autograd import Variable
#from torch.distributions import constraints
#from torch.distributions.distribution import Distribution
#from torch.distributions.utils import broadcast_all, lazy_property

from collections import namedtuple
from functools import update_wrapper

import torch
from torch.autograd import Variable
import warnings




def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.Tensor` and `torch.autograd.Variable` instances are broadcasted as
        per the `broadcasting rules
        <http://pytorch.org/docs/master/notes/broadcasting.html>`_
      - numbers.Number instances (scalars) are upcast to Variables having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to Variables having size
        `(1,)`.

    Args:
        values (list of `numbers.Number`, `torch.autograd.Variable` or
        `torch.Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number`, `torch.Tensor`
            or `torch.autograd.Variable` instance
    """
    values = list(values)
    scalar_idxs = [i for i in range(len(values)) if isinstance(values[i], Number)]
    tensor_idxs = [i for i in range(len(values)) if
                   torch.is_tensor(values[i]) or isinstance(values[i], Variable)]
    if len(scalar_idxs) + len(tensor_idxs) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number, torch.Tensor or ' +
                         'torch.autograd.Variable.')
    if tensor_idxs:
        broadcast_shape = _broadcast_shape([values[i].size() for i in tensor_idxs])
        for idx in tensor_idxs:
            values[idx] = values[idx].expand(broadcast_shape)
        template = values[tensor_idxs[0]]
        if len(scalar_idxs) > 0 and not isinstance(template, torch.autograd.Variable):
            raise ValueError(('Input arguments containing instances of numbers.Number and torch.Tensor '
                              'are not currently supported.  Use torch.autograd.Variable instead of torch.Tensor'))
        for idx in scalar_idxs:
            values[idx] = template.new(template.size()).fill_(values[idx])
    else:
        for idx in scalar_idxs:
            values[idx] = variable(values[idx])
    return values



class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value

class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False

    def __init__(self, batch_shape=torch.Size(), event_shape=torch.Size()):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

    @property
    def batch_shape(self):
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def params(self):
        """
        Returns a dictionary from param names to `Constraint` objects that
        should be satisfied by each parameter of this distribution. For
        distributions with multiple parameterization, only one complete
        set of parameters should be specified in `.params`.
        """
        raise NotImplementedError

    @property
    def support(self):
        """
        Returns a `Constraint` object representing this distribution's support.
        """
        raise NotImplementedError

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        warnings.warn('sample_n will be deprecated. Use .sample((n,)) instead', UserWarning)
        return self.sample(torch.Size((n,)))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def cdf(self, value):
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def icdf(self, value):
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def enumerate_support(self):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched variables in lock-step
        `[[0, 0], [1, 1], ...]`. To iterate over the full Cartesian product
        use `itertools.product(m.enumerate_support())`.

        Returns:
            Variable or Tensor iterating over dimension 0.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor or Variable of shape batch_shape.
        """
        raise NotImplementedError

    def _extended_shape(self, sample_shape=torch.Size()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        shape = torch.Size(sample_shape + self._batch_shape + self._event_shape)
        if not shape and not torch._C._with_scalars():
            shape = torch.Size((1,))
        return shape

    def _validate_log_prob_arg(self, value):
        """
        Argument validation for `log_prob` methods. The rightmost dimensions
        of a value to be scored via `log_prob` must agree with the distribution's
        batch and event shapes.

        Args:
            value (Tensor or Variable): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not (torch.is_tensor(value) or isinstance(value, Variable)):
            raise ValueError('The value argument to log_prob must be a Tensor or Variable instance.')

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                             format(value.size(), self._event_shape))

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                                 format(actual_shape, expected_shape))

    def __repr__(self):
        return self.__class__.__name__ + '()'

def batch_tril(bmat, diagonal=0):
    """
    Given a batch of matrices, returns the lower triangular part of each matrix, with
    the other entries set to 0. The argument `diagonal` has the same meaning as in
    `torch.tril`.
    """
    if bmat.dim() == 2:
        return bmat.tril(diagonal=diagonal)
    else:
        return bmat * torch.tril(bmat.new(*bmat.shape[-2:]).fill_(1.0), diagonal=diagonal)

def _get_batch_shape(bmat, bvec):
    """
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    return torch.Size(vec_shape[:-1])


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    n = bvec.size(-1)
    batch_shape = _get_batch_shape(bmat, bvec)

    # to conform with `torch.bmm` interface, both bmat and bvec should have `.dim() == 3`
    bmat = bmat.expand(batch_shape + (n, n)).contiguous().view((-1, n, n))
    bvec = bvec.unsqueeze(-1).expand(batch_shape + (n, 1)).contiguous().view((-1, n, 1))
    return torch.bmm(bmat, bvec).squeeze(-1).view(batch_shape + (n,))


def _batch_potrf_lower(bmat):
    """
    Applies a Cholesky decomposition to all matrices in a batch of arbitrary shape.
    """
    n = bmat.size(-1)
    cholesky = torch.stack([C.potrf(upper=False) for C in bmat.unsqueeze(0).contiguous().view((-1, n, n))])
    return cholesky.view(bmat.shape)


def _batch_diag(bmat):
    """
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.contiguous().view(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).contiguous().view((-1,) + L.shape[-2:])
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)



class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): positive-definite covariance matrix
        scale_tril (Tensor or Variable): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of `covariance_matrix` or `scale_tril` can be specified.

        Using `scale_tril` will be more efficient: all computations internally
        are based on `scale_tril`. If `covariance_matrix` is passed instead,
        this is only used to compute the corresponding lower triangular
        matrices.
    """
    #params = {'loc': constraints.real_vector}
    #support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        event_shape = torch.Size(loc.shape[-1:])
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified (but not both).")
        if scale_tril is None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be two-dimensional")
            self.covariance_matrix = covariance_matrix
            batch_shape = _get_batch_shape(covariance_matrix, loc)
        else:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be two-dimensional")
            self.scale_tril = scale_tril
            batch_shape = _get_batch_shape(scale_tril, loc)
        self.loc = loc
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @lazy_property
    def scale_tril(self):
        return _batch_potrf_lower(self.covariance_matrix)

    @lazy_property
    def covariance_matrix(self):
        # To use torch.bmm, we first squash the batch_shape into a single dimension
        flat_scale_tril = self.scale_tril.unsqueeze(0).contiguous().view((-1,) + self._event_shape * 2)
        return torch.bmm(flat_scale_tril, flat_scale_tril.transpose(-1, -2)).view(self.scale_tril.shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self.scale_tril, diff)
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det

    def entropy(self):
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        H = 0.5 * (1.0 + math.log(2 * math.pi)) * self._event_shape[0] + log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
