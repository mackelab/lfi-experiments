import numpy as np
import torch

from random import shuffle
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook


def batch_generator(dataset, batch_size=5):
    assert type(dataset) == list, 'dataset should be a list'
    shuffle(dataset)
    N_full_batches = len(dataset) // batch_size
    for i in range(N_full_batches):
        idx_from = batch_size * i
        idx_to = batch_size * (i + 1)
        yield np.asarray(dataset[idx_from:idx_to])

def train_sgd(X, model, optimizer=torch.optim.Adam,
              n_epochs=500, n_minibatch=None, verbose=True):
    N = len(X)
    if n_minibatch is None or n_minibatch > N:
        n_minibatch = N

    optim = optimizer(model.parameters())
    progress = progressbar(range(n_epochs)) if verbose else range(n_epochs)
    losses = []

    for epoch in progress:
        bgen = batch_generator(X, batch_size=n_minibatch)

        for j, x_batch in enumerate(bgen):
            loss, _ = model(x_batch)
            loss_np = loss.data.cpu().numpy()
            losses += [loss_np]

            optim.zero_grad()
            loss.backward()
            optim.step()

        if verbose:
            if (epoch + 1) % 1 == 0 or epoch == 0:
                progress.set_description("loss=%.4f" % loss_np)

    return np.asarray(losses)

class no_tqdm(object):
    def __enter__(self):
        class blank(object):
            def update(self, x):
                pass
        return blank()

    def __exit__(self, type, value, traceback):
        pass

    def update(self, i):
        pass

def progressbar(*args, **kwargs):
    """Creates a tqdm instance for a notebook or command line

    There is an open issue to support this as part of tqdm, see:
    https://github.com/tqdm/tqdm/issues/234
    https://github.com/tqdm/tqdm/issues/372
    """
    try:
        from IPython import get_ipython
        from ipywidgets import FloatProgress
        ipython = get_ipython()
        if not ipython or ipython.__class__.__name__ != 'ZMQInteractiveShell':
            raise RuntimeError
        return tqdm_notebook(*args, **kwargs)
    except BaseException:
        return tqdm(*args, **kwargs)
