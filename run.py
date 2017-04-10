from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import numpy as np
import os
import pdb
import six
import sys
import time

from ast import literal_eval
#from lfmods.hh import HHSimulator
from lfmods.mog import MoGSimulator
from likelihoodfree.Inference import Inference

@click.command()
@click.argument('model', type=click.Choice(['mog', 'hh']))
@click.argument('prefix', type=str)
@click.option('--debug/--no-debug', default=False, is_flag=True,
              help='If True, will enter debugger on error')
@click.option('--device', default='cpu',
              help='Device to compute on')
@click.option('--iw-loss/--no-iw-loss', default=False, is_flag=True,
              help='Use IW loss?')
@click.option('--pdb-iter', type=int, default=None,
              help='Number of iterations after which to debug')
@click.option('--prior-alpha', type=float, default=0.25,
              help='If provided, will use alpha as weight for true prior \
                    in proposal dstribution (only used if `iw_loss` is True)')
@click.option('--rep', type=str, default='2,2',
              help='Specify the number of repetitions per n_components model, \
                    seperation by comma')
@click.option('--seed', type=int, default=None,
              help='If provided, network and simulation are seeded')
@click.option('--sim-kwargs', type=str, default=None,
              help='If provided, will turned into dict and passed as kwargs to simulator')
@click.option('--svi/--no-svi', default=False, is_flag=True,
              help='Use SVI version?')
@click.option('--train-kwargs', type=str, default=None,
              help='If provided, will turned into dict and passed as kwargs to \
                    inference.train')
@click.option('--val', default=0,
              help='Number of samples for validation')
def run(model, prefix, debug, device, iw_loss, pdb_iter, prior_alpha, rep,
        sim_kwargs, seed, svi, train_kwargs, val):
    """Run model

    Call `run.py` together with a prefix and a model to run.

    See `run.py --help` for info on parameters.
    """
    # set env variables
    os.environ["THEANO_FLAGS"] = "device=" + device + ",floatX=float32,lib.cnmem=0.8"

    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_data'] = 'results/'+model+'/data/'
    dirs['dir_nets'] = 'results/'+model+'/nets/'
    dirs['dir_plots'] = 'results/'+model+'/plots/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    # net_kwargs and inference_kwargs to dicts
    # http://stackoverflow.com/a/9305396
    def string_to_kwargs(s):
        if s is None:
            return {}
        else:
            return dict((k, literal_eval(v)) for k, v in (pair.split('=') for pair in s.split()))
    train_kwargs = string_to_kwargs(train_kwargs)
    sim_kwargs = string_to_kwargs(sim_kwargs)

    try:
        # simulator
        if model == 'mog':
            sim = MoGSimulator(seed=seed, **sim_kwargs)
        else:
            raise ValueError('sim not implemented')

        # training
        lfi = Inference(prefix=prefix,
                        seed=seed,
                        sim=sim,
                        **dirs)

        created = False
        iteration = 0
        n_components = 0

        for r in [int(r) for r in rep.split(',')]:
            n_components += 1
            for i in range(r):
                iteration += 1
                print('Iteration {}; {} Component(s)'.format(iteration, n_components))

                if not created:
                    net, props = lfi.net_create(iw_loss=iw_loss,
                                                n_components=n_components,
                                                svi=svi)
                    created = True
                else:
                    path_posterior = '{}{}_iter_{}_posterior.pkl'.format(dirs['dir_nets'],
                        prefix, iteration-1)
                    approx_posterior = io.load(path_posterior)

                    if not iw_loss:
                        prior_alpha = None

                    net, props = lfi.net_reload(n_components=n_components,
                                                postfix='iter_{}'.format(iteration-1),
                                                prior_alpha=prior_alpha,
                                                prior_proposal=approx_posterior)

                lfi.train(net=net, postfix='iter_{}'.format(iteration), **train_kwargs)

    except:
        t, v, tb = sys.exc_info()
        if debug:
            print('')
            print('Exception')
            print(v.with_traceback(tb))
            pdb.post_mortem(tb)
        else:
            raise v.with_traceback(tb)

if __name__ == '__main__':
    run()
