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

#from lfmods.hh import HHSimulator
from lfmods.mog import MoGSimulator
from likelihoodfree.Inference import Inference

@click.command()
@click.argument('prefix', type=str)
@click.argument('model', type=click.Choice(['mog', 'hh']))
@click.option('--debug/--no-debug', default=False, is_flag=True,
              help='If True, will enter debugger on error')
@click.option('--device', default='cpu',
              help='Device to compute on')
@click.option('--iw-loss/--no-iw-loss', default=False, is_flag=True,
              help='Use IW loss?')
@click.option('--minibatch', default=100,
              help='Number of samples per minibatch')
@click.option('--pdb-iter', default=None,
              help='Number of iterations after which to debug')
@click.option('--rep', default=[2,2],
              help='List specifying the number of repetitions per n_components model')
@click.option('--seed', type=int, default=None,
              help='If provided, network and simulation are seeded')
@click.option('--svi/--no-svi', default=False, is_flag=True,
              help='Use SVI version?')
@click.option('--val', default=0,
              help='Number of samples for validation')

def run(prefix, model, debug, device, iw_loss, minibatch,
        pdb_iter, rep, seed, svi, val):
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

    try:
        # simulator
        if model == 'mog':
            dim = 1
            sim = MoGSimulator(dim=dim, seed=seed)
        else:
            raise ValueError('sim not implemented')

        # training
        lfi = Inference(prefix=prefix,
                        sim=sim,
                        seed=seed,
                        **dirs)

        created = False
        iteration = 0
        n_components = 0

        for r in rep:
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
                    net, props = lfi.net_reload(n_components=n_components,
                                                postfix='iter_{}'.format(iteration-1),
                                                prior_alpha=0.1,
                                                prior_proposal=approx_posterior)

                lfi.train(net=net, postfix='iter_{}'.format(iteration))

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
