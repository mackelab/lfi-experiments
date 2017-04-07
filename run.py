from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import numpy as np
import os
import pdb
import sys
import time

#from lfmods.hh import HHSimulator
from lfmods.mog import MoGSimulator
from likelihoodfree.Inference import Inference

@click.command()
@click.argument('prefix', type=str)
@click.argument('sim', type=click.Choice(['mog', 'hh']))
@click.option('--debug/--no-debug', default=False, is_flag=True, 
              help='If True, will enter debugger on error')
@click.option('--device', default='cpu', 
              help='Device to compute on')
@click.option('--iw-loss/--no-iw-loss', default=False, is_flag=True, 
              help='Use IW loss?')
@click.option('--minibatch', default=50, 
              help='Number of samples per minibatch')
@click.option('--numerical-fix', default=False, is_flag=True, 
              help='Reparameterization for stability')
@click.option('--pdb-iter', default=None, 
              help='Number of iterations after which to debug')
@click.option('--seed', type=int, default=None, 
              help='If provided, network and simulation are seeded')
@click.option('--svi/--no-svi', default=False, is_flag=True, 
              help='Use SVI version?')
@click.option('--val', default=0, 
              help='Number of samples for validation')

def run(prefix, debug, device, iw_loss, minibatch, numerical_fix, 
        pdb_iter, seed, sim, svi, val):
    """See `run.py --help` for info on parameters."""

    # set env variables
    os.environ["THEANO_FLAGS"] = "device=" + device + ",floatX=float32,lib.cnmem=0.8"

    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_data'] = 'results/'+sim+'/data/'
    dirs['dir_nets'] = 'results/'+sim+'/nets/'
    dirs['dir_plots'] = 'results/'+sim+'/plots/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    try:
        # simulator
        if sim == 'mog':
            sim = MoGSimulator(seed=seed)
        else:
            raise ValueError('sim not implemented')
        
        # training
        lfi = Inference(prefix=prefix, 
                        sim=sim,
                        seed=seed,
                        **dirs)
        
        # first training iteration
        print('Iteration 1')
        net, props = lfi.net_create(svi=True)
        lfi.train(net=net, postfix='iter_1')

        # next training iteration
        print('Iteration 2')
        net, props = lfi.net_load(postfix='iter_1')
        lfi.train(net=net, postfix='iter_2')
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
