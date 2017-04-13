from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import numpy as np
import os
import pdb
import six
import subprocess
import sys
import time

from ast import literal_eval
from likelihoodfree.Inference import Inference

class ListIntParamType(click.ParamType):
    name = 'list of integers'
    def convert(self, value, param, ctx):
        try:
            if type(value) == list:
                return value
            elif type(value) == int:
                return [value]
            elif type(value) == str:
                return [int(i) for i in value.replace('[','').replace(']','').replace(' ','').split(',')]
            else:
                raise ValueError
        except ValueError:
            self.fail('%s is not a valid input' % value, param, ctx)

@click.command()
@click.argument('model', type=click.Choice(['gauss', 'hh', 'mog']))
@click.argument('prefix', type=str)
@click.option('--enqueue', default=False, is_flag=True, show_default=True,
              help='Enqueue the job rather than running it now. This requires a \
running worker process, which can be started with worker.py')
@click.option('--debug', default=False, is_flag=True, show_default=True,
              help='If provided, will enter debugger on error.')
@click.option('--device', default='cpu', show_default=True,
              help='Device to compute on.')
@click.option('--iw-loss', default=False, is_flag=True, show_default=True,
              help='If provided, will use importance weighted loss.')
@click.option('--nb', default=False, is_flag=True, show_default=True,
              help='If provided, will call nb.py after fitting.')
@click.option('--pdb-iter', type=int, default=None, show_default=True,
              help='Number of iterations after which to debug.')
@click.option('--prior-alpha', type=float, default=0.25, show_default=True,
              help='If provided, will use alpha as weight for true prior in \
proposal distribution (only used if iw_loss is True).')
@click.option('--rep', type=ListIntParamType(), default=[2,1], show_default=True,
              help='Specify the number of repetitions per n_components model, \
seperation by comma. For instance, \'2,1\' would mean that 2 itertions with \
1 component are run, and 1 iteration with 2 components are run.')
@click.option('--rnn', type=int, default=None, show_default=True,
              help='If specified, will use many-to-one RNN with specified \
number of hidden units instead of summary statistics.')
@click.option('--samples', default='2000', type=ListIntParamType(), show_default=True,
              help='Number of samples, provided as either a single number or \
as a comma seperated list. If a list is provided, say \'1000,2000\', \
1000 samples are drawn for the first iteration, and 2000 samples for the second \
iteration. If more iterations than elements in the list are run, 2000 samples \
will be drawn for those (last list element).')
@click.option('--seed', type=int, default=None, show_default=True,
              help='If provided, network and simulation are seeded')
@click.option('--sim-kwargs', type=str, default=None, show_default=True,
              help='If provided, will turned into dict and passed as kwargs to \
simulator.')
@click.option('--svi', default=False, is_flag=True, show_default=True,
              help='If provided, will use SVI version')
@click.option('--train-kwargs', type=str, default=None, show_default=True,
              help='If provided, will turned into dict and passed as kwargs to \
inference.train.')
@click.option('--true-prior', default=False, is_flag=True, show_default=True,
              help='If provided, will use true prior on all iterations.')
@click.option('--units', type=ListIntParamType(), default=[50], show_default=True,
              help='List of integers such that each list element specifies \
the number of units per fully connected hidden layer. The length of the list \
equals the number of hidden layers.')
@click.option('--val', type=int, default=0, show_default=True,
              help='Number of samples for validation.')
def run(model, prefix, enqueue, debug, device, iw_loss, nb, pdb_iter,
        prior_alpha, rep, rnn, samples, sim_kwargs, seed, svi, train_kwargs,
        true_prior, units, val):
    """Run model

    Call run.py together with a prefix and a model to run.

    See run.py --help for info on parameters.
    """
    # set env variables
    os.environ["THEANO_FLAGS"] = "device=" + device + ",floatX=float32,lib.cnmem=0.8"

    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_data'] = 'results/'+model+'/data/'
    dirs['dir_nets'] = 'results/'+model+'/nets/'
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
    sim_kwargs = string_to_kwargs(sim_kwargs)
    train_kwargs = string_to_kwargs(train_kwargs)

    try:
        if model == 'gauss':
            from lfmods.gauss import GaussSimulator as Simulator
        elif model == 'hh':
            from lfmods.hh import HHSimulator as Simulator
            if rnn is not None:
                sim_kwargs['pilot_samples'] = 0
                sim_kwargs['summary_stats'] = 0
        elif model == 'mog':
            from lfmods.mog import MoGSimulator as Simulator
        else:
            raise ValueError('could not import Simulator')

        sim = Simulator(seed=seed,
                        **sim_kwargs)

        lfi = Inference(prefix=prefix,
                        seed=seed,
                        sim=sim,
                        **dirs)

        created = False
        iteration = 0
        n_components = 0
        n_samples = []

        for r in rep:
            n_components += 1
            for i in range(r):
                iteration += 1
                print('Iteration {}; {} Component(s)'.format(iteration, n_components))

                if not created:
                    net, props = lfi.net_create(iw_loss=iw_loss,
                                                n_components=n_components,
                                                n_hiddens=units,
                                                svi=svi,
                                                rnn_hiddens=rnn)
                    created = True
                else:
                    path_posterior = '{}{}_iter_{}_posterior.pkl'.format(dirs['dir_nets'],
                        prefix, iteration-1)
                    approx_posterior = io.load(path_posterior)

                    if not iw_loss:
                        prior_alpha = None

                    if true_prior:
                        prior_alpha = None
                        approx_posterior = None

                    net, props = lfi.net_reload(n_components=n_components,
                                                postfix='iter_{}'.format(iteration-1),
                                                prior_alpha=prior_alpha,
                                                prior_proposal=approx_posterior)


                try:
                    n_samples = samples[iteration-1]
                except IndexError:
                    n_samples = samples[-1]

                lfi.train(debug=debug,
                          n_samples=n_samples,
                          net=net,
                          postfix='iter_{}'.format(iteration),
                          **train_kwargs)

        if nb:
            print('Making plots')
            if debug:
                debug_flag = ['--debug']
            else:
                debug_flag = []
            subprocess.call([sys.executable, 'nb.py', model, prefix] + debug_flag)

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
    args = sys.argv
    if '--enqueue' in args:
        from redis import Redis
        from rq import Queue

        timeout = int(1e6)
        ttl = -1
        connection = Redis()
        queue  = Queue('default', connection=connection)

        func_args = [sys.executable] + sys.argv
        func_args.remove('--enqueue')

        def subprocs_exec(args):
            subprocess.call(args, shell=False)

        queue.enqueue_call(func=subprocs_exec,
                           args=(func_args,),
                           timeout=timeout, ttl=ttl,
                           result_ttl=ttl)
        print('job enqueued')
    else:
        run()
