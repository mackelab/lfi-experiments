from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import likelihoodfree.PDF as pdf
import numpy as np
import os
import pdb
import six
import subprocess
import sys
import time

from ast import literal_eval
from likelihoodfree.Kernel import Kernel

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

class ListFloatParamType(click.ParamType):
    name = 'list of floats'
    def convert(self, value, param, ctx):
        try:
            if type(value) == list:
                return value
            elif type(value) == int:
                return [float(value)]
            elif type(value) == float:
                return [value]
            elif type(value) == str:
                return [float(i) for i in value.replace('[','').replace(']','').replace(' ','').split(',')]
            else:
                raise ValueError
        except ValueError:
            self.fail('%s is not a valid input' % value, param, ctx)

@click.command()
@click.argument('model', type=click.Choice(['autapse','gauss', 'glm', 'hh', 'mog','sqrt']))
@click.argument('prefix', type=str)
@click.option('--accumulate-data', default=False, is_flag=True, show_default=True,
              help='If set, will accumulate the training data on each round by \
reloading data generated in previous round.')
@click.option('--bad-data', default='resample', type=str, show_default=True,
              help='Bad data handling strategy')
@click.option('--early-stopping', default=False, is_flag=True, show_default=True,
              help='If set, will do early stopping. Only works in combination with \
validation set, i.e., make sure that `--val` is greater zero.')
@click.option('--enqueue', default=None, type=str, show_default=True,
              help='Enqueue the job to a given queue instead of running it now. \
This requires a running worker process, which can be started with worker.py')
@click.option('--debug', default=False, is_flag=True, show_default=True,
              help='If provided, will enter debugger on error and show more \
info during runtime.')
@click.option('--device', default='cpu', type=str, show_default=True,
              help='Device to compute on.')
@click.option('--genetic', default=False, is_flag=True, show_default=True,
              help='If provided, will run genetic algorithm after fit.')
@click.option('--iw-loss', default=False, is_flag=True, show_default=True,
              help='If provided, will use importance weighted loss.')
@click.option('--learning-rate-scale', type=ListFloatParamType(), default=None,
              show_default=True,
              help='If provided, will scale learning rate accordingly.')
@click.option('--loss-calib', type=ListFloatParamType(), default=None,
              show_default=True,
              help='If provided, will do loss calibration with the kernel \
specified as loss-calib-kernel centered on x_o. The bandwidth of the kernel \
is determined by the floats provided. Provide a list to use different \
bandwidths on subsequent rounds, e.g. large on the first round and then \
shrinking.')
@click.option('--loss-calib-atleast', type=float, default=0.2, show_default=True,
              help='If set, kernel evaluation (per minibatch) will never return \
zero for all x at which it is evaluated: Iff the fraction of weights returned \
by the kernel are below the limit specified, the kernel will  will default to a \
uniform kernel for which the desired fraction is non-zero.')
@click.option('--loss-calib-kernel', type=str, default='uniform',
              show_default=True,
              help='Kernel type used for loss calibration. Note that the loss \
calibration kernel is only used, if the bandwidth specified as loss-calib is \
not None.')
@click.option('--missing-features', default='resample', type=str,
              show_default=True,
              help='Missing feature handling strategy')
@click.option('--nb', default=False, is_flag=True, show_default=True,
              help='If provided, will call nb.py after fitting.')
@click.option('--numerical-fix', default=False, is_flag=True, show_default=True,
              help='Numerical fix (for the orginal epsilonfree method).')
@click.option('--no-browser', default=False, is_flag=True, show_default=True,
              help='If provided, will not open plots of nb.py in browser.')
@click.option('--mcmc', default=False, is_flag=True, show_default=True,
              help='If provided, will run mcmc after fit.')
@click.option('--pdb-iter', type=int, default=None, show_default=True,
              help='Number of iterations after which to debug.')
@click.option('--prior-alpha', type=float, default=0.0, show_default=True,
              help='If iw_loss is True, will use this alpha as weight for true \
prior in proposal distribution.')
@click.option('--rep', type=ListIntParamType(), default=[2,1], show_default=True,
              help='Specify the number of repetitions per n_components model, \
seperation by comma. For instance, \'2,1\' would mean that 2 rounds with \
1 component are run, and 1 round with 2 components are run.')
@click.option('--rnn', type=int, default=None, show_default=True,
              help='If specified, will use many-to-one RNN with specified \
number of hidden units instead of summary statistics.')
@click.option('--samples', default='2000', type=ListIntParamType(), show_default=True,
              help='Number of samples, provided as either a single number or \
as a comma seperated list. If a list is provided, say \'1000,2000\', \
1000 samples are drawn for the first round, and 2000 samples for the second \
round. If more rounds than elements in the list are run, 2000 samples \
will be drawn for those (last list element).')
@click.option('--seed', type=int, default=None, show_default=True,
              help='If provided, network and simulation are seeded')
@click.option('--sim-kwargs', type=str, default=None, show_default=True,
              help='If provided, will be passed as keyword arguments \
to simulator. Seperate multiple keyword arguments by comma, for example:\
 \'duration=500,cython=True\'.')
@click.option('--svi', default=False, is_flag=True, show_default=True,
              help='If provided, will use SVI version')
@click.option('--train-kwargs', type=str, default=None, show_default=True,
              help='If provided, will be passed as keyword arguments \
to training function (inference.train). Seperate multiple keyword arguments \
by comma, for example: \'n_iter=500,n_minibatch=200\'.')
@click.option('--true-prior', default=False, is_flag=True, show_default=True,
              help='If provided, will use true prior on all rounds.')
@click.option('--units', type=ListIntParamType(), default=[50], show_default=True,
              help='List of integers such that each list element specifies \
the number of units per fully connected hidden layer. The length of the list \
equals the number of hidden layers.')
@click.option('--val', type=int, default=0, show_default=True,
              help='Number of samples for validation.')
def run(model, prefix, accumulate_data, bad_data, early_stopping, enqueue,
        debug, device, genetic, iw_loss, learning_rate_scale,
        loss_calib, loss_calib_atleast,
        loss_calib_kernel, mcmc, missing_features, nb, numerical_fix,
        no_browser, pdb_iter, prior_alpha, rep, rnn, samples, sim_kwargs,
        seed, svi, train_kwargs, true_prior, units, val):
    """Run model

    Call run.py together with a prefix and a model to run.

    See run.py --help for info on parameters.
    """
    # set env variables
    device = device.replace("gpu", "cuda")
    os.environ["THEANO_FLAGS"] = "device=" + device + ",floatX=float32"

    # import modules and functions depending on theano after setting env
    from likelihoodfree.Inference import Inference

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
            return dict((k, literal_eval(v)) for k, v in (pair.split('=') for pair in s.split(',')))

    inference_kwargs = {}
    inference_kwargs['bad_data_indicator'] = lambda x: int(np.any(np.isnan(x.reshape(-1)))) or x is None
    sim_kwargs = string_to_kwargs(sim_kwargs)
    train_kwargs = string_to_kwargs(train_kwargs)

    additional_info = {}

    # SVI: posterior over previous weights as prior
    if 'reg_init' in train_kwargs.keys() and train_kwargs['reg_init']:
        reg_init = True
    else:
        reg_init = False

    try:
        if model == 'autapse':
            from lfmods.autapse import AutapseSimulator as Simulator
            inference_kwargs['bad_data_indicator'] = lambda x: int(np.abs(x.reshape(-1)[-1]) > 1000)
            if rnn is not None:
                sim_kwargs['pilot_samples'] = 0
                sim_kwargs['summary_stats'] = 0
        elif model == 'gauss':
            from lfmods.gauss import GaussSimulator as Simulator
        elif model == 'glm':
            from lfmods.glm import GLMSimulator as Simulator
            if rnn is not None:
                sim_kwargs['pilot_samples'] = 0
                sim_kwargs['summary_stats'] = 0
        elif model == 'hh':
            from lfmods.hh import HHSimulator as Simulator
            if rnn is not None:
                sim_kwargs['pilot_samples'] = 0
                sim_kwargs['summary_stats'] = 0
        elif model == 'mog':
            from lfmods.mog import MoGSimulator as Simulator
        elif model == 'sqrt':
            from lfmods.sqrt import SqrtSimulator as Simulator
        else:
            raise ValueError('could not import Simulator')

        sim = Simulator(seed=seed,
                        **sim_kwargs)

        lfi = Inference(prefix=prefix,
                        seed=seed,
                        bad_data=bad_data,
                        sim=sim,
                        missing_features=missing_features,
                        **dirs,
                        **inference_kwargs)

        created = False
        iteration = 0
        n_components = 0
        n_samples = []

        for r in rep:
            n_components += 1
            for i in range(r):
                iteration += 1
                print('Iteration {}; {} Component(s)'.format(iteration, n_components))

                if not iw_loss and n_components > 1 and iteration > 1+rep[0] and not true_prior:
                    print('Run skipped!')
                    continue

                default_learning_rate = 0.001
                if learning_rate_scale is not None:
                    try:
                        lr_scale = learning_rate_scale[iteration-1]
                    except IndexError:
                        lr_scale = learning_rate_scale[-1]
                else:
                    lr_scale = 1
                train_kwargs['learning_rate'] = lr_scale*default_learning_rate

                if loss_calib is not None:
                    try:
                        bandwidth = loss_calib[iteration-1]
                    except IndexError:
                        bandwidth = loss_calib[-1]
                    kernel = Kernel(sim.obs, bandwidth=bandwidth,
                                    fun=loss_calib_kernel, spherical=True,
                                    atleast=loss_calib_atleast)
                else:
                    kernel = None

                if not created:
                    net, props = lfi.net_create(iw_loss=iw_loss,
                                                loss_calib=kernel,
                                                n_components=n_components,
                                                n_hiddens=units,
                                                numerical_fix=numerical_fix,
                                                svi=svi,
                                                rnn_hiddens=rnn)
                    created = True
                else:
                    path_posterior = '{}{}_iter_{:04d}_posterior.pkl'.format(dirs['dir_nets'],
                        prefix, iteration-1)
                    approx_posterior = io.load(path_posterior)

                    if not iw_loss:
                        prior_alpha = None

                    if true_prior:
                        prior_alpha = None
                        approx_posterior = None

                    net, props = lfi.net_reload(n_components=n_components,
                                                postfix='iter_{:04d}'.format(iteration-1),
                                                prior_alpha=prior_alpha,
                                                prior_proposal=approx_posterior,
                                                loss_calib=kernel)

                try:
                    n_samples = samples[iteration-1]
                except IndexError:
                    n_samples = samples[-1]

                if i == 0 and reg_init:
                    # TODO: cover 2 component case
                    train_kwargs['reg_init'] = False  # do not use init as prior
                elif reg_init:
                    train_kwargs['reg_init'] = True

                if accumulate_data and iteration > 3:
                    train_kwargs['load_trn'] = 'iter_{:04d}'.format(iteration-1)

                additional_info['early_stopping'] = False
                additional_info['early_stopping_iter'] = 0
                if early_stopping and val > 0 and iteration != 1:  # > 3
                    path_prev_loss = '{}{}_iter_{:04d}_loss.pkl'.format(dirs['dir_nets'],
                        prefix, iteration-1)
                    prev_loss = io.load(path_prev_loss)
                    pdict = prev_loss['val_min_params']
                    print('Early stopping : Setting parameters to iteration {} of previous round.'.format(prev_loss['val_min_iter']))
                    net.set_params(pdict)
                    additional_info['early_stopping'] = True
                    additional_info['early_stopping_iter'] = prev_loss['val_min_iter']

                # TODO: first fix, then make option from this
                restore_adam = False
                if restore_adam and iteration > 1:
                    # TODO: cover adding component case
                    path_adam = '{}{}_iter_{:04d}_adam.pkl'.format(dirs['dir_nets'], prefix, iteration-1)
                    adam_state = io.load(path_adam)
                    train_kwargs['load_adam'] = adam_state

                if debug:
                    print('Net')
                    for k, v in props.items():
                        print('{} : {}'.format(k, v))
                    print('Train kwargs')
                    for k, v in train_kwargs.items():
                        print('{} : {}'.format(k, v))

                lfi.train(debug=debug,
                          n_samples=n_samples,
                          n_samples_val=val,
                          net=net,
                          pdb_iter=pdb_iter,
                          postfix='iter_{:04d}'.format(iteration),
                          additional_info=additional_info,
                          **train_kwargs)

        if nb:
            print('Making plots')
            if debug:
                debug_flag = ['--debug']
            else:
                debug_flag = []
            if no_browser:
                browser_flag = ['--no-browser']
            else:
                browser_flag = []
            subprocess.call([sys.executable, 'nb.py', model, prefix] + debug_flag + browser_flag)

        if genetic:
            print('Run genetic algorithm')
            subprocess.call([sys.executable, 'run_genetic.py', model, prefix])

        if mcmc:
            print('Run MCMC')
            subprocess.call([sys.executable, 'run_mcmc.py', model, prefix])

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
    func_args = [sys.executable] + sys.argv

    try:
        enqueue_idx = func_args.index('--enqueue')

        from redis import Redis
        from rq import Queue

        timeout = int(1e6)
        ttl = -1
        connection = Redis()

        try:
            queue  = Queue(func_args[enqueue_idx+1], connection=connection)
        except:
            raise ValueError('--enqueue requires specification of queue')

        func_args.pop(enqueue_idx+1)
        func_args.pop(enqueue_idx)

        def subprocs_exec(args):
            subprocess.call(args, shell=False)


        queue.enqueue_call(func=subprocs_exec,
                           args=(func_args,),
                           timeout=timeout, ttl=ttl,
                           result_ttl=ttl)
        print('job enqueued : {}'.format(" ".join(func_args)))

    except ValueError:
        run()
