#!/bin/python
import fire
import multiprocessing
import os
import pdb
import pickle
import random
import string
import time

from tqdm import tqdm


PATH_POSTERIOR = '../results/mog/{}_seed_{}_round_{}_posterior.pkl'
PATH_INFERENCE = '../results/mog/{}_seed_{}_round_{}_inference.pkl'
PATH_KL = '../results/mog/{}_kl.npy'


def run_single(algo='CDELFI', rounds=2, seed=None, verbose=True):
    """Fits single MoG model

    For more info, see usage returned by:
    python run_mog.py -- --help
    """
    import delfi.distribution as dd
    import delfi.utils.io as io
    import numpy as np

    from delfi.generator import Default
    from delfi.inference import Basic, CDELFI, SNPE
    from delfi.simulator import GaussMixture
    from delfi.summarystats import Identity
    from delfi.simulator.GaussMixture import GaussMixture

    n_params = 1
    m = GaussMixture(dim=n_params, seed=seed)
    p = dd.Uniform(lower=[-10], upper=[10], seed=seed)
    s = Identity()
    g = Default(model=m, prior=p, summary=s)

    obs = np.array([[0.]])
    kwargs = {'generator': g,
              'n_components': 2,
              'n_hiddens': [20],
              'obs': obs,
              'seed': seed,
              'verbose': verbose,
              'prior_norm': False,
              'pilot_samples': 0,
              'reg_lambda': 0.01}

    if algo == 'Basic':
        del kwargs['obs']
        inf = Basic(**kwargs)
    if algo == 'CDELFI':
        inf = CDELFI(**kwargs)
    elif algo == 'SNPE':
        inf = SNPE(**kwargs)
    else:
        raise ValueError

    train = []
    for r in range(rounds):
        train.append(1000)
    train[-1] = 2000

    try:
        if algo == 'Basic':
            logs, train_datasets, posteriors = inf.run(n_train=2000)
        else:
            logs, train_datasets, posteriors = inf.run(n_train=train, n_rounds=rounds)
        posterior = posteriors[-1]
    except:
        posterior = None
        pass

    if verbose:
        for p in posteriors:
            print('var c1 : {}'.format(p.xs[0].S))

    io.save(inf, PATH_INFERENCE.format(algo, seed, rounds))
    io.save_pkl(posterior, PATH_POSTERIOR.format(algo, seed, rounds))

def run_many_single_seed(seed=None, verbose=False):
    # run multiple algorithms for varying number of rounds for one seed
    loop_1 = ['CDELFI', 'SNPE']
    loop_1 = tqdm(loop_1, desc='algo  ') if verbose else loop_1
    loop_2 = range(2, 6+1)
    loop_2 = tqdm(loop_2, desc='round ') if verbose else loop_2

    for algo in loop_1:
        for rounds in loop_2:
            run_single(algo, rounds, seed, verbose=False)

def run_many_range_of_seeds(start=1, end=10):
    # run multiple algorithms for varying number of rounds for multiple seeds

    def init(env_update=os.environ.copy()):
        theano_flags = {key: value for (key, value)
                      in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}
        theano_flags.setdefault("compiledir_format",
                              "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s")
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        theano_flags["compiledir_format"] += '-' + random_str
        env_update["THEANO_FLAGS"] = ",".join(["%s=%s" % (key, value) for (key, value) in theano_flags.items()])
        os.environ = env_update

    #cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(initializer=init,
                                initargs=(os.environ.copy(),))

    tasks = range(start, end+1)  # seeds
    work = run_many_single_seed
    #work = test_work

    for res in tqdm(pool.imap_unordered(work, tasks),
                    total=len(tasks),
                    desc='seed '):
        pass

    pool.close()
    pool.join()

def test_work(x):
    time.sleep(x)

def compute_kl():
    import delfi.distribution as dd
    import delfi.utils.io as io
    import numpy as np

    gt_posterior = dd.MoG(a=[0.5, 0.5],
                          ms=[[0.],[0.]],
                          Ss=[[[1.]],[[0.1]]])
    samples = gt_posterior.gen(100000)

    seeds = 20
    rounds = 6

    res = {}
    for a in ['CDELFI', 'SNPE']:
        res[a] = -1*np.ones((rounds-1, seeds))

        for s in range(10, seeds+1):
            for r in range(2, rounds+1):
                fname = PATH_POSTERIOR.format(a, s, r)
                try:
                    posterior = io.load_pkl(fname)
                    kl_tp = np.mean( gt_posterior.eval(samples) - posterior.eval(samples) )
                    res[a][r-2, s-1] = kl_tp
                    print('success')
                except:
                    print('error on {}'.format(fname))
                    pass

            np.save(PATH_KL.format(a), res[a])

if __name__ == '__main__':
    fire.Fire()
