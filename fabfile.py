import ast
import scipy.stats.distributions as dist
import numpy as np
import pdb
import uuid
import yaml

from fabric.api import local, run, task, warn_only
from sklearn.model_selection import ParameterGrid, ParameterSampler

CPUS_DEFAULT = 20
GPUS_DEFAULT = 2
TMUX_SESSION = 'lf'

@task(alias='a')
def attach(name=None):
    if name is None:
        tmux_session_attach()
    else:
        tmux_window_attach(name)

@task(alias='b')
def browser(name):
    if name in ['j', 'jupyter', 'nb']:
        URL = 'http://localhost:8888/'
    if name == 'rq':
        URL = 'http://localhost:9181/'
    local('open {url}'.format(url=URL), shell='/bin/fish')

@task(alias='c')
def clean():
    local('rq empty default')
    local('rq empty failed')
    local('rq empty cpu')
    local('rq empty gpu')

@task(alias='d')
def dryrun():
    global local, run

    def local(command, capture=False, shell=None):
        print("(dryrun)[localhost] %s" % (command))

    def run(command, shell=True, pty=True, combine_stderr=None, quiet=False,
            warn_only=False, stdout=None, stderr=None, timeout=None,
            shell_escape=None, capture_buffer_size=None):
        print("(dryrun)[%s] %s" % (env.host_string, command))

@task(alias='i')
def info():
    local('nvidia-smi')
    local('rq info')

@task(alias='k')
def kill(sess=TMUX_SESSION):
    clean()
    local('rq suspend')
    local('redis-cli flushdb')
    tmux_session_kill(sess)

@task(alias='l')
def list_windows(name='dashboard'):
    tmux_window_list(echo=True)

@task(alias='r')
def run(name, limit=None):
    limit = parse_limit(limit)

    y = yaml_parse('experiments/' + name +'.yaml')
    assert len(y) == 2, 'yaml needs header and body'

    header = y[0]
    comment = header['comment'] if 'comment' in header else None
    if limit is None and 'limit' in header:
        limit = parse_limit(header['limit'])
    prefix = header['prefix'] + '_' if 'prefix' in header else ''

    runs = y[1]
    for k, v in runs.items():
        if type(v) == str and 'np.' in v:
            v = eval(v)
            runs[k] = list(v)
        elif type(v) == str and 'dist.' in v:
            v = eval(v)
            runs[k] = v
            if limit is None:
                raise ValueError('dist is used, limit required')
        elif type(v) is not list:
            runs[k] = [v]

    grid = True if limit is None else False
    if grid:
        params = ParameterGrid(runs)
    else:
        params = ParameterSampler(runs, n_iter=limit)

    pl = list(params)
    for li in pl:
        assert 'model' in li, 'model is required'
        model = li.pop('model')

        uid = str(uuid.uuid4())[:8]
        runname = prefix + uid

        cmd = "python run.py {m} {r}".format(m=model, r=runname)

        sim_kwargs = ''
        train_kwargs = ''

        for k, v in li.items():
            if 1 == 2:
                pass
            elif 'bimodal' in k and model in ['mog']:
                sim_kwargs += ',cython=' + str(v)
            elif 'cython' in k and model in ['hh']:
                sim_kwargs += ',cython=' + str(v)
            elif 'dim' in k and model in ['glm', 'hh']:
                sim_kwargs += ',dim=' + str(v)
            elif 'n-summary' in k and model in ['gauss']:
                sim_kwargs += ',n_summary=' + str(v)
            elif 'duration' in k and model in ['glm', 'hh']:
                sim_kwargs += ',duration=' + str(v)
            elif 'step-current' in k and model in ['hh']:
                sim_kwargs += ',step_current=' + str(v)
            elif 'loss-calib' in k:
                if float(v) > 0.:
                    cmd += ' --loss-calib ' + str(v)
            elif 'iw-loss' in k:
                if v:
                    cmd += ' --iw-loss'
            elif 'svi' in k:
                if v:
                    cmd += ' --svi'
            elif 'true-prior' in k:
                if v:
                    cmd += ' --true-prior'
            else:
                cmd += ' --{k} {v}'.format(k=k, v=v)

        if model in ['glm', 'hh']:
            sim_kwargs += ',cached_pilot=False,cached_sims=False'

        if sim_kwargs != '':
            cmd += ' --sim-kwargs ' + sim_kwargs[1:]

        device = 'cpu'
        if 'rnn' in li and int(li[rnn]) != 0:
            device = 'gpu'
            cmd += ' --device cuda0'

        cmd += ' --enqueue {d}'.format(d=device)

        cmd += ' --nb --no-browser'

        local(cmd)

    print('\nNumber of runs: {}'.format(len(params)))

@task(alias='s')
def start(cpus=CPUS_DEFAULT, gpus=GPUS_DEFAULT):
    with warn_only():
        kill()
    tmux_session_create()
    tmux_start_workers('cpu', int(cpus))
    tmux_start_workers('gpu', int(gpus))
    tmux_start_cmd('rq-dashboard')
    tmux_start_cmd('jupyter notebook --no-browser')

###

def parse_limit(limit):
    if limit is None:
        limit = None
    elif type(limit) is str and limit == 'None':
        limit = None
    elif type(limit) is str and limit == '0':
        limit = None
    elif type(limit) is str:
        limit = int(limit)
    elif type(limit) is int and limit == 0:
        limit = None
    elif type(limit) is int:
        limit = limit
    else:
        raise ValueError('could not parse limit')
    return limit

def tmux_session_attach(sess=None):
    if sess is None:
        sess = TMUX_SESSION
    cmd = 'tmux attach -t "{sess}"'
    cmd = cmd.format(sess=sess)
    local(cmd)

def tmux_session_create():
    cmd = 'tmux new-session -d -s {sess}'
    cmd = cmd.format(sess=TMUX_SESSION)
    local(cmd)

def tmux_session_kill(sess=TMUX_SESSION):
    cmd = 'tmux kill-session -t {sess}'
    cmd = cmd.format(sess=sess)
    local(cmd)

def tmux_start_cmd(command):
    command_split = command.split()
    cmd = 'tmux new-window -d -a -t "{sess}" -n "{name}"'
    cmd += ' "{cmd}"'
    cmd = cmd.format(sess=TMUX_SESSION, name=command_split[0], cmd=command)
    local(cmd)

def tmux_start_worker(dev='cpu', num=0):
    cmd = 'tmux new-window -d -a -t "{sess}" -n "{dev}{num}"'
    cmd += ' "python worker.py --queue {dev}{num}"'
    cmd = cmd.format(sess=TMUX_SESSION, dev=dev, num=num)
    local(cmd)

def tmux_start_workers(dev='cpu', count=1):
    for i in range(int(count)):
        tmux_start_worker(dev, i)

def tmux_window_list(echo=True):
    cmd = 'tmux list-windows -a -F'
    cmd += ' "{^sess^: ^#{session_name}^,'
    cmd += ' ^name^: ^#{window_name}^,'
    cmd += ' ^id^: ^#{window_id}^}"'
    out = local(cmd, capture=True)

    windows = []
    for line in out.splitlines():
        line = line.replace('^', '"')
        line = ast.literal_eval(line)
        windows += [line]
        if echo:
            print('{sess}: {name} {id}'.format(**line))

    return windows

def tmux_window_attach(name):
    windows = tmux_window_list(echo=False)

    res = [window for window in windows if window['name'] == name]

    if len(res) == 1:
        sess = res[0]['sess']
        cmd = 'tmux select-window -t {sess}:{id}'
        cmd = cmd.format(sess=sess, id=res[0]['id'])
        local(cmd)
        tmux_session_attach(sess=sess)
    else:
        print('\nNumber of matches for {} != 1 but {}'.format(name, len(res)))

def yaml_parse(filename):
    with open(filename, 'r') as stream:
        try:
            return list(yaml.load_all(stream))
        except yaml.YAMLError as exc:
            return exc
