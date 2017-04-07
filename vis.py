from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import likelihoodfree.io as io
import likelihoodfree.viz as viz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import seaborn.apionly as sns
import scipy.stats as stats
import socket
import sys
import time

from visdom import Visdom

@click.command()
@click.argument('model', type=click.Choice(['mog', 'hh']))
@click.option('--debug/--no-debug', default=False, is_flag=True, help='If True, will enter debugger on error')
@click.option('--env', default='main', help='Environment to plot to (main by default)')

def run(model, env, debug):
    """Plotting

    Call `vis.py` together with a string indicating 
    the model.
    
    See `vis.py --help` for info on parameters.
    """
    block = False
    
    dirs = {}
    dirs['dir_nets'] = 'results/'+model+'/nets/'
    dirs['dir_plots'] = 'results/'+model+'/plots/'

    env_ = env

    # loop over pkl files associated with run
    basenames = []
    filepaths = []
    for filename in os.listdir(dirs['dir_nets']):
        if filename.endswith(".pkl") and "info" in filename:
            basenames.append(os.path.splitext(filename)[0])
            filepaths.append(os.path.join(dirs['dir_nets'], filename))

    vlt = Visdom()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 8097))
        print('Make sure to start a visdom server using:')
        print('python -m visdom.server')
    except socket.error as e:
        vurl = 'http://localhost:8097/'
        print('Visdom at {}'.format(vurl))
        print('')
        pass

    try:
        print('Clearing plots')
        vlt.close(win=None, env=env)
                
        for basename, filepath in zip(basenames, filepaths):
            info = io.load(filepath)
            print('Processing {} ...'.format(filepath))

            fig, ax = viz.loss(info)
            pfig = viz.mpl2plotly(fig)
            viz.send2vis(pfig, vlt)
             
            infotext = viz.info(info, html=True)
            options = {}
            options['title'] = '{} : info'.format(basename)
            vlt.text(infotext, opts=options, env=env_)
            
            """
            # plt.gcf() to jpeg
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            options = {}
            options['title'] = ''
            options['jpgquality'] = 100
            vlt.image(data, opts=options, env=env_)
            """
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
