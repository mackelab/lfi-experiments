# http://compbio.ucsd.edu/reproducible-analysis-automated-jupyter-notebook-pipelines/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import nbformat
import nbparameterise
import os
import pdb
import six
import sys
import webbrowser

from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

def read_in_notebook(notebook_fp):
    with open(notebook_fp) as f:
        nb = nbformat.read(f, as_version=4)
    return nb

def set_parameters(nb, params_dict):
    orig_parameters = nbparameterise.extract_parameters(nb)
    params = nbparameterise.parameter_values(orig_parameters, **params_dict)
    new_nb = nbparameterise.replace_definitions(nb, params, execute=False)
    return new_nb

# modified from https://nbconvert.readthedocs.io/en/latest/execute_api.html
def execute_notebook(notebook_filename, notebook_filename_out, params_dict,
    run_path="", timeout=6000000):

    notebook_fp = os.path.join(run_path, notebook_filename)
    nb = read_in_notebook(notebook_fp)
    new_nb = set_parameters(nb, params_dict)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    try:
        ep.preprocess(new_nb, {'metadata': {'path': run_path}})
    except:
        msg = 'Error while executing: "{0}".\n\n'.format(notebook_filename)
        msg = '{0}See notebook "{1}" for traceback.'.format(
                msg, notebook_filename_out)
        print(msg)
        raise
    finally:
        with open(notebook_filename_out, mode='wt') as f:
            nbformat.write(new_nb, f)
        export_notebook_to_html(new_nb, notebook_filename_out)


def export_notebook_to_html(nb, notebook_filename_out):
    html_exporter = HTMLExporter()
    body, resources = html_exporter.from_notebook_node(nb)
    out_fp = notebook_filename_out.replace(".ipynb", ".html")
    with open(out_fp, "w", encoding="utf8") as f:
        f.write(body)

@click.command()
@click.argument('model', type=click.Choice(['autapse', 'gauss', 'hh', 'mog']))
@click.argument('prefix', type=str)
@click.option('--browser/--no-browser', default=True, is_flag=True,
              help='If True, will open results HTML or notebook in browser')
@click.option('--debug/--no-debug', default=False, is_flag=True,
              help='If True, will enter debugger on error')
@click.option('--jupyter/--no-jupyter', default=True, is_flag=True,
              help='If True, will try to open Jupyter NB instead of HTML')
@click.option('--jupyter-port', type=int, default=8888,
              help='Jupyter port')
@click.option('--nb', type=str, default='viz',
              help='Will use notebooks/model_$nb.ipynb, where $nb defaults to viz')
@click.option('--postfix', type=str, default=None,
              help='Postfix')
def run(model, prefix, browser, debug, jupyter, jupyter_port, nb, postfix):
    """Generate notebook and HTML output

    Call `nb.py` together with a prefix and a model to run.

    See `nb.py --help` for info on parameters.
    """
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_nb'] = 'results/'+model+'/notebooks/'

    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

    try:
        path_ipynb = dirs['dir_nb'] + prefix + '.ipynb'
        path_html = dirs['dir_nb'] + prefix + '.html'

        if not jupyter:
            url = 'file://' + os.path.realpath(path_html)
        else:
            url_tpl = 'http://localhost:{}/notebooks/results/{}/notebooks/{}.ipynb'
            url = url_tpl.format(jupyter_port, model, prefix)

        execute_notebook('../../../notebooks/' + model + '_' + nb + '.ipynb',
                         path_ipynb,
                         {'prefix': prefix,
                          'postfix': postfix,
                          'basepath': '../'},
                         run_path=dirs['dir_nb'])

        if browser:
            webbrowser.open(url, autoraise=True, new=True)
    except:
        t, v, tb = sys.exc_info()
        if debug:
            webbrowser.open(url, autoraise=autoraise, new=True)
            print('')
            print('Exception')
            print(v.with_traceback(tb))
            pdb.post_mortem(tb)
        else:
            raise v.with_traceback(tb)

if __name__ == '__main__':
    run()
