#!/usr/bin/env python
"""Worker for jobs
"""
import click
import pdb
import os
import re
import subprocess

from redis import Redis
from rq import Worker, Queue, Connection

def subprocs_exec(args):
    subprocess.call(args, shell=False)

@click.command()
@click.option('--queue', type=str, default='default', show_default=True,
              help='Queue to listen on')
def run(queue):
    listen = ['all', queue]

    cpu_regex = re.compile(r"cpu(\d)+")
    gpu_regex = re.compile(r"gpu(\d)+")

    if cpu_regex.search(queue) is not None:
        listen += ['cpu']

    if gpu_regex.search(queue) is not None:
        listen += ['gpu']
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(queue[3:])

    connection = Redis()
    with Connection(connection):
        worker = Worker(map(Queue, listen))
        worker.work()

if __name__ == '__main__':
    run()
