#!/usr/bin/env python
"""Worker for jobs
"""
import click
import os
import subprocess

from redis import Redis
from rq import Worker, Queue, Connection

def subprocs_exec(args):
    subprocess.call(args, shell=False)


@click.command()
@click.option('--queue', type=str, default='default', show_default=True,
              help='Queue to listen on')
def run(queue):
    connection = Redis()

    listen = ['all', queue]

    with Connection(connection):
        worker = Worker(map(Queue, listen))
        worker.work()

if __name__ == '__main__':
    run()
