#!/usr/bin/env python
"""Worker for jobs
"""
import argparse
import os
import subprocess

from redis import Redis
from rq import Worker, Queue, Connection

def subprocs_exec(args):
    subprocess.call(args, shell=False)

connection = Redis()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', help='Queue to listen on')

    args = parser.parse_args()

    listen = ['default']

    if args.queue is not None:
        listen = [args.queue]

    with Connection(connection):
        worker = Worker(map(Queue, listen))
        worker.work()
