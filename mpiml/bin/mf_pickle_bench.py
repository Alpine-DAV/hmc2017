#!/usr/bin/env python2

import argparse
import cPickle
import numpy as np
from skgarden.mondrian.tree.tree import mpi_send, mpi_recv_regressor
from sklearn.model_selection import ShuffleSplit
import sys
import time
import zlib

from mpiml.config import comm
from mpiml.datasets import prepare_dataset
from mpiml.forest import MondrianForestRegressor
from mpiml.training import fit
from mpiml.utils import profile, toggle_verbose, toggle_profiling, info, debug

@profile('mf-pickle-producer')
def producer(forests, send_to=1):
    start = time.time()

    for forest in forests:
        comm.send(len(forest.estimators_), send_to)
        for tree in forest.estimators_:
            mpi_send(tree, comm, send_to)

    end = time.time()

    info('producer: {} s', end - start)
    comm.barrier()


@profile('mf-pickle-consumer')
def consumer(n_features, n_outputs, iterations, recv_from=0):
    start = time.time()

    forests = []
    for _ in range(iterations):
        n_trees = comm.recv(source=recv_from)
        prd = MondrianForestRegressor()
        prd.estimators_ = []
        for i in range(n_trees):
            prd.estimators_.append(mpi_recv_regressor(n_features, n_outputs, comm, recv_from))
        forests.append(prd)

    end = time.time()

    comm.barrier()
    info('consumer: {} s', end - start)

    return forests

if __name__ == '__main__':
    if comm.size != 2:
        print('{} requires 2 MPI tasks to run'.format(sys.argv[0]))
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Benchmark and optionally profile pickling and communicating of Mondrian forests')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='number of forests to send over MPI')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    producer_rank = 0
    consumer_rank = 1

    X, y = prepare_dataset('boston').points() # sklearn toy regression problem
    splitter = ShuffleSplit(n_splits=args.num_runs)

    splits = [(X[train_index], y[train_index], X[test_index], y[test_index]) \
        for train_index, test_index in splitter.split(X)]

    if comm.rank == producer_rank:
        forests = []
        for train_X, train_y, _, _ in splits:
            f = MondrianForestRegressor(min_samples_split=5)
            f.fit(train_X, train_y)
            forests.append(f)

        producer(forests, send_to=consumer_rank)
        comm.barrier()

        for (f, (_, _, test_X, test_y)) in zip(forests, splits):
            debug('producer score: {}', f.score(test_X, test_y))
            comm.barrier()
            comm.barrier()

    else:
        forests = consumer(X.shape[1], 1, args.num_runs, recv_from=producer_rank)
        comm.barrier()

        for (f, (_, _, test_X, test_y)) in zip(forests, splits):
            comm.barrier()
            debug('consumer score: {}', f.score(test_X, test_y))
            comm.barrier()
