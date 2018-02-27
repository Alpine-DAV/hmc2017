#! /usr/bin/env python2

import argparse
import cPickle
import numpy as np
import skgarden.mondrian.tree.tree as mtree
from sklearn.model_selection import ShuffleSplit
import sys
import time
import zlib

from mpiml.config import comm
from mpiml.datasets import prepare_dataset
from mpiml.output import CSVOutput
from mpiml.utils import debug, profile, toggle_profiling, toggle_verbose

class native_tree_comm(object):

    def __init__(self, n_features, n_outputs, compression=1):
        if compression < 0 or compression > 9:
            raise ValueError('compression must be in [0, 9]')
        self.n_features_ = n_features
        self.n_outputs_ = n_outputs
        self.compression_ = compression

    def send(self, comm, dst, tree):
        return mtree.mpi_send(
            comm, dst, tree, compression=self.compression_, profile=True)

    def recv(self, comm, src):
        return mtree.mpi_recv_regressor(
            comm, src, self.n_features_, self.n_outputs_, profile=True)

    def __repr__(self):
        return 'native_tree_comm(n_features={},n_outputs={},compression={})'.format(
            self.n_features_, self.n_outputs_, self.compression_)

class pickle_tree_comm(object):

    def __init__(self, compression=0):
        if compression < 0 or compression > 9:
            raise ValueError('compression must be in [0, 9]')
        self.compression_ = compression

    def send(self, comm, dst, tree):
        t_start = time.time()
        pkl = cPickle.dumps(tree, cPickle.HIGHEST_PROTOCOL)
        if self.compression_ > 0:
            pkl = zlib.compress(pkl, self.compression_)
        buf = np.frombuffer(pkl, dtype=np.dtype('b'))
        nbytes = buf.shape[0]

        t_send = time.time()
        comm.send(nbytes, dst)
        comm.Send(buf, dst)

        return t_start, t_send, nbytes

    def recv(self, comm, src):
        nbytes = comm.recv(source=src)
        buf = np.empty(nbytes, dtype=np.dtype('b'))

        comm.Recv(buf, source=src)
        t_recv = time.time()

        pkl = buf.tobytes()
        if self.compression_ > 0:
            pkl = zlib.decompress(pkl)
        tree = cPickle.loads(pkl)
        t_end = time.time()

        return tree, t_recv, t_end

    def __repr__(self):
        return 'pickle_tree_comm(compression={})'.format(self.compression_)

def send_tree(comm, tree_comm, dst, tree):
    t_start, t_sent, bytes_sent = tree_comm.send(comm, dst, tree)
    comm.send((t_start, t_sent, bytes_sent), dst)

def recv_tree(comm, tree_com, src):
    tree, t_recv, t_end = tree_comm.recv(comm, src)
    t_start, t_sent, bytes_sent = comm.recv(source=src)

    # (tree, t_total, t_preprocess, t_transmit, t_postprocess, bytes)
    return tree, t_end - t_start, t_sent - t_start, t_recv - t_sent, t_end - t_recv, bytes_sent

@profile('mt-pickle-producer')
def producer(tree_comm, trees, send_to=1):
    for tree in trees:
        comm.barrier()
        send_tree(comm, tree_comm, send_to, tree)


@profile('mt-pickle-consumer')
def consumer(tree_comm, ntrees, recv_from=0):
    t_total_accum = 0
    t_prep_accum = 0
    t_transmit_accum = 0
    t_postp_accum = 0
    bytes_accum = 0

    trees = []
    for _ in range(ntrees):
        comm.barrier()

        tree, t_total, t_preprocess, t_transmit, t_postprocess, bytes_sent = \
            recv_tree(comm, tree_comm, recv_from)

        t_total_accum += t_total
        t_prep_accum += t_preprocess
        t_transmit_accum += t_transmit
        t_postp_accum += t_postprocess
        bytes_accum += bytes_sent

        trees.append(tree)

    return trees, t_total_accum, t_prep_accum, t_transmit_accum, t_postp_accum, bytes_sent

if __name__ == '__main__':
    if comm.size != 2:
        raise ValueError('{} requires exactly 2 MPI tasks to run'.format(sys.argv[0]))

    parser = argparse.ArgumentParser(
        description='Benchmark and optionally profile pickling and communicating of Mondrian trees')
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--num-trees', type=int, default=50)
    parser.add_argument('--output', type=str, default=None, help='output path for CSV (default stdout)')
    parser.add_argument('--append', action='store_true', help='append to output')
    parser.add_argument('--schema', action='store_true',
        help='include the schema as the first line of output')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    consumer_rank = 0
    producer_rank = 1

    ds = prepare_dataset(args.dataset, density=args.density)
    n_samples, n_features = ds.points()[0].shape
    n_outputs = 1

    X, y = ds.points()
    splitter = ShuffleSplit(n_splits=args.num_trees)
    splits = [(X[train_index], y[train_index], X[test_index], y[test_index]) \
        for train_index, test_index in splitter.split(X)]

    tree_comms = [native_tree_comm(n_features, n_outputs, c) for c in range(10)] + \
                 [pickle_tree_comm(compression=c) for c in range(10)]

    schema = ['pickle', 'compression', 'n_trees', 'n_features', 'n_outputs', 'n_samples', 't_total',
              't_preprocess', 't_transmit', 't_postprocess', 'bytes']
    if comm.rank == consumer_rank:
        writer = CSVOutput(schema, output=args.output, write_schema=args.schema, append=args.append)

    for compression in range(10):

        strategies = [ ('t', pickle_tree_comm(compression=compression))
                     , ('f', native_tree_comm(n_features, n_outputs, compression=compression))
                     ]

        for pickle_flag, tree_comm in strategies:
            if comm.rank == consumer_rank:
                trees, t_total, t_prep, t_transmit, t_postp, nbytes = \
                    consumer(tree_comm, args.num_trees, recv_from=producer_rank)

                if args.verbose:
                    for (t, (_, _, test_X, test_y)) in zip(trees, splits):
                        comm.barrier()
                        debug('consumer score: {}', t.score(test_X, test_y))
                        comm.barrier()

                writer.writerow(
                    pickle=pickle_flag, compression=compression, n_trees=args.num_trees,
                    n_features=n_features, n_samples=n_samples, n_outputs=n_outputs,
                    t_total=t_total, t_preprocess=t_prep, t_transmit=t_transmit,
                    t_postprocess=t_postp, bytes=nbytes
                )

            else:
                trees = []
                for train_X, train_y, _, _ in splits:
                    t = mtree.MondrianTreeRegressor()
                    t.fit(train_X, train_y)
                    trees.append(t)

                producer(tree_comm, trees, send_to=consumer_rank)

                if args.verbose:
                    for (t, (_, _, test_X, test_y)) in zip(trees, splits):
                        debug('producer score: {}', t.score(test_X, test_y))
                        comm.barrier()
                        comm.barrier()
