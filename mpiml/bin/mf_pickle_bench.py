#!/usr/bin/env python2

# Benchmark testing performance of Mondrian superforest merging use the native encoding versus the
# Pickle encoding.
#
# The experiment consists simply of training and merging a number of Mondrian forests while
# measuring the merging time. The experiment should be run multiple times with varying
# numbers of MPI tasks. This script only runs the experiment once. Use mf_pickle_bench.sh
# to run this script multiple times with variously sized communicators.
#
#The output is a CSV with the following schema:
# pickle: t if the row contains data for a Pickle encoding, f for native encoding
# n_tasks: number of MPI tasks used in the trial
# compression: level of compression used during encoding (integer in [0, 9])
# n_forests: number of forests in the trial
# t_reduce: total time in seconds spent reducing

import argparse

from mpiml.config import comm
from mpiml.datasets import prepare_dataset
from mpiml.forest import *
from mpiml.output import CSVOutput
from mpiml.training import train_and_test_k_fold
from mpiml.utils import toggle_verbose, toggle_profiling

if __name__ == '__main__':
    if comm.size < 2:
        raise ValueError('{} requires at least 2 MPI tasks to run'.format(sys.argv[0]))

    parser = argparse.ArgumentParser(
        description='Benchmark reduction of Mondrian forests')
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--num-forests', type=int, default=10)
    parser.add_argument('--output', type=str, default=None, help='output path for CSV (default stdout)')
    parser.add_argument('--append', action='store_true', help='append to output')
    parser.add_argument('--schema', action='store_true',
        help='include the schema as the first line of output')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    ds = prepare_dataset(args.dataset, density=args.density)

    forest_types = [
        ('f', MondrianForestRegressor),
        ('t', MondrianForestPickleRegressor)
    ]

    schema = ['pickle', 'n_tasks', 'compression', 'n_forests', 't_reduce']
    if comm.rank == 0:
        writer = CSVOutput(schema, output=args.output, write_schema=args.schema, append=args.append)

    for pickle_flag, forest in forest_types:
        for compression in range(10):
            r = train_and_test_k_fold(ds, forest(compression=compression), k=args.num_forests)

            if comm.rank == 0:
                writer.writerow(
                    pickle=pickle_flag, compression=compression, n_forests=args.num_forests,
                    n_tasks=comm.size, t_reduce=r.time_reduce
                )
