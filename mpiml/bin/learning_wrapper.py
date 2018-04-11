#! /usr/bin/env python

import argparse
import numpy as np
import sys

from mpi4py import MPI

from mpiml.datasets import prepare_dataset
from mpiml.models import get_model, model_names
from mpiml.training import *
from mpiml.utils import *
from mpiml.config import *
import mpiml.config as config

def wrapper(
    model, k, data_path, online=False, density=1.0, pool_size=pool_size, parallel_test=False,
    cycles_per_barrier=10):
    """ input: type of ML model, number of k-fold splits, path to dataset, online, percentage of dataset,
               pool size, whether to test in parallel, number of cycles per barrier
        output: trains ML_type on training data and tests it on testing data
    """
    ds = prepare_dataset(data_path, density=density, pool_size=pool_size)

    root_info('{}', output_model_info(model, online=online, density=density, pool_size=pool_size))

    result = train_and_test_k_fold(
        ds, model, k=k, online=online, parallel_test=parallel_test, cycles_per_barrier=cycles_per_barrier)

    root_info('PERFORMANCE\n{}', result)

if __name__ == '__main__':

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the bubbleShock dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(model_names()))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('--density', type=float, help='fraction of dataset to train on (default 1)', default=1.0)
    parser.add_argument('--pool-size', type=int, help='specify pooling values for online to be trained upon', default=None)
    parser.add_argument('--parallel-test', action='store_true')
    parser.add_argument('--cycles-per-barrier', type=int, default=10,
        help='number of cycles each task should load into memory and train before synching with other tasks')

    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    for model in args.models:
        m = get_model(model, bootstrap=True, oob_score=True, min_samples_split=config.min_samples_split)
        if m is None:
            root_info('error: invalid model {}; valid models are {}', model, model_names())
            sys.exit(1)
        else:
            wrapper(m, args.num_runs, args.data_dir,
                online=args.online, density=args.density,
                pool_size=args.pool_size, parallel_test=args.parallel_test,
                cycles_per_barrier=args.cycles_per_barrier
            )
