#!/usr/bin/env python

import argparse
import csv
import numpy as np
import os
import sys

import mpiml.config as config
from mpiml.datasets import *
from mpiml.models import get_model, model_names, get_cli_name
from mpiml.training import *
from mpiml.utils import *

def _slurm_env(key):
    if key in os.environ:
        return os.environ[key]
    else:
        root_info('not running in slurm')
        sys.exit(1)

def selectcols(schema, **kwargs):
    return [kwargs[col] for col in schema]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test performance of a classifier on a range of problem sizes')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(model_names()))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--num-points', type=int, default=9,
        help='number of evenly spaced density values to test in the range [0.2, 1]')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('--output', type=str, default=None, help='output path for CSV')
    parser.add_argument('--append', action='store_true', help='append to output')
    parser.add_argument('--schema', action='store_true',
        help='include the schema as the first line of output')
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    nodes = _slurm_env('SLURM_NNODES')
    tasks = _slurm_env('SLURM_NTASKS')

    models = []
    for m in args.models:
        model = get_model(m)
        if model is None:
            root_info('error: invalid model {}; valid models are {}', model, model_names())
            sys.exit(1)
        models.append(model)

    if config.comm.rank == 0:
        if args.output is None:
            f = sys.stdout
        else:
            f = open(args.output, 'ab' if args.append  else 'wb')
        writer = csv.writer(f)

    schema = ['model', 'nodes', 'tasks', 'density', 'positive_train_samples', 'negative_train_samples',
              'positive_test_samples', 'negative_test_samples', 'time_train', 'time_reduce', 'time_test',
              'fp', 'fn', 'accuracy', 'RMSE']
    if args.schema and config.comm.rank == 0:
        writer.writerow(schema)

    for model in models:
        for density in np.linspace(0.2, 1, num=args.num_points):
            root_debug('{}',output_model_info(model, online=args.online, density=density))

            ds = prepare_dataset(args.data_dir, density=density)

            result = train_and_test_k_fold(ds, model, k=args.num_runs, online=args.online)
            root_debug('PERFORMANCE\n{}', result)

            if config.comm.rank == 0:
                writer.writerow(selectcols(schema,
                    model=get_cli_name(model), density=density, nodes=nodes, tasks=tasks,
                    positive_train_samples=result.positive_train_samples,
                    negative_train_samples=result.negative_train_samples,
                    positive_test_samples=result.positive_test_samples,
                    negative_test_samples=result.negative_test_samples,
                    time_train=result.time_train,
                    time_reduce=result.time_reduce,
                    time_test=result.time_test,
                    fp=result.fp,
                    fn=result.fn,
                    accuracy=result.accuracy,
                    rmse=result.rmse))

    if args.output and config.comm.rank == 0:
        f.close()
