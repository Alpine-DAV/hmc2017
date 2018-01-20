#!/usr/bin/env python

import argparse
import csv
import numpy as np
import sys

import mpiml.config as config
from mpiml.datasets import prepare_dataset
from mpiml.models import get_model, model_names
from mpiml.utils import *

def selectcols(schema, **kwargs):
    return [kwargs[col] for col in schema]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test performance of a classifier on a range of problem sizes')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('model', type=str, help='model to test {}'.format(model_names()))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--num-points', type=int, default=10,
        help='number of evenly spaced sparsity values to test in the range (0, 1]')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('--output', type=str, default=None, help='output path for CSV')
    parser.add_argument('--append', action='store_true', help='append to output')
    parser.add_argument('--schema', action='store_true',
        help='include the schema as the first line of output')
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    model = get_model(args.model)
    if model is None:
        root_info('error: invalid model {}; valid models are {}', model, model_names())
        sys.exit(1)

    if config.comm.rank == 0:
        if args.output is None:
            f = sys.stdout
        else:
            f = open(args.output, 'ab' if args.append  else 'wb')
        writer = csv.writer(f)

    schema = ['model', 'nodes', 'sparsity', 'positive_train_samples', 'negative_train_samples',
              'positive_test_samples', 'negative_test_samples', 'time_train', 'time_test',
              'fp', 'fn', 'accuracy', 'RMSE']
    if args.schema and config.comm.rank == 0:
        writer.writerow(schema)

    for sparsity in np.linspace(0, 1, num=args.num_points+1):
        if sparsity == 0: continue
        root_debug('{}',output_model_info(model, online=args.online, sparsity=sparsity))

        X, y = prepare_dataset(args.data_dir, sparsity=sparsity)

        result = train_and_test_k_fold(X, y, model, k=args.num_runs, online=args.online)
        root_debug('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

        if config.comm.rank == 0:
            writer.writerow(selectcols(schema,
                model=args.model, sparsity=sparsity, nodes=config.comm.size, **result))

    if args.output and config.comm.rank == 0:
        f.close()
