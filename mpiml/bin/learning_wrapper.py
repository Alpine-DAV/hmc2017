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

def wrapper(model, k, data_path, online=False, density=1.0):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    result = None

    ds = prepare_dataset(data_path, density=density)

    root_info('{}', output_model_info(model, online=online, density=density))

    result = train_and_test_k_fold(ds, model, k=k, online=online)
    
    root_info('PERFORMANCE\n{}', result)

    root_info('{}',output_model_info(model, online=online, density=density))

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
    args = parser.parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    for model in args.models:
        m = get_model(model, bootstrap=True, oob_score=True)
        if m is None:
            root_info('error: invalid model {}; valid models are {}', model, model_names())
            sys.exit(1)
        else:
            wrapper(m, args.num_runs, args.data_dir, online=args.online, density=args.density)
