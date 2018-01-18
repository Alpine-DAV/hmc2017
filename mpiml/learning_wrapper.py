#! /usr/bin/env python

import argparse
import numpy as np
import sys

from mpi4py import MPI

from nbmpi import GaussianNB
from forest import RandomForestRegressor, MondrianForestRegressor

from datasets import get_bubbleshock, get_bubbleshock_byhand_by_cycle, discretize, output_feature_importance, shuffle_data

from utils import *
from config import *

def wrapper(model, k, data_path, online=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    # TODO: Modify this to deal with whatever the standard dataset is, not bubbleshock 
    X, y = get_bubbleshock(data_path)
    # X, y = get_bubbleshock_byhand_by_cycle(data_path, 10000)
    shuffle_data(X, y)

    root_info('{}',output_model_info(model, online=online))

    result = train_and_test_k_fold(X, y, model, k=k, online=online)
    root_info('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

if __name__ == '__main__':

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the bubbleShock dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(models.keys()))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    args = parser.parse_args()

    for model in args.models:
        if model not in models:
            root_info('error: invalid model {}; valid models are {}', model, models.keys())
            sys.exit(1)

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    for model in args.models:
        wrapper(models[model](), args.num_runs, args.data_dir, online=args.online)
