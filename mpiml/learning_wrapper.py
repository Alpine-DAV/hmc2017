#! /usr/bin/env python

import argparse
import numpy as np
import sys

from mpi4py import MPI

import nbmpi
import rfmpi

from datasets import get_bubbleshock, get_bubbleshock_byhand_by_cycle, discretize, output_feature_importance, shuffle_data

from utils import *
from config import *

def wrapper(ML_type, k, data_path, use_online=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """

    X, y = get_bubbleshock(data_path)
    # X, y = get_bubbleshock_byhand_by_cycle(data_path, 10000)
    shuffle_data(X, y)
    discretized_y = discretize(y)

    root_info('{}',output_model_info(ML_type, online=use_online))

    if ML_type == NAIVE_BAYES:
        y = discretized_y

        result = train_and_test_k_fold(X, y, nbmpi.train, k=k, online=use_online)
        root_info('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

    elif ML_type == RANDOM_FOREST:

        result = train_and_test_k_fold(X, y, rfmpi.train, k=k, online=use_online)
        root_info('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

    else:
        raise Exception('Machine learning algorithm not recognized')

if __name__ == '__main__':

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the bubbleShock dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(VALID_MODELS))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    args = parser.parse_args()

    for model in args.models:
        if model not in VALID_MODELS:
            root_info('error: invalid model {}; valid models are {}', model, VALID_MODELS)
            sys.exit(1)

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    use_online = args.online

    for model in args.models:
        wrapper(model, args.num_runs, args.data_dir, use_online=use_online)
