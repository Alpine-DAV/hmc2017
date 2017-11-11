#! /usr/bin/env python

import argparse
import numpy as np
import sys

from mpi4py import MPI

import testing_naivebayes as nb
import testing_randomforest as rf
import nbmpi
import rfmpi

from datasets import get_bubbleshock, discretize, output_feature_importance

from utils import *

comm = MPI.COMM_WORLD

NAIVE_BAYES         = 'nb'
NAIVE_BAYES_MPI     = 'nbp'
RANDOM_FOREST       = 'rf'
RANDOM_FOREST_MPI   = 'rfp'

VALID_MODELS = [NAIVE_BAYES, NAIVE_BAYES_MPI, RANDOM_FOREST, RANDOM_FOREST_MPI]

def wrapper(ML_type, k, data_path, verbose=False, use_online=False, use_mpi=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """

    X, y = get_bubbleshock(data_path)
    discretized_y = discretize(y)

    if ML_type == NAIVE_BAYES:
        if comm.rank == 0:
            print "Training using serial Naive Bayes..."
            y = discretized_y
            model = nb.train_and_test_k_fold(X, y, k)
            print
            print
    elif ML_type == RANDOM_FOREST:
        if comm.rank == 0:
            print "Training using serial Random Forest..."
            forest = rf.train_and_test_k_fold_serial_merge(X, y, k)
            output_feature_importance(forest, data_path)
            print
            print
    elif ML_type == NAIVE_BAYES_MPI:
        if comm.rank == 0:
            print "Training using parallel Naive Bayes..."

            if use_mpi:
                print('will train using MPI')
            if use_online:
                print('will train in online mode')

        y = discretized_y

        result = train_and_test_k_fold(X, y, nbmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)

        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)

    elif ML_type == RANDOM_FOREST_MPI:
        if comm.rank == 0:
            print "Training using parallel Random Forest..."

            if use_mpi:
                print('will train using MPI')
            if use_online:
                print('will train in online mode')

        result = train_and_test_k_fold(X, y, rfmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)
        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)

def get_mpi_task_data(X, y, comm = MPI.COMM_WORLD):
    samps_per_task = X.shape[0] // comm.size

    min_bound = samps_per_task*comm.rank
    if comm.rank == comm.size - 1:
        max_bound = X.shape[0]
    else:
        max_bound = min_bound + samps_per_task

    return (X[min_bound:max_bound],
            y[min_bound:max_bound])

def serial_merge_rf(train_X, test_X, train_y, test_y):
    samps_per_task = train_X.shape[0] // comm.size
    trees = []
    for k in range(1,9):
        min_bound = samps_per_task*comm.rank
        if k == 8:
            max_bound = train_X.shape[0]
        else:
            max_bound = min_bound + samps_per_task

        interval = (train_X[min_bound:max_bound], train_y[min_bound:max_bound])


if __name__ == '__main__':
    np.random.seed(0)

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the bubbleShock dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(VALID_MODELS))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    args = parser.parse_args()

    for model in args.models:
        if model not in VALID_MODELS:
            root_info('error: invalid model {}; valid models are {}', model, VALID_MODELS)
            sys.exit(1)

    verbose    = args.verbose
    use_online = args.online
    use_mpi = running_in_mpi()
    k = 2

    for model in args.models:
        wrapper(model, k, args.data_dir, verbose=verbose, use_mpi=use_mpi, use_online=use_online)
