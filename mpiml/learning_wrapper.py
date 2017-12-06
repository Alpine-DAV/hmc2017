#! /usr/bin/env python

import argparse
import numpy as np
import sys

from mpi4py import MPI

import testing_naivebayes as nb
import testing_randomforest as rf
import nbmpi
import rfmpi

from datasets import get_bubbleshock, discretize, output_feature_importance, shuffle_data

from utils import *
from config import *

def wrapper(ML_type, k, data_path, verbose=False, use_online=False, use_mpi=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """

    X, y = get_bubbleshock(data_path)
    shuffle_data(X, y)
    discretized_y = discretize(y)

    root_info('{}',output_model_info(ML_type, mpi=use_mpi, online=use_online))

    if ML_type == NAIVE_BAYES:
        if comm.rank == 0:
            y = discretized_y
            result = nb.train_and_test_k_fold(X, y, k)

    elif ML_type == RANDOM_FOREST:
        if comm.rank == 0:
            forest = rf.train_and_test_k_fold(X, y, k)
            output_feature_importance(forest, data_path)

    elif ML_type == NAIVE_BAYES_MPI:
        y = discretized_y

        result = train_and_test_k_fold(X, y, nbmpi.train, k=k, vey, online=use_online, mpi=use_mpi)
        root_info('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

    elif ML_type == RANDOM_FOREST_MPI:

        result = train_and_test_k_fold(X, y, rfmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)
        root_info('PERFORMANCE\n{}', prettify_train_and_test_k_fold_results(result))

    # elif ML_type == RANDOM_FOREST_NO_MERGE:
    #     if not use_mpi:
    #         print('You are trying to run Random Forest No Merge without MPI.')
    #         print('This is pointless')
    #         raise Exception('ML Algorithm only for use with MPI')
    #     if use_online and comm.rank == 0:
    #         print('will train in online mode')

    #     result = train_and_test_k_fold_no_merge(X, y, rfmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)
    #     if comm.rank == 0:
    #         print "PERFORMANCE\t%s" % (result,)
    else:
        raise Exception('Machine learning algorithm not recognized')

if __name__ == '__main__':

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
    k = 10

    for model in args.models:
        wrapper(model, k, args.data_dir, verbose=verbose, use_mpi=use_mpi, use_online=use_online)
