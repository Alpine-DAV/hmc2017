#! /usr/bin/env python

import argparse
import numpy as np
import sys

import testing_naivebayes as nb
import testing_randomforest as rf
import nbmpi
import rfmpi

from datasets import get_bubbleshock, shuffle_data, discretize, output_feature_importance
from utils import *
import config
from config import comm

def wrapper(ML_type, data_path, verbose=False, use_online=False, use_mpi=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    if comm.rank == 0:
        print "Num Trees:", config.NumTrees

    X, y = get_bubbleshock(data_path)
    shuffle_data(X, y)
    discretized_y = discretize(y)

    if ML_type == config.NAIVE_BAYES:
        if comm.rank == 0:
            print "Training using serial Naive Bayes with k =", config.kfold, "..."
            y = discretized_y
            model = nb.train_and_test_k_fold(X, y, config.kfold)
            print
            print
    elif ML_type == config.RANDOM_FOREST:
        if comm.rank == 0:
            print "Training using serial Random Forest with k =", config.kfold, "..."
            forest = rf.train_and_test_k_fold(X, y, config.kfold)
            output_feature_importance(forest, data_path)
            print
            print
    elif ML_type == config.NAIVE_BAYES_MPI:
        if comm.rank == 0:
            print "Training using parallel Naive Bayes with k =", config.kfold, "..."

            if use_mpi:
                print('will train using MPI')
            if use_online:
                print('will train in online mode')

        y = discretized_y
        result = train_and_test_k_fold(X, y, nbmpi.train, k=config.kfold, verbose=verbose, online=use_online, mpi=use_mpi)

        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)

    elif ML_type == config.RANDOM_FOREST_MPI:
        if comm.rank == 0:
            print "Training using parallel Random Forest..."

            if use_mpi:
                print('will train using MPI')
            if use_online:
                print('will train in online mode')

        result = train_and_test_k_fold(X, y, rfmpi.train, k=config.kfold, verbose=verbose, online=use_online, mpi=use_mpi)
        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)

if __name__ == '__main__':

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the bubbleShock dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('models', type=str, nargs='+', help='models to test {}'.format(config.VALID_MODELS))
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    args = parser.parse_args()

    for model in args.models:
        if model not in config.VALID_MODELS:
            root_info('error: invalid model {}; valid models are {}', model, config.VALID_MODELS)
            sys.exit(1)

    verbose    = args.verbose
    use_online = args.online
    use_mpi = running_in_mpi()

    for model in args.models:
        wrapper(model, args.data_dir, verbose=verbose, use_mpi=use_mpi, use_online=use_online)
