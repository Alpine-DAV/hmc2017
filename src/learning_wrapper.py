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
            print "############ Training using Naive Bayes ############"
            y = discretized_y
            result = nb.train_and_test_k_fold(X, y, k)
            print "PERFORMANCE\t%s" % (result,)
            print
            print
    elif ML_type == RANDOM_FOREST:
        if comm.rank == 0:
            print "############ Training using Random Forest ############"
            forest = rf.train_and_test_k_fold(X, y, k)
            output_feature_importance(forest, "bubbleShock")
            print "PERFORMANCE\t%s" % (result,)
            print
            print
    elif ML_type == NAIVE_BAYES_MPI:
        if comm.rank == 0:
            print "############ Training using Parallel Naive Bayes ############"

        y = discretized_y

        if use_mpi and comm.rank == 0:
            print('will train using MPI')
        if use_online and comm.rank == 0:
            print('will train in online mode')

        result = train_and_test_k_fold(X, y, nbmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)

        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)

    elif ML_type == RANDOM_FOREST_MPI:
        if comm.rank == 0:
            print "############ Training using Parallel Random Forest ############"

        if use_mpi and comm.rank == 0:
            print('will train using MPI')
        if use_online and comm.rank == 0:
            print('will train in online mode')

        result = train_and_test_k_fold(X, y, rfmpi.train, k=k, verbose=verbose, online=use_online, mpi=use_mpi)
        if comm.rank == 0:
            print "PERFORMANCE\t%s" % (result,)


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
