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

def wrapper(model, k, data_path, online=False, density=1.0, pool_size=pool_size, train_test_split=None):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    ds = prepare_dataset(data_path, density=density, pool_size=pool_size, train_test_split=train_test_split)

    root_info('{}', output_model_info(model, online=online, density=density, pool_size=pool_size))

    result = train_and_test_k_fold(ds, model, k=k, online=online, train_test_split=train_test_split)

    root_info('PERFORMANCE\n{}', result)

def get_train_test_split(args):
    val = None
    if args.train_split != None or args.test_split != None:
        train_split = args.train_split if args.train_split != None else TOTAL_CYCLES - args.test_split
        test_split = args.test_split if args.test_split != None else TOTAL_CYCLES - args.train_split
        if train_split+test_split > TOTAL_CYCLES:
            root_info('invalid training and/or testing split supplied; cumulative cycles supplied {} \
                is greater than total number of cycles {}. Results in testing and training set overlap\
                '.format(train_split+test_split, TOTAL_CYCLES))
            sys.exit(1)
        val = {'train_split': train_split, 'test_split': test_split}
    return val

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
    
    # Online Training Specific Parameters
    parser.add_argument('--train-split', type=int, help='specify a value of cycles to train on. If testing-split is left \
         unspecified, the remaining cycles will be trained upon')    
    
    # Online Testing Specific Parameters
    parser.add_argument('--test-split', type=int, help='specify the number of cycles to test upon after training is completed')

    args = parser.parse_args()

    toggle_verbose(args.verbose) 
    toggle_profiling(args.profile)

    for model in args.models:
        m = get_model(model, bootstrap=True, oob_score=True, min_samples_split=config.min_samples_split)
        if m is None:
            root_info('error: invalid model {}; valid models are {}', model, model_names())
            sys.exit(1)
        else:
            train_test_sp = get_train_test_split
            wrapper(m, args.num_runs, args.data_dir, online=args.online, density=args.density, pool_size=args.pool_size,
                    train_test_split=get_train_test_split(args)) 
