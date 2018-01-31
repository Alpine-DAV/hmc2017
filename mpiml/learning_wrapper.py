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

def wrapper(model, k, data_path, training_cycles=TOTAL_CYCLES/2, testing_cycles=TOTAL_CYCLES-TOTAL_CYCLES/2, online=False, online_pool=1):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    # TODO: Modify this to deal with whatever the standard dataset is, not bubbleshock 
    result = {}
    if "byHand" in data_path:
        train_time = 0
        test_time  = 0
        negative_train_samples = 0
        positive_train_samples = 0
        negative_test_samples  = 0
        positive_test_samples  = 0

        cycle = 0
        X, y, rem_X, rem_y = [], [], [], []
        while cycle < TOTAL_CYCLES and cycle < training_cycles: 
            cycle_X, cycle_y = get_bubbleshock_byhand_by_cycle(data_path, cycle)
            cycle += 1

            X.extend(cycle_X)
            y.extend(cycle_y)

            # pool cycles until data above online_pool for all processes
            if len(y) < online_pool*comm.size:
                continue
            
            get_pool_samples(X, y, rem_X, rem_y, online_pool)
            if running_in_mpi():
                X, y = get_mpi_task_data(X, y)
            train_time += train_by_cycle(X, y, model, online=online, online_pool=online_pool)
            
            train_pos, train_neg = num_classes(y)
            positive_train_samples += train_pos
            negative_train_samples += train_neg

            X = rem_X
            y = rem_y
            root_info('trained through cycle: {}'.format(cycle))
        
        # train on remaining samples from cycles
        while len(y) != 0:
            get_pool_samples(X, y, rem_X, rem_y, online_pool)
            if running_in_mpi():
                X, y = get_mpi_task_data(X, y)
            train_time += train_by_cycle(X, y, model, online=online, online_pool=online_pool)
            train_pos, train_neg = num_classes(y)
            positive_train_samples += train_pos
            negative_train_samples += train_neg
            X = rem_X
            y = rem_y
        
        if running_in_mpi(): 
            root_info('Done training by cycle, reducing and testing.')
            model = model.reduce()
            positive_train_samples = comm.reduce(positive_train_samples, op=MPI.SUM, root=0)
            negative_train_samples = comm.reduce(negative_train_samples, op=MPI.SUM, root=0)

        if comm.rank == 0:
            fp = 0
            fn = 0
            RMSE_sum   = 0
            RMSE_total = 0
            while cycle < TOTAL_CYCLES and cycle < training_cycles+testing_cycles:    
                X, y = get_bubbleshock_byhand_by_cycle(data_path, cycle)
                if running_in_mpi():
                    X, y = get_mpi_task_data(X, y)
                results_partial = test_by_cycle(X, y, model, online=online, online_pool=online_pool)
                fp += results_partial['fp']
                fn += results_partial['fn']
                
                RMSE_sum += results_partial['RMSE_partial']
                RMSE_total += len(y)
                
                test_pos, test_neg = num_classes(y)
                positive_test_samples += test_pos
                negative_test_samples += test_neg

                test_time += results_partial['cycle_test_time']
                
                cycle += 1


            result = {
                'fp': fp,
                'fn': fn,
                'RMSE': RMSE_sum / RMSE_total,
                'accuracy': 1 - ((fp + fn) / (RMSE_total)),
                'time_train': train_time,
                'time_test': test_time,
                'runs': 1,
                'negative_train_samples': negative_train_samples,
                'positive_train_samples': positive_train_samples,
                'negative_test_samples': negative_test_samples,
                'positive_test_samples': positive_test_samples,
                'clf': model
            }
    else:
        X, y = get_bubbleshock(data_path)
        shuffle_data(X, y)
        result = train_and_test_k_fold(X, y, model, k=k, online=online)

    root_info('{}',output_model_info(model, online=online))
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
    parser.add_argument('--online-pool', type=int, default=1, help='specify the pooling of values to train the online classifier upon')
    parser.add_argument('--training-cycles', type=int, default=TOTAL_CYCLES/2,
        help='number of cycles to train on before testing for the bubbleShock_byHand dataset')
    parser.add_argument('--testing-cycles', type=int, default=TOTAL_CYCLES-TOTAL_CYCLES/2,
        help='number of cycles to test on for the bubbleShock_byHand dataset')
    args = parser.parse_args()

    for model in args.models:
        if model not in models:
            root_info('error: invalid model {}; valid models are {}', model, models.keys())
            sys.exit(1)

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    for model in args.models:
        wrapper(models[model](), args.num_runs, args.data_dir, online=args.online, online_pool=args.online_pool,
            training_cycles=args.training_cycles, testing_cycles=args.testing_cycles)
