from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
import os
import sys
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from skgarden.mondrian.ensemble import MondrianForestRegressor

__all__ = [ "info"
          , "root_info"
          , "accuracy"
          , "get_k_fold_data"
          , "get_mpi_task_data"
          , "running_in_mpi"
          , "train_and_test_k_fold"
          , "prettify_train_and_test_k_fold_results"
          , "num_classes"
          , "train_with_method"
          ]

def _extract_arg(arg, default, kwargs):
    if arg in kwargs:
        res = kwargs[arg]
        del kwargs[arg]
        return res
    else:
        return default

def _info(fmt, *args, **kwargs):
    print(fmt.format(*args, **kwargs), file=sys.stderr)

def info(fmt, *args, **kwargs):
    comm = _extract_arg('comm', MPI.COMM_WORLD, kwargs)
    if type(fmt) == str:
        fmt = 'rank {}: ' + fmt
    else:
        args = [fmt]
        fmt = '{}'
    _info(fmt, comm.rank, *args, **kwargs)

def root_info(fmt, *args, **kwargs):
    comm = _extract_arg('comm', MPI.COMM_WORLD, kwargs)
    root = _extract_arg('root', 0, kwargs)
    if comm.rank != root:
        return
    if type(fmt) != str:
        args = [fmt]
        fmt = '{}'
    _info(fmt, *args, **kwargs)

# Compute the accuracy of a set of predictions against the ground truth values.
def accuracy(actual, predicted):
    return np.sum(predicted == actual) / actual.shape[0]

# Compute number of false positives and false negatives in a set of predictions
def num_errors(actual, predicted, threshold=4e-6):
    fp = fn = 0
    for i in range(len(actual)):
        if actual[i] <= threshold and predicted[i] > threshold:
            fp += 1
        elif actual[i] > threshold and predicted[i] <= threshold:
            fn += 1
    return fp, fn

# Return a pair of the number of positive examples (class > threshold) and the number of negative
# examples (class <= threshold)
def num_classes(y, threshold=4e-6):
    return np.sum(y > threshold), np.sum(y <= threshold)

# A generator yielding a tuple of (training features, training labels, test features, test labels)
# for each run in a k-fold cross validation experiment. By default, k=10.
def get_k_fold_data(X, y, k=10):
    X_splits = np.array_split(X, k)
    y_splits = np.array_split(y, k)
    for i in range(k):
        train_X = np.concatenate([X_splits[j] for j in range(k) if j != i])
        test_X = X_splits[i]
        train_y = np.concatenate([y_splits[j] for j in range(k) if j != i])
        test_y = y_splits[i]
        yield (train_X, test_X, train_y, test_y)

# Get a subset of a dataset for the current task. If each task in an MPI communicator calls this
# function, then every sample in the dataset will be distributed to exactly one task.
def get_mpi_task_data(X, y, comm = MPI.COMM_WORLD):
    return np.array_split(X, comm.size)[comm.rank], \
           np.array_split(y, comm.size)[comm.rank]

# Determine if we are running as an MPI process
def running_in_mpi():
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ

# Train and test a model using k-fold cross validation (default is 10-fold).
# Return a dictionary of statistics, which can be passed to prettify_train_and_test_k_fold_results
# to get a readable performance summary.
# If running in MPI, only root (rank 0) has a meaningful return value.
# `train` should be a function which, given a feature vector and a class vector, returns a trained
# instance of the desired model. In addition, kwargs passed to train_and_test_k_fold will be
# forwarded train.
def train_and_test_k_fold(X, y, train, verbose=False, k=10, comm=MPI.COMM_WORLD, **kwargs):
    kwargs.update(verbose=verbose, k=k, comm=comm)

    runs = 0
    fp_accum = 0
    fn_accum = 0
    train_pos_accum = 0
    train_neg_accum = 0
    test_pos_accum = 0
    test_neg_accum = 0
    time_train = 0
    time_test = 0

    classes = np.unique(y)
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y, k=k):
        if running_in_mpi():
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        if verbose:
            info('training with {} samples'.format(comm.rank, train_X.shape[0]))

        start_train = time.time()
        clf = train(train_X, train_y, **kwargs)
        end_train = time.time()
        time_train += end_train - start_train

        if type(clf) == type([]):
            clf = clf[0]

        if comm.rank == 0:
            # Only root has the final model, so only root does the predicting

            start_test = time.time()
            prd = clf.predict(test_X)
            end_test = time.time()
            time_test += end_test - start_test

            fp, fn = num_errors(test_y, prd)
            fp_accum += fp
            fn_accum += fn

            train_pos, train_neg = num_classes(train_y)
            train_pos_accum += train_pos
            train_neg_accum += train_neg
            test_pos, test_neg = num_classes(test_y)
            test_pos_accum += test_pos
            test_neg_accum += test_neg

            runs += 1
            if verbose:
                root_info('run {}: {} false positives, {} false negatives', runs, fp, fn)
                root_info('final model: {}', clf)

        comm.barrier()

    if comm.rank == 0:
        return {
            'fp': fp_accum / runs,
            'fn': fn_accum / runs,
            'accuracy': 1 - ((fp_accum + fn_accum) / (train_neg_accum + train_pos_accum)),
            'time_train': time_train / runs,
            'time_test': time_test / runs,
            'runs': runs,
            'negative_train_samples': train_neg_accum / runs,
            'positive_train_samples': train_pos_accum / runs,
            'negative_test_samples': test_neg_accum / runs,
            'positive_test_samples': test_pos_accum / runs
        }
    else:
        return {}

def prettify_train_and_test_k_fold_results(d):
    if d:
        return \
"""
timing
    train:                      {time_train}
    test:                       {time_test}
dataset
    negative training examples: {negative_train_samples}
    positive training examples: {positive_train_samples}
    total training examples:    {train_total}
    negative testing examples:  {negative_test_samples}
    positive testing examples:  {positive_test_samples}
    total testing examples:     {test_total}
performance
    false positives:            {fp}
    false negatives:            {fn}
    accuracy:                   {accuracy}

(statistics averaged over {runs} runs)
""".format(train_total = d['negative_train_samples'] + d['positive_train_samples'],
           test_total = d['negative_test_samples'] + d['positive_test_samples'],
           **d)

online_classifiers = (
    GaussianNB,
    MondrianForestRegressor
)

methods = [
    'batch',
    'online'
]

def train_with_method(clf, X, y, **kwargs):
    if 'method' not in kwargs:
        kwargs['method'] = 'batch'
    if not isinstance(clf, online_classifiers) or kwargs['method'] == 'batch':
        if not isinstance(clf, online_classifiers) and kwargs['method'] != 'batch':
            print('Forcing batch training for non-online classifier method')
        clf.fit(X,y)
    elif kwargs['method'] == 'online':
        for i in range(X.shape[0]):
            clf.partial_fit(X[i:i+1], y[i:i+1])
    else:
        raise ValueError("Invalid argument supplied for --method flag. \
                 Please use one of the following: %s", methods)
    return clf
