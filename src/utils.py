from __future__ import division
from __future__ import print_function

from mpi4py import MPI
import numpy as np
import os
import sys
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from skgarden.mondrian.ensemble import MondrianForestClassifier

__all__ = [ "info"
          , "root_info"
          , "accuracy"
          , "get_k_fold_data"
          , "get_mpi_task_data"
          , "get_testing_data"
          , "running_in_mpi"
          , "train_and_test_k_fold"
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
def num_errors(actual, predicted):
    decision_boundary = 4e-6
    fp = fn = 0
    for i in range(len(actual)):
        if actual[i] == 0 and predicted[i] > decision_boundary:
            fp += 1
        elif actual[i] > 0 and predicted[i] <= decision_boundary:
            fn += 1
    return fp, fn

# Return a pair of the number of positive examples (class > 0) and the number of negative examples
# (class == 0)
def num_classes(y):
    return np.sum(y != 0), np.sum(y == 0)

# A generator yielding a tuple of (training features, training labels, test features, test labels)
# for each run in a k-fold cross validation experiment. By default, k=10.
def get_k_fold_data(X, y, k=10):
    n_samples = X.shape[0]
    n_test = n_samples // k
    for i in range(k):
        train_X = np.concatenate((X[:n_test*i], X[n_test*(i+1):]))
        test_X = X[n_test*i:n_test*(i+1)]
        train_y = np.concatenate((y[:n_test*i], y[n_test*(i+1):]))
        test_y = y[n_test*i:n_test*(i+1)]
        yield (train_X, test_X, train_y, test_y)

# Get a subset of a dataset for the current task. If each task in an MPI communicator calls this
# function, then every sample in the dataset will be distributed to exactly one task.
def get_mpi_task_data(X, y, comm = MPI.COMM_WORLD):
    samps_per_task = X.shape[0] // comm.size

    min_bound = samps_per_task*comm.rank
    if comm.rank == comm.size - 1:
        max_bound = X.shape[0]
    else:
        max_bound = min_bound + samps_per_task

    return (X[min_bound:max_bound],
            y[min_bound:max_bound])

def get_testing_data(X, y, comm):
    samps_per_task = X.shape[0] // (comm.size+1)
    min_bound = samps_per_task*(comm.rank+1)
    max_bound = X.shape[0]

    return (X[min_bound:max_bound],
            y[min_bound:max_bound])

# Determine if we are running as an MPI process
def running_in_mpi():
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ

# Train and test a model using k-fold cross validation (default is 10-fold). Return the false 
# positives, false negatives, training time and testing time over all k runs (testing on root).
# If running in MPI, only root (rank 0) has a meaningful return value. `train` should be a
# function which, given a feature vector and a class vector, returns a trained instance of the
# desired model.
def train_and_test_k_fold(X, y, train, k=10, verbose=False, comm=MPI.COMM_WORLD, root=0, **kwargs):
    kwargs.update(verbose=verbose, k=k, comm=comm)
    fp_accum = fn_accum = 0
    test_accum = 0
    time_train = time_test = 0
    
    runs = 0
    classes = np.unique(y)
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y, k=k):
        if running_in_mpi():
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        if verbose:
            info('training with {} samples'.format(train_X.shape[0]))

        start_train = time.time()
        clf = train(train_X, train_y, classes=classes, **kwargs)
        end_train = time.time()

        if comm.rank == root:
            # Only root has the final model, so only root does the predicting
            start_test = time.time()
            prd = clf.predict(test_X)
            end_test = time.time()

            time_train += end_train-start_train
            time_test += end_test - start_test

            fp, fn = num_errors(test_y, prd)
            fp_accum += fp
            fn_accum += fn
            test_accum += len(test_y)
            runs += 1
            if verbose:
                print('run {}: {} false positives, {} false negatives'.format(runs, fp, fn))
                print('final model: {}'.format(clf))

        comm.barrier()

    if comm.rank == root:
        return fp_accum, fn_accum, test_accum, time_train, time_test
    else:
        # This allows us to tuple destructure the result of this function without checking whether
        # we are root
        return None, None, None, None, None

online_classifiers = [
    GaussianNB,
    MondrianForestClassifier
]

methods = [
    'batch',
    'online'
]

def train_with_method(clf, X, y, **kwargs):
    if 'method' not in kwargs:
        kwargs['method'] = 'batch'
    if clf not in online_classifiers or kwargs['method'] == 'batch':
        if clf not in online_classifiers and kwargs['method'] != 'batch':
            print('Forcing batch training for non-online classifier method')
        clf.fit(X,y)
    elif kwargs['method'] == 'online':
        for i in range(X.shape[0]):
            clf.partial_fit(X[i], y[i])
    else:
        raise ValueError("Invalid argument supplied for --method flag. \
                 Please use one of the following: %s", methods)
    return clf