from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
import os
import sys
import time

import config

__all__ = [ "info"
          , "debug"
          , "root_info"
          , "root_debug"
          , "accuracy"
          , "get_k_fold_data"
          , "get_mpi_task_data"
          , "running_in_mpi"
          , "train_and_test_k_fold"
          , "default_trainer"
          , "fit"
          , "output_model_info"
          , "prettify_train_and_test_k_fold_results"
          , "num_classes"
          , "toggle_profiling"
          , "toggle_verbose"
          ]

_verbose = False

def toggle_verbose(v=True):
    global _verbose
    _verbose = v

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

def debug(fmt, *args, **kwargs):
    if _verbose:
        info(fmt, *args, **kwargs)

def root_info(fmt, *args, **kwargs):
    comm = _extract_arg('comm', MPI.COMM_WORLD, kwargs)
    root = _extract_arg('root', 0, kwargs)
    if comm.rank != root:
        return
    if type(fmt) != str:
        args = [fmt]
        fmt = '{}'
    _info(fmt, *args, **kwargs)

def root_debug(fmt, *args, **kwargs):
    if _verbose:
        root_info(fmt, *args, **kwargs)

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
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ or \
           'MPIRUN_ID' in os.environ

_profiling_enabled = False
def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            if not _profiling_enabled:
                return f(*args, **kwargs)

            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)
            return result
        return wrap_f
    return prof_decorator

def toggle_profiling(enabled=True):
    global _profiling_enabled
    _profiling_enabled = enabled

def default_trainer(X, y, clf, online=False, online_pool=1, classes=None, **kwargs):
    fit(clf, X, y, online=online, online_pool=online_pool, classes=classes)
    if running_in_mpi():
        clf.reduce()
    return clf

# Train and test a model using k-fold cross validation (default is 10-fold).
# Return a dictionary of statistics, which can be passed to prettify_train_and_test_k_fold_results
# to get a readable performance summary.
# If running in MPI, only root (rank 0) has a meaningful return value.
# `trainer` should be a function which, given a feature vector, a label vector, and a model, trains
# the model on the data and returns the trained model. In addition, kwargs passed to
# train_and_test_k_fold will be forwarded train. The default trainer calls either fit or
# partial_fit and then reduce.
@profile('train_and_test_k_fold_prof')
def train_and_test_k_fold(X, y, clf, trainer=default_trainer, k=10, comm=MPI.COMM_WORLD, classes=None, **kwargs):
    classes = np.unique(y) if classes is None else classes
    kwargs.update(trainer=trainer, k=k, comm=comm, classes=classes)

    runs = 0
    fp_accum = 0
    fn_accum = 0
    train_pos_accum = 0
    train_neg_accum = 0
    test_pos_accum = 0
    test_neg_accum = 0
    time_train = 0
    time_test = 0

    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y, k=k):
        train_y_orig = train_y

        if running_in_mpi():
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        debug('training with {} samples', train_X.shape[0])

        start_train = time.time()
        clf = trainer(train_X, train_y, clf, **kwargs)
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

            train_pos, train_neg = num_classes(train_y_orig)
            train_pos_accum += train_pos
            train_neg_accum += train_neg
            test_pos, test_neg = num_classes(test_y)
            test_pos_accum += test_pos
            test_neg_accum += test_neg

            RMSE = np.sqrt( sum(pow(test_y - prd, 2)) / test_y.size )

            runs += 1
            root_debug('run {}: {} false positives, {} false negatives', runs, fp, fn)
            root_debug('final model: {}', clf)

        comm.barrier()

    if comm.rank == 0:
        return {
            'fp': fp_accum / runs,
            'fn': fn_accum / runs,
            'RMSE': RMSE,
            'accuracy': 1 - ((fp_accum + fn_accum) / (train_neg_accum + train_pos_accum)),
            'time_train': time_train / runs,
            'time_test': time_test / runs,
            'runs': runs,
            'negative_train_samples': train_neg_accum / runs,
            'positive_train_samples': train_pos_accum / runs,
            'negative_test_samples': test_neg_accum / runs,
            'positive_test_samples': test_pos_accum / runs,
            'clf': clf
        }
    else:
        return {}

def output_model_info(model, online):
    output_str = \
"""
---------------------------
ML model:  {ml_type}
num cores: {num_cores}
MPI:       {use_mpi}
online:    {use_online}
---------------------------
""".format(ml_type=model.__class__.__name__,
           num_cores=config.comm.size,
           use_mpi=running_in_mpi(), use_online=online)

    if hasattr(model, 'estimators_'):
        output_str += \
"""
num trees: {num_trees}
---------------------------
""".format(num_trees=len(model.estimators_))

    return output_str

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
    RMSE:                       {RMSE}

(statistics averaged over {runs} runs)
""".format(train_total = d['negative_train_samples'] + d['positive_train_samples'],
           test_total = d['negative_test_samples'] + d['positive_test_samples'],
           **d)

def fit(clf, X, y, classes=None, online=False, online_pool=1):
    if online:
        classes = np.unique(y) if classes is None else classes
        for i in xrange(0,X.shape[0],online_pool):
            clf.partial_fit(X[i:i+online_pool], y[i:i+online_pool], classes=classes)
    else:
        clf.fit(X,y)
    return clf

if not running_in_mpi():
    root_info("WARNING: NOT RUNNING WITH MPI")
