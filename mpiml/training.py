from __future__ import division

import numpy as np
import sklearn.base as sk
import time

import config
from datasets import empty_dataset, concatenate, threshold_count, discretize
from utils import *

__all__ = [ "get_k_fold_data"
          , "default_trainer"
          , "train_and_test_k_fold"
          , "train_and_test_once"
          , "prettify_train_and_test_k_fold_results"
          , "fit"
          ]

def default_trainer(ds, clf, online=False, classes=None, **kwargs):
    if isinstance(clf, sk.ClassifierMixin):
        ds = discretize(ds)
    elif not isinstance(clf, sk.RegressorMixin):
        raise TypeError('expected classifier or regressor, but got {}'.format(type(ds)))

    fit(clf, ds, online=online, classes=classes)
    if running_in_mpi():
        clf = clf.reduce()
    return clf

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

# A generator yielding a tuple of (training set, testing set) for each run in a k-fold cross
# validation experiment. By default, k=10.
def get_k_fold_data(ds, k=10):
    splits = ds.split(k)
    for i in range(k):
        yield (concatenate(splits[j] for j in range(k) if j != i), splits[i])

# Get a subset of a dataset for the current task. If each task in an MPI communicator calls this
# function, then every sample in the dataset will be distributed to exactly one task.
def get_mpi_task_data(ds, comm=config.comm):
    return ds.split(comm.size)[comm.rank]

# Train and test a model using k-fold cross validation (default is 10-fold).
# Return a dictionary of statistics, which can be passed to prettify_train_and_test_k_fold_results
# to get a readable performance summary.
# If running in MPI, only root (rank 0) has a meaningful return value.
# `trainer` should be a function which, given a feature vector, a label vector, and a model, trains
# the model on the data and returns the trained model. In addition, kwargs passed to
# train_and_test_k_fold will be forwarded train. The default trainer calls either fit or
# partial_fit and then reduce.
@profile('train_and_test_k_fold_prof')
def train_and_test_k_fold(ds, prd, k=10, comm=config.comm, **kwargs):
    kwargs.update(k=k, comm=comm)

    if k <= 0:
        raise ValueError("k must be positive")
    elif k == 1:
        splits = ds.split(10)
        train = concatenate(splits[j] for j in range(9))
        test = splits[9]
        return train_and_test_once(train, test, prd, **kwargs)

    runs = 0
    fp_accum = 0
    fn_accum = 0
    train_pos_accum = 0
    train_neg_accum = 0
    test_pos_accum = 0
    test_neg_accum = 0
    time_train = 0
    time_test = 0
    rmse_accum = 0

    for train, test in get_k_fold_data(ds, k=k):
        if running_in_mpi():
            train = get_mpi_task_data(train)

        res = train_and_test_once(train, test, prd, **kwargs)

        if comm.rank == 0: # Only root has the final model
            time_test += res['time_test']
            fp_accum += res['fp']
            fn_accum += res['fn']
            train_pos_accum += res['positive_train_samples']
            train_neg_accum += res['negative_train_samples']
            test_pos_accum += res['positive_test_samples']
            test_neg_accum += res['negative_test_samples']
            rmse_accum += res['RMSE']
            runs += 1

        comm.barrier()

    if comm.rank == 0:
        return {
            'fp': fp_accum / runs,
            'fn': fn_accum / runs,
            'RMSE': rmse_accum / runs,
            'accuracy': 1 - ((fp_accum + fn_accum) / (train_neg_accum + train_pos_accum)),
            'time_train': time_train / runs,
            'time_test': time_test / runs,
            'runs': runs,
            'negative_train_samples': train_neg_accum / runs,
            'positive_train_samples': train_pos_accum / runs,
            'negative_test_samples': test_neg_accum / runs,
            'positive_test_samples': test_pos_accum / runs,
            'prd': prd
        }
    else:
        return {}

def train_and_test_once(train, test, prd, trainer=default_trainer, comm=config.comm, **kwargs):
    kwargs.update(trainer=trainer, k=k, comm=comm)

    if running_in_mpi():
        train = get_mpi_task_data(train)

    start_train = time.time()
    prd = trainer(ds, prd, **kwargs)
    end_train = time.time()
    time_train = end_train - start_train

    if type(prd) == type([]):
        prd = prd[0]

    if comm.rank == 0:
        # Only root has the final model, so only root does the predicting
        start_test = time.time()
        test_X, test_y = test.points()
        out = prd.predict(test_X)
        end_test = time.time()
        time_test = end_test - start_test

        fp, fn = num_errors(test_y, out)

        train_pos, train_neg = threshold_count(train, 1e-6)
        test_pos, test_neg = threshold_count(test, 1e-6)

        RMSE = np.sqrt( sum(pow(test_y - out, 2)) / test_y.size )

    comm.barrier()

    if comm.rank == 0:
        return {
            'fp': fp,
            'fn': fn,
            'RMSE': RMSE,
            'accuracy': 1 - ((fp + fn) / (train_neg + train_pos)),
            'time_train': time_train,
            'time_test': time_test,
            'runs': 1,
            'negative_train_samples': train_neg,
            'positive_train_samples': train_pos,
            'negative_test_samples': test_neg,
            'positive_test_samples': test_pos,
            'prd': prd
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
    RMSE:                       {RMSE}

(statistics averaged over {runs} runs)
""".format(train_total = d['negative_train_samples'] + d['positive_train_samples'],
           test_total = d['negative_test_samples'] + d['positive_test_samples'],
           **d)

def fit(clf, ds, classes=None, online=False):
    if online:
        for X, y in ds.cycles():
            clf.partial_fit(X, y, classes=ds.classes())
    else:
        clf.fit(*ds.points())
    return clf
