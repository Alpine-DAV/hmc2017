from __future__ import division

import numpy as np
import sklearn.base as sk
import time

import config
from datasets import concatenate, threshold_count, discretize, prepare_dataset
from utils import *

__all__ = [ "get_k_fold_data"
          , "train_and_test_k_fold"
          , "train_and_test_once"
          , "fit"
          ]

# Compute the accuracy of a set of predictions against the ground truth values.
def accuracy(actual, predicted):
    return np.sum(predicted == actual) / actual.shape[0]

# Compute number of false positives and false negatives in a set of predictions
def num_errors(actual, predicted, threshold=config.decision_boundary):
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

# Concatenate datasets at the beginning of the split if the k-folding pattern results in train or test
# set containing the end and beginning StrictDataSets
def wrapped_concatenate(splits, start, end, total_cycles):
    rng = range(start, min(end, total_cycles))
    if end > total_cycles:
        rng.extend(range(0, end%total_cycles))
    return concatenate(splits[j] for j in rng)

# Get a subset of a dataset for the current task. If each task in an MPI communicator calls this
# function, then every sample in the dataset will be distributed to exactly one task.
def get_mpi_task_data(ds, comm=config.comm, task=None):
    if task is None:
        task = comm.rank
    return ds.split(comm.size)[task]

class TrainingResult(object):

    def __init__(self, time_load, time_train, time_reduce, time_test, fp, fn,
                 positive_train_samples, negative_train_samples,
                 positive_test_samples, negative_test_samples,
                 rmse, runs=1):

        self.runs = runs
        self.time_load = time_load
        self.time_train = time_train
        self.time_reduce = time_reduce
        self.time_test = time_test
        self.fp = fp
        self.fn = fn
        self.positive_train_samples = positive_train_samples
        self.negative_train_samples = negative_train_samples
        self.positive_test_samples = positive_test_samples
        self.negative_test_samples = negative_test_samples
        self.rmse = rmse

    @property
    def accuracy(self):
        return 1 - ((self.fp + self.fn) / (self.negative_train_samples + self.positive_train_samples))

    def __add__(self, r):
        def average(prop):
            if self.runs == 0:
                return getattr(r, prop)
            elif r.runs == 0:
                return getattr(self, prop)
            else:
                return (getattr(self, prop)*self.runs + getattr(r, prop)*r.runs) / (self.runs + r.runs)

        return TrainingResult(
            time_load=average('time_load'),
            time_train=average('time_train'),
            time_reduce=average('time_reduce'),
            time_test=average('time_test'),
            fp=average('fp'),
            fn=average('fn'),
            positive_train_samples=average('positive_train_samples'),
            negative_train_samples=average('negative_train_samples'),
            positive_test_samples=average('positive_test_samples'),
            negative_test_samples=average('negative_test_samples'),
            rmse=average('rmse'),
            runs=self.runs + r.runs
        )

    def __str__(self):
        return \
"""
timing
    load:                       {time_load}
    train:                      {time_train}
    reduce:                     {time_reduce}
    test:                       {time_test}
dataset
    negative training examples: {negative_train_samples}
    positive training examples: {positive_train_samples}
    negative testing examples:  {negative_test_samples}
    positive testing examples:  {positive_test_samples}
performance
    false positives:            {fp}
    false negatives:            {fn}
    accuracy:                   {accuracy}
    RMSE:                       {rmse}

(statistics averaged over {runs} runs)
""".format(time_load=self.time_load, time_train=self.time_train,
           time_reduce=self.time_reduce, time_test=self.time_test,
           negative_train_samples=self.negative_train_samples,
           positive_train_samples=self.positive_train_samples,
           negative_test_samples=self.negative_test_samples,
           positive_test_samples=self.positive_test_samples,
           fp=self.fp,
           fn=self.fn,
           accuracy=self.accuracy,
           rmse=self.rmse,
           runs=self.runs)

def null_training_result():
    return TrainingResult(*([None]*11), runs=0)

# Train and test a model using k-fold cross validation (default is 10-fold).
@profile('train_and_test_k_fold_prof')
def train_and_test_k_fold(ds, prd, k=10, comm=config.comm, online=False, classes=None):

    train_and_test = lambda tr, te: train_and_test_once(
        tr, te, prd, comm=comm, online=online, classes=classes)

    if k <= 0:
        raise ValueError("k must be positive")
    elif k == 1:
        splits = ds.split(10)
        train = concatenate(splits[j] for j in range(9))
        test = splits[9]
        return train_and_test(train, test)
    else:
        r = null_training_result()
        for train, test in get_k_fold_data(ds, k=k):
            r += train_and_test(train, test)
            comm.barrier()

        return r

def train_and_test_once(train, test, prd, comm=config.comm, online=False, classes=None):

    if running_in_mpi():
        train = get_mpi_task_data(train)

    if isinstance(prd, sk.ClassifierMixin):
        train = discretize(train)
        test = discretize(test)
    elif not isinstance(prd, sk.RegressorMixin):
        raise TypeError('expected classifier or regressor, but got {}'.format(type(ds)))

    prd, time_train, time_load = fit(
        prd, train, online=online, classes=classes, time_training=True, time_loading=True)

    if running_in_mpi():
        start_reduce = time.time()
        prd = prd.reduce()
        time_reduce = time.time() - start_reduce
    else:
        time_reduce = 0

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

        train_pos, train_neg = threshold_count(train, config.decision_boundary)
        test_pos, test_neg = threshold_count(test, config.decision_boundary)

        rmse = np.sqrt( sum(pow(test_y - out, 2)) / test_y.size )

    comm.barrier()

    if comm.rank == 0:
        return TrainingResult(fp=fp, fn=fn, rmse=rmse,
            time_load=time_load, time_train=time_train, time_reduce=time_reduce, time_test=time_test,
            negative_train_samples=train_neg, positive_train_samples=train_pos,
            negative_test_samples=test_neg, positive_test_samples=test_pos)
    else:
        return null_training_result()

@profile('fit_prof')
def fit(prd, ds, classes=None, online=False, time_training=False, time_loading=False):
    # Reset the predictor
    prd.fit(*ds.get_cycle(0))

    if online:
        train_time = 0
        load_time = 0
        for i in range(ds.num_cycles()):
            start_load = time.time()
            X, y = ds.get_cycle(i)
            load_time += time.time() - start_load

            if i > 0 and i % 1000 == 0:
                root_info('training cycle {}/{}', i + 1, ds.num_cycles())
                root_info('current stats:')
                root_info('  I/O time: {} s', load_time)
                root_info('  train time: {} s', train_time)

            start_train = time.time()
            prd.partial_fit(X, y, classes=ds.classes())
            train_time += time.time() - start_train

    else:
        start_load = time.time()
        X, y = ds.points()
        load_time = time.time() - start_load

        root_info('loaded dataset in {} s', load_time)

        start_train = time.time()
        prd.fit(X, y)
        train_time = time.time() - start_train

    results = [prd]
    if time_training:
        results.append(train_time)
    if time_loading:
        results.append(load_time)
    if len(results) == 0:
        results = results[0]
    return results
