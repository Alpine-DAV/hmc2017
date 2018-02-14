from __future__ import division

import numpy as np
import sklearn.base as sk
import time

import config
from datasets import concatenate, threshold_count, discretize
from utils import *
from config import TOTAL_CYCLES as total_cycles

__all__ = [ "get_k_fold_data"
          , "train_and_test_k_fold"
          , "train_and_test_once"
          , "fit"
          ]

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
# validation experiment. By default, k=10. If running on a subset of the data/different split
# points with {training, testing}_split, returns datasets of sizes according to number of cycles
# according to those splits
def get_k_fold_data(ds, k=10, train_test_split=None):
    if train_test_split != None:
        splits = ds.split(total_cycles)
        train_split = train_test_split['train_split']
        test_split = train_test_split['test_split']
        for i in range(k):
            tr_start = int(total_cycles*i/k)
            tr_end = tr_start+train_split
            tr = wrapped_concatenate(splits, tr_start, tr_end)
            te = wrapped_concatenate(splits, tr_end, tr_end+test_split)            
            yield (tr, te)
    else:
        splits = ds.split(k)
        for i in range(k):
            yield (concatenate(splits[j] for j in range(k) if j != i), splits[i])

# Concatenate datasets at the beginning of the split if the k-folding pattern results in train or test
# set containing the end and beginning StrictDataSets
def wrapped_concatenate(splits, start, end):
    ds = concatenate(splits[j] for j in range(start, min(end, total_cycles)))
    if end > total_cycles:
        ds = concatenate([ds, (splits[j] for j in range(0, end%total_cycles))])
    return ds

# Get a subset of a dataset for the current task. If each task in an MPI communicator calls this
# function, then every sample in the dataset will be distributed to exactly one task.
def get_mpi_task_data(ds, comm=config.comm):
    return ds.split(comm.size)[comm.rank]

class TrainingResult(object):

    def __init__(self, time_train, time_reduce, time_test, fp, fn,
                 positive_train_samples, negative_train_samples,
                 positive_test_samples, negative_test_samples,
                 rmse, runs=1):

        self.runs = runs
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
""".format(time_train=self.time_train, time_reduce=self.time_reduce, time_test=self.time_test,
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
    return TrainingResult(*([None]*10), runs=0)

# Train and test a model using k-fold cross validation (default is 10-fold).
@profile('train_and_test_k_fold_prof')
def train_and_test_k_fold(ds, prd, k=10, comm=config.comm, online=False, classes=None, train_test_split=None):

    train_and_test = lambda tr, te: train_and_test_once(
        tr, te, prd, comm=comm, online=online, classes=classes)
    
    if k <= 0:
        raise ValueError("k must be positive")
    """ TODO: fairly certain this is the same
    elif k == 1:
        splits, train, test = None, None, None
        
        if train_split != None:
            splits = ds.split(config.TOTAL_CYCLES)
            train = concatenate(splits[j] for j in range(train_split))
            test = concatenate(splits[j] for j in range(train_split, train_split+test_split))    
        else:   
            splits = ds.split(10)
            train = concatenate(splits[j] for j in range(9))
            test = splits[9]

        if running_in_mpi():
            train = get_mpi_task_data(train)

        return train_and_test(train, test)
    """
    r = null_training_result()
    for train, test in get_k_fold_data(ds, k=k, train_test_split=train_test_split):
        if running_in_mpi():
            train = get_mpi_task_data(train)
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

    start_train = time.time()
    fit(prd, train, online=online, classes=classes)
    time_train = time.time() - start_train

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

        train_pos, train_neg = threshold_count(train, 1e-6)
        test_pos, test_neg = threshold_count(test, 1e-6)

        rmse = np.sqrt( sum(pow(test_y - out, 2)) / test_y.size )
    comm.barrier()

    if comm.rank == 0:
        return TrainingResult(fp=fp, fn=fn, rmse=rmse,
            time_train=time_train, time_reduce=time_reduce, time_test=time_test,
            negative_train_samples=train_neg, positive_train_samples=train_pos,
            negative_test_samples=test_neg, positive_test_samples=test_pos)
    else:
        return null_training_result()

def fit(prd, ds, classes=None, online=False, **kwargs):
    if online:
        first = True
        for X, y in ds.cycles():
            if first:
                prd.fit(X, y)
                first = False
            else:
                prd.partial_fit(X, y, classes=ds.classes())
    else:
        prd.fit(*ds.points())
    return prd
