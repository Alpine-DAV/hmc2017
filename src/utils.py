from __future__ import division

from mpi4py import MPI
import numpy as np
import os
from sklearn import datasets

__all__ = [ "info"
          , "accuracy"
          , "shuffle_data"
          , "get_k_fold_data"
          , "get_mpi_task_data"
          , "get_testing_data"
          , "running_in_mpi"
          , "prepare_dataset"
          , "train_and_test_k_fold"
          ]

def info(fmt, *args, **kwargs):
    if 'comm' in kwargs:
        comm = kwargs['comm']
        del kwargs['comm']
    else:
        comm = MPI.COMM_WORLD
    print(('rank {}: ' + fmt).format(comm.rank, *args, **kwargs))

# Compute the accuracy of a set of predictions against the ground truth values.
def accuracy(actual, predicted):
    return np.sum(predicted == actual) / actual.shape[0]

# Reorder a dataset to remove patterns between adjacent samples. The random state is seeded with a
# constant before-hand, so the results will not vary between runs.
def shuffle_data(X, y, seed=0):
    np.random.seed(0)
    seed = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(seed)
    np.random.shuffle(y)

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

# Load the requested example dataset and randomly reorder it so that it is not grouped by class
def prepare_dataset(dataset):
    iris = getattr(datasets, 'load_{}'.format(dataset))()
    X = iris.data
    y = iris.target
    shuffle_data(X, y)
    return X, y

# Train and test a model using k-fold cross validation (default is 10-fold). Return the average
# accuracy over all k runs. If running in MPI, only root (rank 0) has a meaningful return value.
# `train` should be a function which, given a feature vector and a class vector, returns a trained
# instance of the desired model.
def train_and_test_k_fold(X, y, train, verbose=False, use_mpi=False, use_online=False, k=10, comm=MPI.COMM_WORLD):
    acc_accum = 0
    runs = 0
    classes = np.unique(y)
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y):
        if use_mpi:
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        if verbose:
            info('training with {} samples'.format(comm.rank, train_X.shape[0]))

        clf = train(train_X, train_y, classes=classes, online=use_online, mpi=use_mpi)
        if comm.rank == 0:
            # Only root has the final model, so only root does the predicting
            acc = clf.score(test_X, test_y)

            runs += 1
            acc_accum += acc
            if verbose:
                print('run {}: accuracy={}'.format(runs, acc))
                print('final model: {}'.format(clf))

        comm.barrier()

    if comm.rank == 0:
        return acc_accum / runs
