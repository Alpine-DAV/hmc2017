from __future__ import division

from mpi4py import MPI
import numpy as np
import os
from sklearn import datasets

__all__ = [ "accuracy"
          , "shuffle_data"
          , "get_k_fold_data"
          , "get_mpi_task_data"
          , "running_in_mpi"
          , "prepare_dataset"
          ]

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

# Determine if we are running as an MPI process
def running_in_mpi():
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ

# Load the requested example dataset and randomly reorder it so that it is not grouped by class
def prepare_dataset(dataset='iris'):
    iris = getattr(datasets, 'load_{}'.format(dataset))()
    X = iris.data
    y = iris.target
    shuffle_data(X, y)
    return X, y
