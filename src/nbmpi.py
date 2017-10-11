#! /usr/bin/env python
from __future__ import division

import argparse
import os
import sys

from mpi4py import MPI

import numpy as np

from sklearn import datasets
from sklearn.naive_bayes import BaseNB
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted

comm = MPI.COMM_WORLD

# Gaussian naive Bayes classifier. This implementation is heavily based off of sckit learn's
# version. However, we provide an additional method, reduce, for use with MPI.
class GaussianNB(BaseNB):
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        return self._partial_fit(X, y, np.unique(y), _refit=True,
                                 sample_weight=sample_weight)

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight / n_new)
            new_var = np.average((X - new_mu) ** 2, axis=0,
                                 weights=sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_past / float(n_new * n_total)) *
                     (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self._partial_fit(X, y, classes, _refit=False,
                                 sample_weight=sample_weight)

    # When running in MPI, coordinate with other tasks to combine each task's local model into a
    # global model. The global model is returned. Each process's local model is unchanged. Note: the
    # return value is only meaningful for root (rank == 0)
    def reduce(self, verbose=False):
        # Variables which will collect the global results
        n = np.copy(self.class_count_)
        mu = np.copy(self.theta_)
        var = np.copy(self.sigma_)

        # tree-reduce the global count, mean, and variance to the root process
        node_diff = 1
        while(node_diff < comm.size):
            if(comm.rank % (node_diff*2) == 0 and comm.rank+node_diff < comm.size):
                n_1 = np.copy(n)
                mu_1 = np.copy(mu)
                var_1 = np.copy(var)

                n_2 = comm.recv(source=comm.rank+node_diff, tag=1)
                mu_2 = comm.recv(source=comm.rank+node_diff, tag=2)
                var_2 = comm.recv(source=comm.rank+node_diff, tag=3)

                n = n_1 + n_2
                for i in range(len(self.classes_)):
                    mu[i] = (mu_1[i]*n_1[i] + mu_2[i]*n_2[i]) / n[i]

                    var[i] = var_1[i]*n_1[i] + var_2[i]*n_2[i] + \
                             n_1[i]*pow((mu_1[i]-mu[i]),2) + \
                             n_2[i]*pow((mu_2[i]-mu[i]),2)
                    var[i] /= n[i]

            if(comm.rank % (node_diff*2) == node_diff):
                comm.send(n, dest=comm.rank-node_diff, tag=1)
                comm.send(mu, dest=comm.rank-node_diff, tag=2)
                comm.send(var, dest=comm.rank-node_diff, tag=3)

            node_diff *= 2

        clf = GaussianNB()
        clf.class_count_ = n
        clf.theta_ = mu
        clf.sigma_ = var
        clf.classes_ = self.classes_ # N.B. assumes classes_ is the same for all local models
        if self.priors is None:
            clf.class_prior_ = clf.class_count_ / clf.class_count_.sum()
        else:
            clf.class_prior_ = np.asarray(self.priors)
        return clf

    def __repr__(self):
        return 'GaussianNB(\n\tn={},\n\tmean={},\n\tvariance={}\n)'.format(
            self.class_count_, self.theta_, self.sigma_)

    def _partial_fit(self, X, y, classes=None, _refit=False,
                     sample_weight=None):
        X, y = check_X_y(X, y)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        epsilon = 1e-9 * np.var(X, axis=0).max()

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.sigma_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            n_classes = len(self.classes_)
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of'
                                     ' classes.')
                # Check that the sum is 1
                if priors.sum() != 1.0:
                    raise ValueError('The sum of the priors should be 1.')
                # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_),
                                             dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.sigma_[:, :] -= epsilon

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = np.in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the "
                             "initial classes %s" %
                             (unique_y[~unique_y_in_classes], classes))

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
                X_i, sw_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += epsilon

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

# Compute the accuracy of a set of predictions against the ground truth values.
def accuracy(actual, predicted):
    return np.sum(predicted == actual) / actual.shape[0]

# Create a new classifier and train it using the given training data. If online=True, the data will
# be passed to the classifier one sample at a time, updating the model each time. Otherwise, the
# classifier is trained on the whole dataset in batch fashion. If mpi=True, then each task trains a
# local model, and the local models are combined into a global model at the end. In this mode, the
# result is only meaningful if comm.rank == 0.
def train(X, y, classes, online=False, mpi=False):
    clf = GaussianNB()
    if online:
        classes = np.unique(y)
        for i in range(X.shape[0]):
            clf.partial_fit(X[i:i+1], y[i:i+1], classes=classes)
    else:
        clf.fit(X, y)
    if mpi:
        clf = clf.reduce()
    return clf

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
def get_mpi_task_data(X, y):
    samps_per_task = X.shape[0] // comm.size

    min_bound = samps_per_task*comm.rank
    if comm.rank == comm.size - 1:
        max_bound = X.shape[0]
    else:
        max_bound = min_bound + samps_per_task

    return (X[min_bound:max_bound],
            y[min_bound:max_bound])

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a naive Bayes classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--online', action='store_true', help='train in online mode')

    return parser.parse_args()

# Load the iris dataset and randomly reorder it so that it is not grouped by class
def prepare_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    shuffle_data(X, y)
    return X, y

# Train and test the model using k-fold cross validation (default is 10-fold). Return the average
# accuracy over all k runs. If running in MPI, only root (rank 0) has a meaningful return value.
def train_and_test_k_fold(X, y, verbose=False, use_mpi=False, use_online=False, k=10):
    acc_accum = 0
    runs = 0
    classes = np.unique(y)
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y):
        if use_mpi:
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        if verbose:
            print('process {} with {} samples'.format(comm.rank, train_X.shape[0]))

        clf = train(train_X, train_y, classes, online=use_online, mpi=use_mpi)
        if comm.rank == 0:
            # Only root has the final model, so only root does the predicting
            prd = clf.predict(test_X)

            runs += 1
            acc = accuracy(test_y, prd)
            acc_accum += acc
            if verbose:
                print('run {}: accuracy={}'.format(runs, acc))
                print('final model: {}'.format(clf))

        comm.barrier()

    if comm.rank == 0:
        return acc_accum / runs

if __name__ == '__main__':

    args = parse_args()
    verbose = args.verbose
    use_online = args.online
    use_mpi = 'MPICH_INTERFACE_HOSTNAME' in os.environ

    if use_mpi and comm.rank == 0:
        print('will train using MPI')
    if use_online and comm.rank == 0:
        print('will train in online mode')

    X, y = prepare_dataset()
    acc = train_and_test_k_fold(X, y, verbose=verbose, use_online=use_online, use_mpi=use_mpi)

    if comm.rank == 0:
        print('average accuracy: {}'.format(acc))
