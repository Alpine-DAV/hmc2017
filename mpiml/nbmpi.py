#! /usr/bin/env python
from __future__ import division

import argparse
import os
import sys

from mpi4py import MPI

import numpy as np

from sklearn import datasets
import sklearn.naive_bayes as sk
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted

import config
from config import comm
from datasets import get_bubbleshock, shuffle_data, discretize
from utils import *

__all__ = ["GaussianNB"
          ]

# Gaussian naive Bayes classifier. This implementation is heavily based off of sckit learn's
# version. However, we provide an additional method, reduce, for use with MPI.
class GaussianNB(sk.GaussianNB):
    def __init__(self, priors=None):
        super(GaussianNB, self).__init__(priors=priors)

    # When running in MPI, coordinate with other tasks to combine each task's local model into a
    # global model. The global model is returned. Each process's local model is unchanged. Note: the
    # return value is only meaningful for root (rank == 0)
    def reduce(self):
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
                    if n[i] != 0:
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

    def fit(self, X, y):
        y = discretize(y)
        sk.GaussianNB.fit(self, X, y)

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            classes = discretize(classes)
        y = discretize(y)
        sk.GaussianNB.partial_fit(self, X, y, classes=classes)

    def __repr__(self):
        return 'GaussianNB(\n\tn={},\n\tmean={},\n\tvariance={}\n)'.format(
            self.class_count_, self.theta_, self.sigma_)

config.register_model('nb', GaussianNB)
