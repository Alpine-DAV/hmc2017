#! /usr/bin/env python
from __future__ import division

import argparse
from mpi4py import MPI
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree._tree import Tree

from utils import *

comm = MPI.COMM_WORLD

def info(fmt, *args, **kwargs):
    print(('rank {}: ' + fmt).format(comm.rank, *args, **kwargs))

def train(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a decision tree classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    verbose = args.verbose
    use_mpi = running_in_mpi()

    runs = 0
    acc_accum = 0

    X, y = prepare_dataset('iris')
    if use_mpi:
        train_X, train_y = get_mpi_task_data(X, y, comm, True)
        rf = train(train_X, train_y)
        all_estimators = comm.gather(rf.estimators_, root=0)
        if comm.rank == 0:
            super_forest = []
            for forest in all_estimators:
                super_forest.extend(forest)
            rf.estimators_ = super_forest
            test_X, test_y = get_testing_data(X, y, comm)
            acc = rf.score(test_X, test_y)
            if verbose:
                info('accuracy: {}', acc)