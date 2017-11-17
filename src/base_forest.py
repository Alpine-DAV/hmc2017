#! /usr/bin/env python

import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD

__all__ = [ "parse_args"
          , "forest_train"
          ]

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a decision tree classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    return parser.parse_args()

# Pass in MF or RF to be trained as clf
def forest_train(X, y, clf, **kwargs):
    clf.fit(X, y)
    all_estimators = comm.gather(clf.estimators_, root=0)
    if comm.rank == 0:
        super_forest = []
        for trees in all_estimators:
            super_forest.extend(trees)
        clf.estimators_ = super_forest
    return clf