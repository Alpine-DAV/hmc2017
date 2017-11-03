#! /usr/bin/env python

import argparse
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor

from datasets import prepare_dataset
from utils import *

comm = MPI.COMM_WORLD

#trains on segment of data, then places final model in 0th process
def train(X, y, **kwargs):
    rf = RandomForestRegressor()
    rf.fit(X, y)
    all_estimators = comm.gather(rf.estimators_, root=0)
    if comm.rank == 0:
        super_forest = []
        for forest in all_estimators:
            super_forest.extend(forest)
        rf.estimators_ = super_forest
    return rf

# Compose all decision trees into one super forest of decision trees
def reduce(rf):
    all_estimators = comm.gather(rf.estimators_, root=0)
    if comm.rank == 0:
        super_forest = []
        for forest in all_estimators:
            super_forest.extend(forest)
        rf.estimators_ = super_forest

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

    if verbose and comm.rank == 0:
        if use_mpi:
            info('training using MPI')
        else:
            info('training on one processor')

    data, target = prepare_dataset('iris')
    acc = train_and_test_k_fold(data, target, train, verbose=verbose)

    if comm.rank == 0:
        info('average accuracy: {}'.format(acc))
