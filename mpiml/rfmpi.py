#! /usr/bin/env python

import argparse
from sklearn.ensemble import RandomForestRegressor
from datasets import prepare_dataset

from utils import *
import config
from config import comm

def reduce(rf):
    all_estimators = comm.gather(rf.estimators_, root=0)
    if comm.rank == 0:
        super_forest = []
        for forest in all_estimators:
            super_forest.extend(forest)
        rf.estimators_ = super_forest

#trains on segment of data, then places final model in 0th process
def train(X, y, **kwargs):
    rf = RandomForestRegressor(n_estimators=config.NumTrees, n_jobs=config.parallelism, random_state=config.rand_seed)
    rf.fit(X, y)
    if running_in_mpi():
        reduce(rf)
    return rf

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a decision tree classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    use_mpi = running_in_mpi()

    toggle_verbose(args.verbose)

    runs = 0
    acc_accum = 0

    data, target = prepare_dataset('iris')
    result = train_and_test_k_fold(data, target, train)

    if comm.rank == 0:
        info('average accuracy: {}'.format(result))
