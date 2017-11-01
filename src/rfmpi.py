#! /usr/bin/env python

import argparse
from mpi4py import MPI
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from utils import *

comm = MPI.COMM_WORLD

def train(X, y, **kwargs):
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

    if verbose and comm.rank == 0:
        if use_mpi:
            info('training using MPI')
        else:
            info('training on one processor')


    runs = 0
    acc_accum = 0

    data, target = prepare_dataset('iris')
    acc = train_and_test_k_fold(
        data, target, train, verbose=verbose, use_mpi=use_mpi)

    if comm.rank == 0:
        info('average accuracy: {}'.format(acc))
