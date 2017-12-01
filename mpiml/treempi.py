#! /usr/bin/env python
from __future__ import division

import argparse
from mpi4py import MPI
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from utils import *

comm = MPI.COMM_WORLD

def info(fmt, *args, **kwargs):
    print(('rank {}: ' + fmt).format(comm.rank, *args, **kwargs))

def send_tree(tree, dst=0):
    comm.send(tree, dst)

def recv_tree(src=1):
    tree = comm.recv(source=src)
    return tree

def train(X, y):
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    return tree

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
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y):
        tree = train(train_X, train_y)

        if use_mpi:
            if comm.rank == 1:
                send_tree(tree, dst=0)
            elif comm.rank == 0:
                tree = recv_tree(src=1)
                acc = tree.score(test_X, test_y)
                acc_accum += acc
                if verbose:
                    info('run {}: accuracy={}', runs, acc)

            comm.barrier()
        else:
            acc = tree.score(test_X, test_y)
            acc_accum += acc
            if verbose:
                info('run {}: accuracy={}', runs, acc)

        runs += 1

    if comm.rank == 0:
        info('average accuracy ({} runs): {}', runs, acc_accum / runs)
