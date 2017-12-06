#! /usr/bin/env python

from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier

from datasets import prepare_dataset
from utils import *
from forest_utils import *

comm = MPI.COMM_WORLD

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
    res = train_and_test_k_fold(
        data, target, forest_train, model=RandomForestClassifier(), verbose=verbose, use_mpi=use_mpi)

    root_info('### PERFORMANCE ###\n{}', prettify_train_and_test_k_fold_results(res))