#! /usr/bin/env python

from mpi4py import MPI
from skgarden.mondrian.ensemble import MondrianForestRegressor

from utils import *
from forest_utils import *
from datasets import get_bubbleshock, shuffle_data

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

    data, target = get_bubbleshock(args.data_dir)
    shuffle_data(data, target)
    acc = train_and_test_k_fold(
        data, target, forest_train, model=MondrianForestRegressor(), verbose=verbose, use_mpi=use_mpi)

    if comm.rank == 0:
        info('average accuracy: {}'.format(acc))
