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
    if running_in_mpi():
        all_estimators = comm.gather(rf.estimators_, root=0)
        if comm.rank == 0:
            rf = all_estimators
    return rf

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a decision tree classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    return parser.parse_args()

def running_in_mpi():
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ

# Train and test a model using k-fold cross validation (default is 10-fold). Return the false 
# positives, false negatives, training time and testing time over all k runs (testing on root).
# If running in MPI, only root (rank 0) has a meaningful return value. `train` should be a
# function which, given a feature vector and a class vector, returns a trained instance of the
# desired model.
def train_and_test_k_fold(X, y, train, verbose=False, k=10, comm=MPI.COMM_WORLD, root=0, **kwargs):
    kwargs.update(verbose=verbose, k=k, comm=comm)
    fp_accum = fn_accum = 0
    test_accum = 0
    time_train = time_test = 0
    
    runs = 0
    classes = np.unique(y)
    for train_X, test_X, train_y, test_y in get_k_fold_data(X, y, k=k):
        if running_in_mpi():
            train_X, train_y = get_mpi_task_data(train_X, train_y)
        if verbose:
            info('training with {} samples'.format(train_X.shape[0]))

        start_train = time.time()
        clf = train(train_X, train_y, classes=classes, **kwargs)
        end_train = time.time()

        if comm.rank == root:
            # Only root has the final model, so only root does the predicting
            for forest in clf:
                start_test = time.time()
                prd = forest.predict(test_X)
                end_test = time.time()

                time_train += end_train-start_train
                time_test += end_test - start_test

                fp, fn = num_errors(test_y, prd)
                fp_accum += fp
                fn_accum += fn
                test_accum += len(test_y)
                runs += 1
                if verbose:
                    print('run {}: {} false positives, {} false negatives'.format(runs, fp, fn))
                    print('final model: {}'.format(clf))

        comm.barrier()

    if comm.rank == root:
        return fp_accum, fn_accum, test_accum, time_train, time_test
    else:
        # This allows us to tuple destructure the result of this function without checking whether
        # we are root
        return None, None, None, None, None


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
