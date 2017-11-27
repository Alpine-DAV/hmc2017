#! /usr/bin/env python

import argparse
from contextlib import contextmanager
import itertools
from mpi4py import MPI
import numpy as np
import random
from nbmpi import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from skgarden.mondrian.ensemble import MondrianForestRegressor
from datasets import get_bubbleshock, shuffle_data, discretize
from utils import *
import sys

comm = MPI.COMM_WORLD

def unzip(xys):
    xs = [x for (x, _) in xys]
    ys = [y for (_, y) in xys]
    return xs, ys

def probable(p):
    return random.uniform(0, 1) <= p

def get_bcast_sample(X,y,criterion,p__positive,p__negative):
    should_bcast = lambda x, y: probable(p__positive) \
                        if eval(criterion) \
                        else probable(p__negative)

    bcast = []
    for i in range(X.shape[0]):
        samp = X[i:i+1]
        label = y[i:i+1]
        if should_bcast(samp, label):
            bcast += [(samp, label)]
    return bcast

def get_local_sample(X, y, criterion, pkeep_positive, pkeep_negative):
    should_keep = lambda x, y: probable(pkeep_positive) \
                    if eval(criterion) \
                    else probable(pkeep_negative)

    # Sample from the training data, keep samples which satisfy should_keep
    sampled_X = []
    sampled_y = []
    for i in range(X.shape[0]):
        if should_keep(X[i:i+1], y[i:i+1]):
            sampled_X.append(X[i:i+1,:])
            sampled_y.append(y[i:i+1])
    return sampled_X, sampled_y

def train_at_root(clf, X, y, root=0, comm=MPI.COMM_WORLD, verbose=False, criterion='True',
                  pcast_positive=1, pcast_negative=0, pkeep_positive=1, pkeep_negative=1,
                  method='online', **kwargs):
    if verbose:
        root_info('training with parameters:\n'
                  '  criterion={}\n'
                  '  pcast_positive={}\n'
                  '  pcast_negative={}\n'
                  '  pkeep_positive={}\n'
                  '  pkeep_negative={}\n',
                  criterion,
                  pcast_positive,
                  pcast_negative,
                  pkeep_positive,
                  pkeep_negative)

    if (comm.rank == root):
        sampled_X, sampled_y = get_local_sample(X, y, criterion, pkeep_positive, pkeep_negative)

        # Get the broadcast results from other processes
        for proc in range(comm.size):
            if proc == comm.rank:
                continue
            new_X, new_y = unzip(comm.recv(source=proc))
            sampled_X.extend(new_X)
            sampled_y.extend(new_y)
        return train_with_method(clf, np.vstack(sampled_X), np.concatenate(sampled_y), method=method)

    else:
        bcast = get_bcast_sample(X, y, criterion, pcast_positive, pcast_negative)
        comm.send(bcast, dest=root)

def train_on_all(clf, X, y, root=0, comm=MPI.COMM_WORLD, verbose=False, criterion=')',
                  pcast_positive=1, pcast_negative=0, pkeep_positive=1, pkeep_negative=1,
                  method='online', **kwargs):
    if verbose:
        root_info('training with parameters:\n'
                  '  criterion={}\n'
                  '  pcast_positive={}\n'
                  '  pcast_negative={}\n'
                  '  pkeep_positive={}\n'
                  '  pkeep_negative={}\n',
                  criterion,
                  pcast_positive,
                  pcast_negative,
                  pkeep_positive,
                  pkeep_negative)

    # Sample from the training data, selecting samples to broadcast
    bcast0 = get_bcast_sample(X, y, criterion, pcast_positive, pcast_negative)
    bcast = [elem for l in comm.allgather(bcast0) for elem in l]

    # Get this node's samples
    sampled_X, sampled_y = get_local_sample(X, y, criterion, pkeep_positive, pkeep_negative)

    # Add in the new samples from other nodes
    new_X, new_y = unzip(bcast)
    sampled_X.extend(new_X)
    sampled_y.extend(new_y)

    train_with_method(clf, np.vstack(sampled_X), np.concatenate(sampled_y), method=method)
    if isinstance(clf, GaussianNB):
        return clf.reduce()
    elif isinstance(clf, RandomForestRegressor):
        all_estimators = comm.gather(clf.estimators_, root=0)

        if comm.rank == 0:
            super_forest = []
            for trees in all_estimators:
                super_forest.extend(trees)
            clf.estimators_ = super_forest
            return clf

def train(X, y, model=GaussianNB, mpi=False, **kwargs):
    kwargs.update(model=model, mpi=mpi)

    clf = model()

    if not mpi:
        clf.fit(X, y)
        return clf

    if "recipients" in kwargs and kwargs["recipients"] == "all":
        return train_on_all(clf, X, y, **kwargs)

    return train_at_root(clf, X, y, **kwargs)

# Users can specify a range of parameter values to investigate using the CLI. This function parses
# a parameter range expression and returns a list of values to try. Valid expression formats are:
#
# n         (just test a single value)
# l:h       (test values from l to h at intervals of default_step)
# l:s:h     (test values from l to h at intervals of s)
#
# The program will run one 10-fold cross validation trial for each combination of paramter values.
def parse_range(expr, default_step=0.01):
    parts = [float(part) for part in expr.split(':')]
    if len(parts) == 1:
        return [parts[0]]
    elif len(parts) == 2:
        return np.arange(parts[0], parts[1], default_step)
    elif len(parts) == 3:
        return np.arange(parts[0], parts[2], parts[1])
    else:
        root_info('error: unrecognized probability range {}', expr)
        sys.exit(1)

models = {
    'nb': GaussianNB,
    'rf': RandomForestRegressor,
    'mf': MondrianForestRegressor
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the designated training node strategy')
    parser.add_argument('data_dir', type=str, help='path to bubble shock data')
    parser.add_argument('--model', type=str, default='rf', help='model to test (default rf)')
    parser.add_argument('--recipients', type=str, default='root',
        help='specify whether to broadcast to the root node or all nodes')
    parser.add_argument('--method', type=str, default='online',
        help='specify whether to train in online mode, batch mode, or an otherwise specified mode. See train_with_method in utils')
    parser.add_argument('--criterion', metavar='EXPRESSION', type=str, default='y > 0.5',
        help='Python expression evaluated to determine whether a sample should be broadcast. If '
             'x_0 is a feature vector and y_0 a class label, the sample (x_0, y_0) is said to '
             'satisfy criterion if (lambda x, y: EXPRESSION)(x_0, y_0) == True')
    parser.add_argument('--pcast-positive', metavar='P', type=str, default='1',
        help='probability of a sample on a non-training node which satisfies criterion being sent '
             'to the training node')
    parser.add_argument('--pcast-negative', metavar='P', type=str, default='0',
        help='probability of a sample on a non-training node which does not satisfy criterion '
             'being sent to the training node')
    parser.add_argument('--pkeep-positive', metavar='P', type=str, default='1',
        help='probability of a sample on the training node which satisfies criterion being used '
             'for training')
    parser.add_argument('--pkeep-negative', metavar='P', type=str, default='1',
        help='probability of a sample on the training node which does not satisfy criterion being '
             'used for training')
    parser.add_argument('--seed', type=int, default=None,
        help='seed the ranom state (default is nondeterministic)')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    verbose = args.verbose
    use_mpi = running_in_mpi()
    if args.model in models:
        model = models[args.model]
        if verbose: root_info('using model {}', model)
    else:
        root_info('unknown model "{}": valid models are {}', args.model, models.keys())
        sys.exit(1)

    pcast_positive = parse_range(args.pcast_positive)
    pcast_negative = parse_range(args.pcast_negative)
    pkeep_positive = parse_range(args.pkeep_positive)
    pkeep_negative = parse_range(args.pkeep_negative)

    if use_mpi:
        root_info('will train using MPI')
    else:
        root_info('will train serially')

    random.seed(args.seed)

    data, target = get_bubbleshock(args.data_dir)
    shuffle_data(data, target)
    target = discretize(target)
    num_pos, num_neg = num_classes(target)

    nrows = 0
    total_rows = len(pcast_positive) * len(pcast_negative) * len(pkeep_positive) * len(pkeep_negative)
    if comm.rank == 0:
        print('model,pcp,pcn,pkp,pkn,npos,nneg,fp,fn,t_train,t_test')
    for pcp, pcn, pkp, pkn in itertools.product(pcast_positive, pcast_negative, pkeep_positive, pkeep_negative):
        fp, fn, total, train_time, test_time = train_and_test_k_fold(
            data, target, train, k=10, verbose=verbose, use_mpi=use_mpi, mpi=use_mpi, model=model,
            pkeep_positive=pkp, pkeep_negative=pkn, pcast_positive=pcp, pcast_negative=pcn,
            criterion=args.criterion, recipients=args.recipients, method=args.method)
        if comm.rank == 0:
            print('{},{pcp},{pcn},{pkp},{pkn},{num_pos},{num_neg},{fp},{fn},{train_time},{test_time}'
                .format(args.model, **locals()))
        nrows += 1
        if nrows % 10 == 0:
            root_info('{}/{} trials complete', nrows, total_rows)

        comm.barrier()
