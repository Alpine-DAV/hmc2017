#! /usr/bin/env python

import argparse
from contextlib import contextmanager
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from utils import *
import sys

comm = MPI.COMM_WORLD

def unzip(xys):
    xs = [x for (x, _) in xys]
    ys = [y for (_, y) in xys]
    return xs, ys

def probable(p):
    return random.uniform(0, 1) <= p

def train_at_root(clf, X, y, root=0, comm=MPI.COMM_WORLD, verbose=False, criterion='True',
                  pcast_positive=1, pcast_negative=0, pkeep_positive=1, pkeep_negative=1, **kwargs):
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

        # Get the broadcast results from other processes
        for proc in range(comm.size):
            if proc == comm.rank:
                continue
            new_X, new_y = unzip(comm.recv(source=proc))
            sampled_X.extend(new_X)
            sampled_y.extend(new_y)

        clf.fit(np.vstack(sampled_X), np.concatenate(sampled_y))
        return clf
    else:
        should_bcast = lambda x, y: probable(pcast_positive) \
                            if eval(criterion) \
                            else probable(pcast_negative)

        # Sample from the training data, selecting samples to broadcast
        bcast = []
        for i in range(X.shape[0]):
            samp = X[i:i+1]
            label = y[i:i+1]
            if should_bcast(samp, label):
                bcast += [(samp, label)]
        comm.send(bcast, dest=root)

def train(X, y, model=GaussianNB, mpi=False, **kwargs):
    kwargs.update(model=model, mpi=mpi)

    clf = model()

    if not mpi:
        clf.fit(X, y)
        return clf

    return train_at_root(clf, X, y, **kwargs)

models = {
    'nb': GaussianNB,
    'rf': RandomForestClassifier
}
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the designated training node strategy')
    parser.add_argument('--dataset', metavar="NAME", type=str, default='iris',
        help='scikit-learn example dataset with which to evaluate the model (default iris)')
    parser.add_argument('--model', type=str, default='nb', help='model to test')
    parser.add_argument('--criterion', metavar='EXPRESSION', type=str, default='y == 0',
        help='Python expression evaluated to determine whether a sample should be broadcast. If '
             'x_0 is a feature vector and y_0 a class label, the sample (x_0, y_0) is said to '
             'satisfy criterion if (lambda x, y: EXPRESSION)(x_0, y_0) == True')
    parser.add_argument('--pcast-positive', metavar='P', type=float, default=1,
        help='probability of a sample on a non-training node which satisfies criterion being sent '
             'to the training node')
    parser.add_argument('--pcast-negative', metavar='P', type=float, default=0,
        help='probability of a sample on a non-training node which does not satisfy criterion '
             'being sent to the training node')
    parser.add_argument('--pkeep-positive', metavar='P', type=float, default=1,
        help='probability of a sample on the training node which satisfies criterion being used '
             'for training')
    parser.add_argument('--pkeep-negative', metavar='P', type=float, default=1,
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

    if use_mpi:
        root_info('will train using MPI')
    else:
        root_info('will train serially')

    random.seed(args.seed)

    data, target = prepare_dataset(args.dataset)
    acc = train_and_test_k_fold(
        data, target, train, verbose=verbose, use_mpi=use_mpi, mpi=use_mpi, model=model,
            pkeep_positive=args.pkeep_positive, pkeep_negative=args.pkeep_negative,
            pcast_positive=args.pcast_positive, pcast_negative=args.pcast_negative,
            criterion=args.criterion)

    root_info('average accuracy: {}'.format(acc))
