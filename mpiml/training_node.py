#! /usr/bin/env python

import argparse
from contextlib import contextmanager
import itertools
from mpi4py import MPI
import numpy as np
import random
from forest import RandomForestRegressor, MondrianForestRegressor
from nbmpi import GaussianNB
from datasets import get_bubbleshock, shuffle_data, discretize
from utils import *
import sys

from config import models

comm = MPI.COMM_WORLD

def unzip(xys):
    xs = [x for (x, _) in xys]
    ys = [y for (_, y) in xys]
    return xs, ys

def probable(p):
    return random.uniform(0, 1) <= p

def get_sample(X, y, criterion, pkeep_positive, pkeep_negative):
    should_keep = lambda x, y: probable(pkeep_positive) \
                    if eval(criterion) \
                    else probable(pkeep_negative)

    # Sample from the training data, keep samples which satisfy should_keep
    sampled_X = []
    sampled_y = []
    for i in range(X.shape[0]):
        if should_keep(X[i:i+1], y[i:i+1]):
            sampled_X.extend(X[i:i+1,:])
            sampled_y.extend(y[i:i+1])
    return np.array(sampled_X), np.array(sampled_y)

def train_at_root(clf, X, y, root=0, comm=MPI.COMM_WORLD, criterion='True', pcast_positive=1,
                  pcast_negative=0, pkeep_positive=1, pkeep_negative=1, online=False, online_pool=1,
                  classes=None, **kwargs):
    classes = np.unique(y) if classes is None else classes
    root_debug('training with parameters:\n'
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

    if comm.rank == root:
        p_pos, p_neg = pkeep_positive, pkeep_negative
    else:
        p_pos, p_neg = pcast_positive, pcast_negative

    sampled_X, sampled_y = get_sample(X, y, criterion, p_pos, p_neg)
    all_X = comm.gather(sampled_X, root=root)
    all_y = comm.gather(sampled_y, root=root)
    if comm.rank == root:
        return fit(clf, np.concatenate(all_X), np.concatenate(all_y),
                    online=online, online_pool=online_pool, classes=classes)

def train_on_all(clf, X, y, root=0, comm=MPI.COMM_WORLD, criterion='True', pcast_positive=1,
                 pcast_negative=0, pkeep_positive=1, pkeep_negative=1, online=False, online_pool=1,
                 classes=None, **kwargs):
    classes = np.unique(y) if classes is None else classes
    root_debug('training with parameters:\n'
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

    cast_X, cast_y = get_sample(X, y, criterion, pcast_positive, pcast_negative)
    keep_X, keep_y = get_sample(X, y, criterion, pkeep_positive, pkeep_negative)

    all_X = comm.allgather(cast_X)
    all_X.append(keep_X)

    all_y = comm.allgather(cast_y)
    all_y.append(keep_y)

    clf = fit(clf, np.concatenate(all_X), np.concatenate(all_y),
        online=online, online_pool=online_pool, classes=classes)
    return clf.reduce()

def train(X, y, clf, recipients='all', **kwargs):
    kwargs.update(model=model)

    if not running_in_mpi():
        clf.fit(X, y)
        return clf

    if recipients == "all":
        return train_on_all(clf, X, y, **kwargs)
    elif recipients == "root":
        return train_at_root(clf, X, y, **kwargs)
    else:
        root_info('Invalid value "{}" for recipients: expected "all" or "root"', recipients)
        sys.exit(1)

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test a classifier using the designated training node strategy')
    parser.add_argument('data_dir', type=str, help='path to bubble shock data')
    parser.add_argument('--model', type=str, default='rf', help='model to test (default rf)')
    parser.add_argument('--recipients', type=str, default='root',
        help='specify whether to broadcast to the root node or all nodes')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('--online-pool', type=int, default=1,
        help='number of samples to collect in online training before a partial_fit is applied')
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
    parser.add_argument('--num-runs', type=int, default=10, help='k for k-fold validation')
    parser.add_argument('--profile', action='store_true', help='enable performance profiling')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    # Special wrapper flags that specify & overwrite some of the above values
    parser.add_argument('--super-forest', action='store_true',
        help='wrapper to perform simple super forest model (only passes completely trained trees, no data). Requires a model to be specified.')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    toggle_verbose(args.verbose)
    toggle_profiling(args.profile)

    if args.model in models:
        model = models[args.model]()
        root_debug('using model {}', model.__class__.__name__)
    else:
        root_info('unknown model "{}": valid models are {}', args.model, models.keys())
        sys.exit(1)

    pcast_positive = parse_range(args.pcast_positive)
    pcast_negative = parse_range(args.pcast_negative)
    pkeep_positive = parse_range(args.pkeep_positive)
    pkeep_negative = parse_range(args.pkeep_negative)

    # Super forest training; when all nodes train on only their own data
    if args.super_forest:
        pcast_positive = pcast_negative = [0.]
        pkeep_positive = pkeep_negative = [1.]
        args.recipients = 'all'

    random.seed(args.seed)

    data, target = get_bubbleshock(args.data_dir)
    shuffle_data(data, target)
    target = discretize(target)
    num_pos, num_neg = num_classes(target)

    nrows = 0
    total_rows = len(pcast_positive) * len(pcast_negative) * len(pkeep_positive) * len(pkeep_negative)
    if comm.rank == 0:
        sf_fmt   = ''
        online_fmt = ''
        if args.super_forest:
            sf_fmt = ', super forest'
        if args.online:
            online_fmt = ', pooling: ' + str(args.online_pool)
        root_info('Training with model: {}, recipients: {}, method: {}{}{}'.format(
            args.model, args.recipients, 'online' if args.online else 'batch', online_fmt, sf_fmt))
        root_info('model,pcp,pcn,pkp,pkn,npos,nneg,fp,fn,t_train,t_test')
    for pcp, pcn, pkp, pkn in itertools.product(pcast_positive, pcast_negative, pkeep_positive, pkeep_negative):
        res = train_and_test_k_fold(
            data, target, model, k=args.num_runs, model=model, pkeep_positive=pkp,
            pkeep_negative=pkn, pcast_positive=pcp, pcast_negative=pcn, criterion=args.criterion,
            recipients=args.recipients, online=args.online, online_pool=args.online_pool)
        if comm.rank == 0:
            print('{},{pcp},{pcn},{pkp},{pkn},{num_pos},{num_neg},{fp},{fn},{train_time},{test_time}'
                .format(args.model, fp=res['fp'], fn=res['fn'], train_time=res['time_train'],
                        test_time=res['time_test'], **locals()))
        nrows += 1
        if nrows % 10 == 0:
            root_info('{}/{} trials complete', nrows, total_rows)

        comm.barrier()
