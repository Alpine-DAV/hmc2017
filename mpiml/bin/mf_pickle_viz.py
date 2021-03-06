#! /usr/bin/env python2

# Visualization tool for results of mf_pickle_bench.py

import argparse
import numpy as np
import os

from mpiml.plot import *

# Column names
PICKLE_FLAG = 'pickle'
COMPRESSION = 'compression'
REDUCE_TIME = 't_reduce'
NUM_TASKS = 'n_tasks'

def bool_to_int(b):
    if b == 't':
        return 1
    elif b == 'f':
        return 0
    else:
        raise ValueError('Invalid boolean literal {}'.format(b))

def group_by(key, rows):
    srows = np.sort(rows, order=key)
    groups = []

    def get_key(row):
        return [row[k] for k in key]

    cur_key = get_key(srows[0])
    split_indices = []
    for i, row in enumerate(srows):
        if get_key(row) != cur_key:
            split_indices.append(i)
            cur_key = get_key(row)

    return np.split(srows, split_indices)

class Strategy(object):

    def __init__(self, data):
        self.t_reduce_ = data[REDUCE_TIME]
        self.compression_ = data[0][COMPRESSION]
        self.n_tasks_ = data[NUM_TASKS]
        self.pickle_ = True if data[0][PICKLE_FLAG] == 1 else 0

    def plot_time_against_n_tasks(self):
        plot(self.n_tasks_, self.t_reduce_, label='Reduction Time')

def round_to_nearest(n, resolution):
    return (int(n / resolution) + 1) * resolution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize data generated by mf_pickle_bench.py')
    parser.add_argument(
        '--input', type=str, help='CSV file generated by mt_pickle_bench.py (default stdin)', default=None)
    parser.add_argument(
        '--output', type=str, default='.', help='output directory for graphs (default .)')
    args = parser.parse_args()

    csv = np.genfromtxt(args.input if args.input is not None else sys.stdin,
                        names=True, delimiter=',', converters={PICKLE_FLAG: bool_to_int})

    max_time = round_to_nearest(max(csv[REDUCE_TIME]), 1)

    strategies = [Strategy(g) for g in group_by([PICKLE_FLAG, COMPRESSION], csv)]
    for s in strategies:
        name = 'Pickle' if s.pickle_ else 'Native'

        clf()
        s.plot_time_against_n_tasks()
        legend()
        suptitle('Strong Scaling of {} Strategy'.format(name))
        title('(compression = {})'.format(s.compression_))
        xlabel('Number of Tasks')
        ylabel('Time (s)')
        ylim(0, max_time)
        grid()
        savefig(os.path.join(args.output, 'mf_pickle_strong_scaling_{}_{}.png'.format(
            name, s.compression_)), format='png')
