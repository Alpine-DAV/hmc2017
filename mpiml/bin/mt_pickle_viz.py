#! /usr/bin/env python2

# Visualization tool for results of mt_pickle_bench.py

import argparse
import numpy as np
import os

from mpiml.plot import *

# Column names
PICKLE_FLAG = 'pickle'
COMPRESSION = 'compression'
TOTAL_TIME = 't_total'
PREPROCESS_TIME = 't_preprocess'
TRANSMISSION_TIME = 't_transmit'
POSTPROCESS_TIME = 't_postprocess'
BYTES = 'bytes'

unit_to_factor = {
    'MB': 1e-6
}

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

    cur_key = srows[0][key]
    split_indices = []
    for i, row in enumerate(srows):
        if row[key] != cur_key:
            split_indices.append(i)
            cur_key = row[key]

    return np.split(srows, split_indices)

class Strategy(object):

    def __init__(self, data):
        self.t_total_ = data[TOTAL_TIME]
        self.t_preprocess_ = data[PREPROCESS_TIME]
        self.t_transmit_ = data[TRANSMISSION_TIME]
        self.t_postprocess_ = data[POSTPROCESS_TIME]
        self.bytes_ = data[BYTES]
        self.compressions_ = data[COMPRESSION]
        self.pickle_ = True if data[0][PICKLE_FLAG] == 1 else 0

    def plot_time_against_compression(self):
        plot(self.compressions_, self.t_preprocess_, label='Preprocessing Time')
        plot(self.compressions_, self.t_transmit_, label='Transmission Time')
        plot(self.compressions_, self.t_postprocess_, label='Postprocessing Time')
        plot(self.compressions_, self.t_total_, label='End-to-end Time')

    def plot_space_against_compression(self, unit='MB'):
        factor = unit_to_factor[unit]
        plot(self.compressions_, self.bytes_*factor, label='Amount of Data Sent ({})'.format(unit))

    def plot_transmission_time_against_space(self, space_unit='MB'):
        factor = unit_to_factor[space_unit]
        plot(self.bytes_*factor, self.t_transmit_, label='Transmission Time')

    def bar_times(self, index, nbars, compression):
        bar(index, nbars, [self.t_preprocess_[compression], self.t_transmit_[compression],
            self.t_postprocess_[compression], self.t_total_[compression]],
            label= 'Pickle' if self.pickle_ else 'Native')
        xticks(nbars, ["Preprocess", "Transmit", "Postprocess", "Total"])

def round_to_nearest(n, resolution):
    return (int(n / resolution) + 1) * resolution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize data generated by mt_pickle_bench.py')
    parser.add_argument(
        '--input', type=str, help='CSV file generated by mt_pickle_bench.py (default stdin)', default=None)
    parser.add_argument(
        '--output', type=str, default='.', help='output directory for graphs (default .)')
    args = parser.parse_args()

    csv = np.genfromtxt(args.input if args.input is not None else sys.stdin,
                        names=True, delimiter=',', converters={PICKLE_FLAG: bool_to_int})

    max_time = round_to_nearest(max(csv[TOTAL_TIME]), 5)
    max_mb = round_to_nearest(max(csv[BYTES]) * 1e-6, 10)

    strategies = [Strategy(g) for g in group_by(PICKLE_FLAG, csv)]
    for s in strategies:
        name = 'Pickle' if s.pickle_ else 'Native'

        clf()
        s.plot_time_against_compression()
        legend()
        title('Temporal Performance of {} Strategy'.format(name))
        xlabel('Compression')
        ylabel('Time (s)')
        ylim(0, max_time)
        grid()
        savefig(os.path.join(args.output, 'mt_pickle_time_{}.png'.format(name)), format='png')

        clf()
        s.plot_space_against_compression(unit='MB')
        legend()
        title('Spatial Performance of {} Strategy'.format(name))
        xlabel('Compression')
        ylabel('Data Sent (MB)')
        ylim(0, max_mb)
        grid()
        savefig(os.path.join(args.output, 'mt_pickle_space_{}.png'.format(name)), format='png')

        clf()
        s.plot_transmission_time_against_space(space_unit='MB')
        legend()
        suptitle('Cost of Transmission by Data Size')
        title('({} Strategy)'.format(name))
        xlabel('Data Size (MB)')
        ylabel('Transmission Time (s)')
        ylim(0, round_to_nearest(max(csv[TRANSMISSION_TIME]), 0.1))
        xlim(0, max_mb)
        grid()
        savefig(os.path.join(args.output, 'mt_pickle_space_time_{}.png'.format(name)), format='png')

    clf()
    s0, s1 = strategies[0], strategies[1]
    pickle = s0 if s0.pickle_ else s1
    native = s1 if s0.pickle_ else s0

    pickle.bar_times(0, 2, 0)
    native.bar_times(1, 2, 0)
    legend()
    title('Comparative Performance of Encoding Strategies')
    ylabel('Time (s)')
    ylim(0, round_to_nearest(pickle.t_total_[0], 5))
    grid()
    savefig(os.path.join(args.output, 'mt_comparative_time.png'), format='png')
