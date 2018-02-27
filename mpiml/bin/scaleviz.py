#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
from numpy.lib.recfunctions import append_fields
from operator import attrgetter, methodcaller
import os
from scipy.optimize import curve_fit
import sys
from uncertainties import ufloat

from mpiml.models import get_model_id, get_cli_name
from mpiml.plot import *

# Column names
MODEL = 'model'
TASKS = 'tasks'
TIME_REDUCE = 'time_reduce'
TIME_TRAIN = 'time_train'
NODES = 'nodes'
DENSITY = 'density'
SPEEDUP = 'speedup'
SPEEDUP_M = 'speedup_m'
SPEEDUP_M_UNC = 'speedup_m_unc'
SPEEDUP_B = 'speedup_b'
SPEEDUP_B_UNC = 'speedup_b_unc'

def chi_sq(expected, observed):
    return sum((o - e)**2 / e for o, e in zip(observed, expected))

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

def scalarize(arr):
    val = arr[0]
    for v in arr:
        if v != val:
            raise ValueError('Scalarizing non-constant array {}'.format(arr))
    return val

class StrongScaling(object):
    def __init__(self, data):
        self.model_ = get_cli_name(scalarize(data[MODEL]))
        self.density_ = scalarize(data[DENSITY])
        self.nodes_ = data[NODES]
        self.train_times_ = data[TIME_TRAIN]
        self.reduce_times_ = data[TIME_REDUCE]
        self.times_ = self.train_times_ + self.reduce_times_
        self.t0_ = self.times_[0]
        self.speedup_ = self.t0_ / data[TIME_TRAIN]
        self.m_ = None
        self.b_ = None
        self.tasks_per_node_ = scalarize(data[TASKS] / self.nodes_)

        self._fit()

    def density(self):
        return self.density_

    def apply(self, n):
        return self.m_*n + self.b_

    def invert(self, t):
        return ((self.t0_ / t) - self.b_) / self.m_

    def apply_perfect(self, n):
        n0 = self.nodes_[0]
        return n / n0

    def model(self):
        return self.model_

    def tasks_per_node(self):
        return self.tasks_per_node_

    def save_plot(self, output):
        plot(self.nodes_, self.speedup_, '-o', color=(0, 0.1, 0.9), label='speedup')

        plot_continuous(self.apply, min(self.nodes_), max(self.nodes_),
            color=(0, 0.2, 0.6), linewidth=1, label='mx + b\nm={}\nb={}'.format(self.m_, self.b_))

        chi2 = chi_sq(self.speedup_, map(self.apply_perfect, self.nodes_))
        plot_continuous(self.apply_perfect, min(self.nodes_), max(self.nodes_),
            color=(0.6, 0, 0.2), label='perfect scaling\nchi2={}'.format(chi2))

        legend()
        xlabel('# of Nodes')
        ylabel('Speedup')
        suptitle('Strong Scaling'.format(self.density()))
        title('(model={}, density={}, tasks/node={})'.format(
            self.model_, self.density_, self.tasks_per_node_))
        grid()

        savefig(os.path.join(
            output, 'strong_scaling_{}_{}_{}.png'.format(
                self.model(), self.density(), self.tasks_per_node())),
            format='png')
        clf()

    def _fit(self):
        # fit function: speedup(n) = mn + b
        def f(n, m, b):
            return m*n + b

        (m, b), pcov = curve_fit(f, self.nodes_, self.speedup_, p0=(1, 1))
        m_err, b_err = np.sqrt(np.diag(pcov))

        self.m_ = ufloat(m, m_err)
        self.b_ = ufloat(b, b_err)

class WeakScaling(object):
    def __init__(self, time, strong_experiments):
        self.time_ = time
        self.strongs_ = strong_experiments
        self.densities_ = [e.density() for e in self.strongs_]
        self.nodes_ = [e.invert(self.time_) for e in self.strongs_]
        self.model_ = scalarize([e.model() for e in self.strongs_])
        self.tasks_per_node_ = scalarize([e.tasks_per_node() for e in self.strongs_])

        self._fit()

    def apply(self, d):
        return self.m_*d + self.b_

    def apply_perfect(self, d):
        return self.nodes_[0] * (d / self.densities_[0])

    def save_plot(self, output):
        plot(self.densities_, self.nodes_, color=(0, 0.1, 0.9), zorder=4, label='nodes required')

        plot_continuous(self.apply, min(self.densities_), max(self.densities_),
            color=(0, 0.2, 0.6), linewidth=1, zorder=3, label='md + b\nm={}\nb={}'.format(self.m_, self.b_))

        chi2 = chi_sq(self.nodes_, map(self.apply_perfect, self.densities_))
        plot_continuous(self.apply_perfect, min(self.densities_), max(self.densities_),
            color=(0.8, 0, 0.2), label='perfect scaling\nchi2={}'.format(chi2))

        legend()
        xlabel('Density')
        ylabel('# of Nodes')
        suptitle('Weak Scaling'.format(self.time_))
        title('(model={}, time={}s, tasks/node={}'.format(
            self.model_, self.time_, self.tasks_per_node_))
        grid()

        savefig(os.path.join(output, 'weak_scaling_{}_{}_{}.png'.format(
            self.model_, self.time_, self.tasks_per_node_)),
            format='png')
        clf()

    def _fit(self):
        # fit function: f(d) = md + b
        def f(d, m, b):
            return m*d + b

        (m, b), pcov = curve_fit(f, nominal_value(self.densities_),
                                    nominal_value(self.nodes_),
                                    sigma=std_dev(self.nodes_))
        m_err, b_err = np.sqrt(np.diag(pcov))

        self.m_ = ufloat(m, m_err)
        self.b_ = ufloat(b, b_err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize data generated by scaling.py')
    parser.add_argument(
        '--input', type=str, help='CSV file generated by scaling.py (default stdin)', default=None)
    parser.add_argument(
        '--output', type=str, default='.', help='output directory for graphs (default .)')
    args = parser.parse_args()

    csv = np.genfromtxt(args.input if args.input is not None else sys.stdin,
        names=True, delimiter=',', converters={MODEL: get_model_id})

    for model in group_by(MODEL, csv):
        strong_experiments = [StrongScaling(group) for group in group_by(DENSITY, model)]
        weak_experiments = [WeakScaling(t, strong_experiments) for t in [0.2, 0.5, 1.0]]

        for e in strong_experiments + weak_experiments:
            e.save_plot(args.output)
