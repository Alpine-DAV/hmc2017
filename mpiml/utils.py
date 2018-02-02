from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
import os
import sys

import config

__all__ = [ "info"
          , "debug"
          , "root_info"
          , "root_debug"
          , "running_in_mpi"
          , "output_model_info"
          , "toggle_profiling"
          , "toggle_verbose"
          , "profile"
          ]

_verbose = False

def toggle_verbose(v=True):
    global _verbose
    _verbose = v

def _extract_arg(arg, default, kwargs):
    if arg in kwargs:
        res = kwargs[arg]
        del kwargs[arg]
        return res
    else:
        return default

def _info(fmt, *args, **kwargs):
    print(fmt.format(*args, **kwargs), file=sys.stderr)

def info(fmt, *args, **kwargs):
    comm = _extract_arg('comm', MPI.COMM_WORLD, kwargs)
    if type(fmt) == str:
        fmt = 'rank {}: ' + fmt
    else:
        args = [fmt]
        fmt = '{}'
    _info(fmt, comm.rank, *args, **kwargs)

def debug(fmt, *args, **kwargs):
    if _verbose:
        info(fmt, *args, **kwargs)

def root_info(fmt, *args, **kwargs):
    comm = _extract_arg('comm', MPI.COMM_WORLD, kwargs)
    root = _extract_arg('root', 0, kwargs)
    if comm.rank != root:
        return
    if type(fmt) != str:
        args = [fmt]
        fmt = '{}'
    _info(fmt, *args, **kwargs)

def root_debug(fmt, *args, **kwargs):
    if _verbose:
        root_info(fmt, *args, **kwargs)

# Determine if we are running as an MPI process
def running_in_mpi():
    return 'MPICH_INTERFACE_HOSTNAME' in os.environ or \
           'MPIRUN_ID' in os.environ

_profiling_enabled = False
def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            if not _profiling_enabled:
                return f(*args, **kwargs)

            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)
            return result
        return wrap_f
    return prof_decorator

def toggle_profiling(enabled=True):
    global _profiling_enabled
    _profiling_enabled = enabled

def output_model_info(model, online, density):
    output_str = \
"""
---------------------------
ML model:  {ml_type}
num cores: {num_cores}
MPI:       {use_mpi}
online:    {online}
density:   {density}
---------------------------
""".format(ml_type=model.__class__.__name__,
           num_cores=config.comm.size,
           use_mpi=running_in_mpi(),
           **locals())

    if hasattr(model, 'estimators_'):
        output_str += \
"""
num trees: {num_trees}
---------------------------
""".format(num_trees=len(model.estimators_))

    return output_str

if not running_in_mpi():
    root_info("WARNING: NOT RUNNING WITH MPI")
