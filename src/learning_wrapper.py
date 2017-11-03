#!usr/bin/env python
from __future__ import division

import argparse
import os
import sys

from mpi4py import MPI
import cPickle
from FeatureDataReader import FeatureDataReader
import glob
import numpy as np
from numpy import isinf, mean, std
import os
import random
import scipy
from scipy.spatial.distance import euclidean
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble.forest import RandomForestRegressor
import sys
import time
import testing_naivebayes as bayes
import testing_randomforest as rand_forest
import nbmpi
from sklearn import datasets

comm = MPI.COMM_WORLD


#===============================================================================
# GLOBAL CONSTANTS
#===============================================================================

TEST_ON_FAIL_ONLY = 1
TEST_ON_FAIL_PLUS_CYCLE_ZERO = 2
TEST_ON_FAIL_PLUS_CYCLE_MAX = 3
TEST_ON_TRAIN_SPEC = 4

enable_feature_importance = True
enable_load_pickled_model = False
enable_save_pickled_model = False
enable_print_predictions = False
start_cycle = 0   # start sampling "good" zones at this cycle
#end_cycle = 0    # only sample "good" from cycle 0
end_cycle = -1    # stop sampling "good" zones at this cycle (-1 means train all cycles from run 0 - decay_window)
sample_freq = 1000 # how frequently to sample "good" zones
#decay_window = 1000 # how many cycles back does decay function go for failed (zone,cycle)
decay_window = 100 # how many cycles back does decay function go for failed (zone,cycle)
load_learning_data = False  # if true, load pre-created learning data
                            # other wise, load raw simulation data and
                            # calculate learning data on-the-fly

# random forest configuration
NumTrees = 1000
rand_seed = None
#rand_seed = 58930865 # (gallagher23) use a fixed seed for reproducible results
parallelism = -1 # note: -1 = number of cores on the system


#===============================================================================
# GLOBAL VARIABLES
#===============================================================================

learning_data_cache = {}

data_readers = {}
def get_reader(data_dir):

    global data_readers

    if data_dir in data_readers:
        reader = data_readers[data_dir]
    else:
        num_partitions = get_num_partitions(data_dir)
        reader = FeatureDataReader(data_dir)
        print "Creating reader for data directory: %s" % (data_dir)
        data_readers[data_dir] = reader

    return reader

#===============================================================================
# FUNCTIONS
#===============================================================================

#
# Looks at directory structure and returns the number of partition index files.
#
def get_num_partitions(data_dir):

    files = glob.glob("%s/indexes/indexes_p*_r000.txt" % data_dir)
    return len(files)

#
# Get features names in order they appear in feature vectors
#
feature_name_cache = None
def get_feature_names(data_dir):
  global feature_name_cache
  if feature_name_cache is None:
    reader = get_reader(data_dir)
    feature_name_cache = reader.getFeatureNames()

  return feature_name_cache


#
# Returns a pair (index,data) where:
#  - index is a list of N (cycle,zone_id) pairs
#  - data is a 2d numpy array of (N instances x F features)
#
# cycles - list of cycles to include
#
def get_learning_data_with_index_for_cycle_range(data_dir, cycles, run):
    reader = get_reader(data_dir)

    index_list = []
    data_list = []
    for cycle in cycles:
        zone_ids = reader.getCycleZoneIds()
        index = zip([cycle]*len(zone_ids), zone_ids)
        data = reader.readAllZonesInCycle(run, cycle)

        index_list = index_list + index
        data_list.append(data)

    return (index_list, np.concatenate(data_list, axis=0))


#
# Take last_weight and decay according to some decay function.
#
def decay(decay_function, step, num_steps):

    if decay_function=='linear':
        return 1.0-step*(1.0/num_steps)

#
# return [failures, failed_cycles] where:
#
# failures is a list of (part, run, cycle, zone) tuples
# failued_cycles is a list containing only the cycles
def get_failures(data_dir):
    # read failure data
    filenames = []
    partitions = range(get_num_partitions(data_dir))
    for pnum in partitions:
        filenames.append("%s/failures/side_p%02d" % (data_dir, pnum))
        filenames.append("%s/failures/corner_p%02d" % (data_dir, pnum))

    failures = []
    failed_cycles = []
    for file in filenames:
        if os.path.isfile(file):
            print "Reading file: ", file

            name,part = file.split("_p")

            with open(file, "r") as fin:
                state = 0
                for line in fin:
                    vals = line.split(",")

                    if vals[0] == "Run":
                        state = 1
                    elif vals[0] == "volume":
                        state = 2
                    else:
                        if state == 1:
                            part = int(part)
                            run = int(vals[0])
                            cycle = int(vals[1])
                            zone = int(vals[2])

                            failures.append((part, run, cycle, zone))
                            failed_cycles.append(int(cycle))

    if len(failures) < 1:
        raise IOError("No failure data found in data directory '%s'." % (data_dir))

    return [failures, failed_cycles]


#
# Create a 2d numpy array of (instance x features)
#
# Good zones come from run 0. To specify a different run, use get_learning_data_for_run.
#
def get_learning_data(data_dir, start_cycle, end_cycle, sample_freq, decay_window, num_failures=-1):
    (index,data) = get_learning_data_for_run(data_dir, start_cycle, end_cycle, sample_freq, decay_window, 0, num_failures)
    return data

#
# Returns a pair (index,data) where:
#  - index is a list of N (cycle,zone_id) pairs
#  - data is a 2d numpy array of (N instances x F features)
#
# Fetch data from cycle 0 and then every 'sample_freq' cycles until end
# of simulation.
#
# Also fetch failed zone data for all applicable cycles.
#
# sample_freq - frequency for sampling cycles (0 mean don't include any "good" examples)
#
def get_learning_data_for_run(data_dir, start_cycle, end_cycle, sample_freq, decay_window, run_for_good_zones, num_failures=-1):
    reader = get_reader(data_dir)

    # cache learning data in memory to improve run time
    global learning_data_cache
    key = ":".join([ data_dir, str(start_cycle), str(end_cycle), str(sample_freq), str(decay_window), str(run_for_good_zones), str(num_failures) ])
    if num_failures < 0 and key in learning_data_cache:
      return learning_data_cache[key] 

    # read failure data
    failures = reader.getAllFailures()
    failed_cycles = [f[1] for f in failures]

    # get first failure cycle
    pre_first_fail = min(failed_cycles) - decay_window

    if (start_cycle > pre_first_fail):
        sys.stderr.write("Warning: specified start_cycle = %d > pre_first_fail = %d. Setting start_cycle = %d.\n" % (start_cycle, pre_first_fail, pre_first_fail))
        start_cycle = pre_first_fail

    if (end_cycle == -1):
        end_cycle = pre_first_fail
    elif (end_cycle > pre_first_fail):
        sys.stderr.write("Warning: specified end_cycle = %d > pre_first_fail = %d. Setting end_cycle = %d.\n" % (end_cycle, pre_first_fail, pre_first_fail))
        end_cycle = pre_first_fail

    if sample_freq == 0:
        # don't include any "good" examples
        candidate_cycles = []
    else:
        candidate_cycles = range(start_cycle, end_cycle+1, sample_freq)


    # remove cycles in range of failures from sample cycles
    good_cycles = []
    for candidate_cycle in candidate_cycles:
        candidate_is_good = True
        for bad_cycle in failed_cycles:
            if candidate_cycle <= bad_cycle and candidate_cycle > (bad_cycle - decay_window):
                candidate_is_good = False
                break

        if candidate_is_good:
            good_cycles.append(candidate_cycle)

    # read data for sample of good zones
    (index_good,good_zones) = get_learning_data_with_index_for_cycle_range(data_dir, good_cycles, run_for_good_zones)
    if good_zones is None:
        Y_good = None
    else:
        Y_good = [0] * good_zones.shape[0]

    # sample failures
    # choose 'num_failures' failures uniformly at random
    if (num_failures > -1 and num_failures < len(failures)):
        random.shuffle(failures)
        failures = failures[0:num_failures]

    # read data for bad zones
    # assign weights to failures based on function 'decay'
    bad_zones_list = []
    Y_bad = []
    index_bad = []
    for fail in failures:
        [run, fail_cycle, zone, features] = fail
        weight = 1.0

        all_cycle_data = reader.readAllCyclesForFailedZone(run,fail_cycle,zone)
        data = list(reversed(all_cycle_data[-(1+decay_window):-1:]))
        bad_zones_list.append(data)

        for step in range(decay_window):

            cycle = fail_cycle - step
            weight = decay('linear', step, decay_window)

            Y_bad.append(weight)
            index_bad.append((cycle,zone))

    bad_zones = np.concatenate(bad_zones_list, axis=0)

    # combine good and bad zones
    if bad_zones is None:
        X = good_zones
        Y_list = Y_good
        index = index_good
    elif good_zones is None:
        X = bad_zones
        Y_list = Y_bad
        index = index_bad
    else:
        X = np.vstack((good_zones, bad_zones))
        Y_list = Y_good + Y_bad
        index = index_good + index_bad

    Y = np.array(Y_list).reshape(len(Y_list),1)

    # hook for adding additional features
    new_features = add_features(index)
    X = np.hstack((X, new_features))

    # combine X and Y
    XY = np.hstack((X,Y))

    np.savetxt("FeatureData.csv", XY, delimiter=',')

    # debug output

    # cache data for subsequent calls to this function
    return_val = (index, XY)
    if num_failures < 0:
        learning_data_cache[key] = return_val

    return return_val

#
# Input:
# - index is a list of N (cycle,zone_id) pairs
#
# Output:
# - N x F numpy array where:
#   - F is the number of new features added.
#   - Array element (n,f) is the value of feature f for the nth (cycle, zone_id) pair in index.
#
def add_features(index):

    new_feature_vals = []

    # for each cycle and zone_id
    for (cycle, zone_id) in index:

      # lookup or calculate feature values for this cycle and zone

      #########################################
      # vvv INSERT YOUR NEW FEATURES HERE vvv #
      #########################################

      cycle_zone_values = [] 
      #cycle_zone_values = [1,1,1] # just as a placeholder, we add three new features all with value=1

      #########################################
      # ^^^ INSERT YOUR NEW FEATURES HERE ^^^ #
      #########################################

      np_array_row = np.array(cycle_zone_values).reshape(1,len(cycle_zone_values))
      new_feature_vals.append(np_array_row)

    return np.concatenate(new_feature_vals, axis=0)


#####################################################################

#####################################################################

#####################################################################

def wrapper(ML_type, k, data_path, verbose=False, use_online=False):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """

    dataset = get_data(data_path, TEST_ON_TRAIN_SPEC)
    X = dataset[:,0:-1]
    y = np.ravel(dataset[:,[-1]])
    train, test = data_split(data_path, 1.0/k, TEST_ON_TRAIN_SPEC)

    discretized_y = np.zeros(len(y))
    for element in range(len(y)):
        if y[element] != 0:
            discretized_y[element] = 1

    # X = np.delete(X, np.s_[0,1,2,3,4,5], 1)

    if ML_type == "naive bayes":
        print "############ Training using Naive Bayes ############"
        # bayes.train_many_test_many(train, test, TEST_ON_TRAIN_SPEC)
        y = discretized_y

        bayes.train_and_test_k_fold(X, y, k)

        print 
        print
    elif ML_type == "random forest":
        print "############ Training using Random Forest ############"
        rand_forest.train_and_test_k_fold(X, y, k)
        print 
        print
    elif ML_type == "nbmpi":
        print "############ Training using Parallel Naive Bayes ############"
        y = discretized_y

        use_mpi = 'MPICH_INTERFACE_HOSTNAME' in os.environ

        if use_mpi and comm.rank == 0:
            print('will train using MPI')
        if use_online and comm.rank == 0:
            print('will train in online mode')

        fp, fn = nbmpi.train_and_test_k_fold(X, y, k=k, verbose=verbose, use_online=use_online, use_mpi=use_mpi)

        if comm.rank == 0:
            print "PERFORMANCE\t%d\t%d" % (fp, fn)

    else:
        raise Exception('Machine learning algorithm not recognized')

def data_split(data_path, k, test_data_spec):
    all_data = get_bubbleshock()

    # all_data = np.delete(all_data, np.s_[0,1,2,3,4,6,8,9,10,11,12,13,14,15], 1)
    k_percent = int(len(all_data)*k)
    train = all_data[:k_percent,:]
    test = all_data[k_percent:,:]

    return train, test

def get_bubbleshock():
    dataset = None
    start = time.time()
    dataset = get_learning_data('bubbleShock', start_cycle, end_cycle, sample_freq, decay_window)
    end = time.time()
    print "TIME load training data: ", end-start
    return dataset


if __name__ == '__main__':

    # Read command line inputs
    parser = argparse.ArgumentParser(
        description='Train and test a naive Bayes classifier using the bubbleShock dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('-nb', action='store_true', help='train using naive bayes')
    parser.add_argument('-rf', action='store_true', help='train using random forest')
    parser.add_argument('-nbp', action='store_true', help='train using parallel naive bayes')
    args = parser.parse_args()

    verbose    = args.verbose
    use_online = args.online
    model_nb   = args.nb
    model_rf   = args.rf
    model_nbp  = args.nbp

    bubbleShock = get_bubbleshock()
    X = bubbleShock[:,0:-1]
    y = np.ravel(bubbleShock[:,[-1]])


    discretized_y = np.zeros(len(y))
    for element in range(len(y)):
        if y[element] != 0:
            discretized_y[element] = 1

    if model_nb:
        wrapper('naive bayes', k, ['bubbleShock'], verbose=verbose)
    if model_rf:
        wrapper('random forest', k, ['bubbleShock'], verbose=verbose)
    if model_nbp:
        use_mpi = 'MPICH_INTERFACE_HOSTNAME' in os.environ
        use_online = False

        if use_mpi and comm.rank == 0:
            print('will train using MPI')
        if use_online and comm.rank == 0:
            print('will train in online mode')
        
        y = discretized_y

        fp, fn = train_and_test_k_fold(X, y, k=10, verbose=verbose, use_online=use_online, use_mpi=use_mpi)

        if comm.rank == 0:
            print "PERFORMANCE\t%d\t%d" % (fp, fn)
