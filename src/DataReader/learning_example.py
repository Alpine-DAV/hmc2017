#!usr/bin/env python

import cPickle
from FeatureDataReader import FeatureDataReader
import glob
import numpy as np
from numpy import isinf, mean, std
import os
import random
import scipy
from scipy.spatial.distance import euclidean
from sklearn.ensemble.forest import RandomForestRegressor
import sys
import time


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

    # debug output

    # cache data for subsequent calls to this function
    return_val = (index, XY)
    if num_failures < 0:
        learning_data_cache[key] = return_val

    return return_val

#
# Print feature importance for a random forest model.
#
def output_feature_importance(rand_forest, data_dir):
    importances = rand_forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = get_feature_names(data_dir)
    for f in range(len(importances)):
        feature_index = indices[f]
        try:
          feature_name = feature_names[feature_index]
        except IndexError:
          feature_name = 'UNKNOWN'
        print "FEATURE\t%d\t%d\t%s\t%f" % ((f + 1), feature_index, feature_name, importances[feature_index])

#
# Train a single model on all train_data_paths, evaluate separately on each of test_data_paths.
#
def train_many_test_many(train_data_paths, test_data_paths, test_data_spec):

    test_run = 0

    if test_data_spec == TEST_ON_FAIL_ONLY:
        test_start_cycle = 0; test_end_cycle = 0; test_sample_freq = 0 # don't use any "good" examples
    elif test_data_spec == TEST_ON_FAIL_PLUS_CYCLE_ZERO:
        test_start_cycle = 0; test_end_cycle = 1; test_sample_freq = 1000 # use only "good" example from first cycle
    elif test_data_spec == TEST_ON_FAIL_PLUS_CYCLE_MAX:
        test_start_cycle = 9999999; test_end_cycle = 9999999; test_sample_freq = 1000 # use only "good" example from "max" cycle
    elif test_data_spec == TEST_ON_TRAIN_SPEC:
        test_start_cycle = start_cycle; test_end_cycle = end_cycle; test_sample_freq = sample_freq
    else:
        sys.err.write("Invalid test_data_spec '%d'. Must be one of: TEST_ON_FAIL_ONLY, TEST_ON_FAIL_PLUS_CYCLE_ZERO, TEST_ON_FAIL_PLUS_CYCLE_MAX\n" % test_data_spec)

    # print output headers
    print "PERFORMANCE\test_data_spec\ttest_path\ttest_run\tpiston_param\tdensity_param\trmse\tfp\tfn\tnum_instances\truntime_secs"

    # load pickled model
    if enable_load_pickled_model:
        print 'Using pre-trained model from: randomforest.pkl. This may take a couple of minutes...'
        with open('randomforest.pkl','rb') as f:
            rand_forest = cPickle.load(f)
    else:
        train = None
        start = time.time()
        train_list = []
        for train_data_path in train_data_paths:
            print train_data_path
            train_next = get_learning_data(train_data_path, start_cycle, end_cycle, sample_freq, decay_window)
            train_list.append(train_next)
        train = np.concatenate(train_list, axis=0)
        print "training data: " , train.shape

        end = time.time()
        print "TIME load training data: ", end-start

        # Train the random forest
        train_X = train[:,0:-1]
        train_Y = np.ravel(train[:,[-1]])
        start = time.time()
        rand_forest = RandomForestRegressor(n_estimators=NumTrees, n_jobs=parallelism, random_state=rand_seed)
        rand_forest.fit(train_X, train_Y)
        end = time.time()
        print "TIME train: ", end-start


    # output feature importance
    if enable_feature_importance:
        output_feature_importance(rand_forest, train_data_paths[0])

    # pickle model for future use
    if enable_save_pickled_model:
        print "Writing random forest model to file: randomforest.pkl. This may take a couple of minutes..."
        with open('randomforest.pkl','wb') as f:
            cPickle.dump(rand_forest,f)
        print "Wrote random forest model to file: randomforest.pkl"

    for test_path in test_data_paths:

        start = time.time()

        piston_param = 0
        density_param = 0
        try:
            (index,test) = get_learning_data_for_run(test_path, test_start_cycle, test_end_cycle, test_sample_freq, decay_window, test_run)
            print "test data: ", test.shape

            # Check results on cv set
            test_X = test[:,0:-1]
            test_Y = np.ravel(test[:,[-1]])
            cv_predict = rand_forest.predict(test_X)
            #decision_boundary = min(cv_predict)
            decision_boundary = 4e-6
            RMSE = np.sqrt( sum(pow(test_Y - cv_predict, 2)) / test_Y.size )
            #err = sum(cv_predict - test_Y) / test_Y.size 
            #pos_indices = [i for i, x in enumerate(test_Y) if x > 0]
            #neg_indices = [i for i, x in enumerate(test_Y) if x == 0]
            #err_on_pos = sum(np.array([cv_predict[i] for i in pos_indices]) - np.array([test_Y[i] for i in pos_indices])) / len(pos_indices)
            #err_on_neg = sum(np.array([cv_predict[i] for i in neg_indices]) - np.array([test_Y[i] for i in neg_indices])) / len(neg_indices)

            # calculate false positives and false negatives
            fp = fn = 0
            for i in range(len(test_Y)):
                if test_Y[i] == 0 and cv_predict[i] > decision_boundary:
                    fp += 1
                elif test_Y[i] > 0 and cv_predict[i] <= decision_boundary:
                    fn += 1

            end = time.time()

            if enable_print_predictions:
                for i in range(len(test_Y)):
                    print test_Y[i], cv_predict[i]

            if "piston" in test_path:
                piston_offset = test_path.find("piston") + len("piston")
                piston_param = int(test_path[piston_offset:piston_offset+3])
                density_offset = test_path.find("density") + len("density")
                density_param = float(test_path[density_offset:density_offset+4])

            print "PERFORMANCE\t%d\t%s\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_path, test_run, piston_param, density_param, RMSE, fp, fn, len(test_Y), round(end-start))
            sys.stdout.flush()
        except:
            end = time.time()
            print "PERFORMANCE\t%d\t%s\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_path, test_run, piston_param, density_param, 0, 0, 0, 0, round(end-start))

#####################################################################

#####################################################################

#####################################################################

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

#
# Run quick learning test.
#
def main():

    if len(sys.argv) < 2:
        print "Usage: python %s <data_directory>" % sys.argv[0]
        sys.exit(1)
 
    train_path = sys.argv[1]
    test_path = train_path

    train_many_test_many([train_path], [test_path], TEST_ON_TRAIN_SPEC)


if __name__ == '__main__':
  main()

