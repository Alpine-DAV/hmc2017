from functools import wraps
import glob
import numpy as np
import sklearn.datasets as sk
import sys
import time
from utils import root_info
import config

from DataReader.FeatureDataReader import FeatureDataReader

__all__ = [ "get_bubbleshock"
          , "get_bubbleshock_byhand_by_cycle"
          , "get_bubbleshock_by_hand"
          , "prepare_dataset"
          , "shuffle_data"
          , "discretize"
          , "get_reader"
          , "output_feature_importance"
          , "get_num_partitions"
          , "concatenate"
          ]

def every_kth(arr, k):
    return [arr[start::k] for start in range(k)]

class NullDataSet(object):
    def map(self, f):
        pass

    def classes(self):
        return []

    def cycles(self):
        return []

    def points(self):
        return [],[]

    def split(self, k):
        return [self]*k

    def concat(self, ds):
        return ds

class StrictDataSet(object):

    def __init__(self, X, y, pool_size=config.pool_size):
        self.X = X
        self.y = y
        self.pool_size_ = pool_size

    def map(self, f):
        return StrictDataSet(*f(self.X, self.y))

    def classes(self):
        return np.unique(self.y)

    def cycles(self):
        return ((self.X[i:i+self.pool_size_], self.y[i:i+self.pool_size_])
            for i in xrange(0, self.X.shape[0], self.pool_size_))

    def points(self):
        return self.X, self.y

    def split(self, k):
        split_X = every_kth(self.X, k)
        split_y = every_kth(self.y, k)
        return [StrictDataSet(X, y) for X, y in zip(split_X, split_y)]

    def concat(self, ds):
        if isinstance(ds, NullDataSet):
            return self
        else:
            X, y = ds.points()
            return StrictDataSet(np.vstack((self.X, X)), np.concatenate((self.y, y)))

class LazyDataSet(object):
    # gen: a function returning a generator that lazily loads a DataSet object for each cycle
    #      gen must be a factory function rather than a generator object so that we can reuse the
    #      generator (by creating a new instance from the factory)
    def __init__(self, make_gen):
        self.make_gen_ = make_gen

    def map(self, f):
        return LazyDataSet(lambda: (ds.map(f) for ds in self.make_gen_()))

    def classes(self):
        return np.unique(np.array(ds.classes() for ds in self.make_gen_()))

    def cycles(self):
        return (ds.points() for ds in self.make_gen_())

    def points(self):
        return concatenate(self.make_gen_()).points()

    def split(self, k):
        def gen(i):
            for ds in self.make_gen_():
                X, y = ds.points()
                yield StrictDataSet(X[i::k], y[i::k])
        return [LazyDataSet(lambda: gen(i)) for i in range(k)]

    def concat(self, ds):
        if isinstance(ds, NullDataSet):
            return self

        def gen():
            for d in self.make_gen_():
                yield d
            for X, y in ds.cycles():
                yield StrictDataSet(X, y)
        return LazyDataSet(lambda: gen())

def concatenate(datasets):
    return reduce(lambda x, y: x.concat(y), datasets, NullDataSet())

# Turn a function that transforms X and y vectors into a function that transforms a DataSet
def ds_map(f):
    @wraps(f)
    def mapper(ds, *args, **kwargs):
        return ds.map(lambda X, y: f(X, y, *args, **kwargs))
    return mapper

# Reorder a dataset to remove patterns between adjacent samples. The random state is seeded with a
# constant before-hand, so the results will not vary between runs.
@ds_map
def shuffle_data(X, y, seed=0):
    np.random.seed(0)
    seed = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(seed)
    np.random.shuffle(y)
    return X, y

def get_bubbleshock_byhand_by_cycle(data_dir, cycle, density=1.0, pool_size=config.pool_size):
    dataset = None
    reader = get_reader(data_dir)
    feature_names = reader.getFeatureNames()
    zids = reader.getCycleZoneIds()
    dataset = reader.readAllZonesInCycle(0, cycle)

    X = dataset[:,0:-1]
    y = np.ravel(dataset[:,[-1]])

    return make_sparse(StrictDataSet(X, y, pool_size), density)

def get_bubbleshock_by_hand(data_dir, density=1.0, pool_size=config.pool_size, train_test_split=None):
    ds = None
    if train_test_split == None:
        ds = LazyDataSet(lambda: (get_bubbleshock_byhand_by_cycle(data_dir, cycle, density, pool_size)
                                for cycle in range(config.TOTAL_CYCLES)))
    else:
        subset_cycles = train_test_split['train_split'] + train_test_split['test_split']
        ds = LazyDataSet(lambda: (get_bubbleshock_byhand_by_cycle(data_dir, cycle, density, pool_size)
                                for cycle in range(subset_cycles)))
    return ds

def get_bubbleshock(data_dir='bubbleShock', discrete=False, density=1.0):
    dataset = None
    start = time.time()
    dataset = get_learning_data(data_dir, config.start_cycle, config.end_cycle, config.sample_freq, config.decay_window)
    end = time.time()
    root_info("TIME load training data: {}", end-start)

    X = dataset[:,0:-1]
    y = np.ravel(dataset[:,[-1]])

    if discrete:
        y = discretize(y)

    return make_sparse(StrictDataSet(X, y), density)

# Load the requested example dataset and randomly reorder it so that it is not grouped by class
def prepare_dataset(dataset, discrete=False, density=1.0, pool_size=config.pool_size, train_test_split=None):
    global _dataset
    _dataset = dataset
    if hasattr(sk, 'load_{}'.format(dataset)):
        dataset = getattr(sk, 'load_{}'.format(dataset))()
        ds = shuffle_data(StrictDataSet(dataset.data, dataset.target))
    elif 'byHand' in dataset:
        ds = get_bubbleshock_by_hand(dataset, pool_size, train_test_split=train_test_split)
    else:
        ds = shuffle_data(get_bubbleshock(data_dir=dataset))

    if discrete:
        ds = discretize(ds)

    return make_sparse(ds, density)

@ds_map
def make_sparse(X, y, density):
    if 1.0 - density < 0.001:
        return X, y
    indices = np.random.choice(y.shape[0], int(y.shape[0]*density), replace=False)
    return X[indices], y[indices]

@ds_map
def discretize(X, y):
    if y.ndim != 1:
        raise ValueError("can only discretize 1d array (got {})".format(y.ndim))
    discretized = np.zeros(y.shape[0])
    for element in range(y.shape[0]):
        if y[element] != 0:
            discretized[element] = 1
    return X, discretized

# Return a pair of the number of positive examples (class > threshold) and the number of negative
# examples (class <= threshold)
def threshold_count(ds, thresh):
    npos = 0
    nneg = 0
    for _, y in ds.cycles():
        npos += np.sum(y > thresh)
        nneg += np.sum(y <= thresh)
    return npos, nneg

learning_data_cache = {}
data_readers = {}
def get_reader(data_dir):

    global data_readers

    if data_dir in data_readers:
        reader = data_readers[data_dir]
    else:
        num_partitions = get_num_partitions(data_dir)
        reader = FeatureDataReader(data_dir)
        root_info("Creating reader for data directory: {}", data_dir)
        data_readers[data_dir] = reader

    return reader

#
# Print feature importance for a random forest model.
#
def output_feature_importance(result, data_dir):
    rand_forest = result['clf']
    if rand_forest:
        importances = rand_forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        reader = get_reader(data_dir)
        feature_names = reader.getFeatureNames()
        result = ""
        for f in range(len(importances)):
            feature_index = indices[f]
            try:
              feature_name = feature_names[feature_index]
            except IndexError:
              feature_name = 'UNKNOWN'
            result += "FEATURE\t%d\t%d\t%s\t%f" % ((f + 1), feature_index, feature_name, importances[feature_index])
            result += "\n"
        return result
#
# Looks at directory structure and returns the number of partition index files.
#
def get_num_partitions(data_dir):

    files = glob.glob("%s/indexes/indexes_p*_r000.txt" % data_dir)
    return len(files)


#===============================================================================
# Private stuff
#===============================================================================

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
            root_info("Reading file: {}", file)

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
