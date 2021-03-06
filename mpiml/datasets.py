from functools import wraps
import glob
from math import ceil
import numpy as np
import sklearn.datasets as sk
import sys
import time
from utils import root_info, debug
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

class DataSet(object):
    """
    Abstract representation of a data set.

    This class exists to allow implementations which load data lazily from disk, allowing us to use
    datasets which do not fit in memory. To support this, the abstraction breaks the dataset into a
    series of chunks called "cycles". For example, one cycle may correspond to a single timestep of
    a simulation. Cycles can then be loaded from disk one at a time and independently from one
    another.

    This class functions as an abstract base class. Derived classes should implement num_cycles,
    which takes no arguments and returns the total number of cycles, and get_cycle, which takes an
    integer in [0, num_cycles()) and returns an (X, y) pair where X is an array of feature vectors
    and y is an array of labels.
    """

    def map(self, f):
        """
        Transform a dataset by applying the given function to each cycle. The function should take
        two arguments: the first is an array of feature vectors, and then second is an array of
        labels. Returns a new DataSet object.
        """

        class MapDataSet(DataSet):
            def __init__(self, ds, f):
                self.ds_ = ds
                self.f_ = f

            def get_cycle(self, i):
                return self.f_(*self.ds_.get_cycle(i))

            def num_cycles(self):
                return self.ds_.num_cycles()
        return MapDataSet(self, f)

    def classes(self):
        """
        Return an array of unique labels from the dataset.
        """
        return np.unique(self.points()[1])

    def cycles(self):
        """
        Return a generator which yields, in order, each cycle in the dataset.
        """
        return (self.get_cycle(i) for i in range(self.num_cycles()))

    def points(self):
        xs, ys = zip(*self.cycles())
        return np.vstack(xs), np.concatenate(ys)

    def split(self, k):
        """
        Split a dataset into k disjoint subsets. Returns an array of DataSet objects.
        """

        class SplitDataSet(DataSet):
            def __init__(self, ds, start):
                self.ds_ = ds
                self.start_ = start

            def get_cycle(self, i):
                return self.ds_.get_cycle(self.start_ + k*i)

            def num_cycles(self):
                return int((self.ds_.num_cycles() - self.start_) / k)

        return [SplitDataSet(self, start) for start in range(k)]

    def concat(self, ds):
        """
        Append the samples in this dataset and the samples in ds into a new DataSet object.
        """
        class ConcatDataSet(DataSet):
            def __init__(self, ds1, ds2):
                self.ds1_ = ds1
                self.ds2_ = ds2

            def get_cycle(self, i):
                if i < self.ds1_.num_cycles():
                    return self.ds1_.get_cycle(i)
                else:
                    return self.ds2_.get_cycle(i - self.ds1_.num_cycles())

            def num_cycles(self):
                return self.ds1_.num_cycles() + self.ds2_.num_cycles()

        return ConcatDataSet(self, ds)

class InMemoryDataSet(DataSet):
    """
    Implementation of the DataSet interface which stores the entire dataset in memory.
    """

    def __init__(self, X, y, pool_size=config.pool_size):
        """
        Initialize an InMemoryDataSet with given data.

        X: array of feature vectors
        y: array of labels
        pool_size: number of samples to be "pooled together" in each cycle
        """
        self.X_ = X
        self.y_ = y
        self.pool_size_ = pool_size

    def get_cycle(self, i):
        return self.X_[i*self.pool_size_ : (i + 1)*self.pool_size_], \
               self.y_[i*self.pool_size_ : (i + 1)*self.pool_size_]

    def num_cycles(self):
        return int(ceil(float(self.X_.shape[0]) / config.pool_size))

class EmptyDataSet(DataSet):
    """
    Implementation of the DataSet interface which represents a dataset with no samples. Useful as a
    base case for combining many datasets.
    """

    def get_cycle(self, i):
        raise ValueError('cannot get cycle from empty dataset')

    def num_cycles(self):
        return 0

def concatenate(datasets):
    """
    Concatenate an iterable of DataSet objects, returning a single new DataSet.
    """
    return reduce(lambda x, y: x.concat(y), datasets, EmptyDataSet())

def ds_map(f):
    """
    Decorator which turns a function that transforms X and y vectors into a function that transforms
    a DataSet using the DataSet.map interface.
    """

    @wraps(f)
    def mapper(ds, *args, **kwargs):
        return ds.map(lambda X, y: f(X, y, *args, **kwargs))
    return mapper

@ds_map
def shuffle_data(X, y, seed=0):
    """
    Randomly reorder a dataset to remove patterns between adjacent samples.
    """

    np.random.seed(0)
    seed = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(seed)
    np.random.shuffle(y)
    return X, y

def get_bubbleshock_by_hand(data_dir):
    """
    Return a DataSet object representing the Bubble Shock By Hand dataset, which should be stored in
    the file system at the given directory. The resulting dataset loads cycles lazily from disk, so
    the entire dataset need not fit in memory at once.
    """

    class ByHandDataSet(DataSet):
        def __init__(self):
            self.reader_ = get_reader(data_dir)

        def get_cycle(self, i):
            debug('read byHand cycle {}', i)

            dataset = self.reader_.readAllZonesInCycle(0, i)

            X = dataset[:,0:-2]
            y = np.ravel(dataset[:,[-1]])
            zone_scale = np.ravel(dataset[:,[-2]])

            for j in range(len(y)):
                if zone_scale[j] > 0:
                    y[j] = y[j] / zone_scale[j]

            return X, y

        def num_cycles(self):
            return config.TOTAL_CYCLES

    return ByHandDataSet()

def get_bubbleshock(data_dir='bubbleShock', pool_size=config.pool_size):
    """
    Return a DataSet object representing the Bubble Shock dataset, which should be stored in the
    file system at the given directory. Since the Bubble Shock dataset is relatively small, this
    function uses the InMemoryDataSet class to represent the dataset. The pool_size parameter
    corresponds to the pool_size parameter of the InMemoryDataSet constructor.
    """

    dataset = None
    start = time.time()
    dataset = get_learning_data(data_dir, config.start_cycle, config.end_cycle, config.sample_freq, config.decay_window)
    end = time.time()
    root_info("TIME load training data: {}", end-start)

    X = dataset[:,0:-1]
    y = np.ravel(dataset[:,[-1]])

    return InMemoryDataSet(X, y, pool_size=pool_size)

def prepare_dataset(dataset, discrete=False, density=1.0, pool_size=config.pool_size):
    """
    Load a desired DataSet and prepare it for learning. Preparation may involve, for example,
    shuffling or discretizing the dataset.

    The dataset parameter is a string specifying the dataset. It can be the name of one of the
    scikit-learn example datasets, or a path to bubble shock or bubble shock by hand. We use a hacky
    heuristic to differentiate between the latter two: a path is considered to point to the by hand
    dataset if it contains the string "byHand".

    If discrete is True, the labels in the dataset will be discretized to 0 or 1 to make the dataset
    suitable for a classification problem, as if by calling discretize.

    Density should be a float between 0 and 1 inclusive representing a fraction of the dataset to
    return, as if by applying make_sparse to the dataset. This is useful for constructing learning
    problems of various sizes.

    pool_size will be passed to InMemoryDataSet if the resulting dataset uses that implementation of
    the DataSet interface.
    """

    if hasattr(sk, 'load_{}'.format(dataset)): # Does it look like a sklearn example dataset?
        dataset = getattr(sk, 'load_{}'.format(dataset))()
        ds = shuffle_data(InMemoryDataSet(dataset.data, dataset.target, pool_size=config.pool_size))
    elif 'byHand' in dataset: # Does it look like bubble shock by hand? TODO make this more robust
        ds = get_bubbleshock_by_hand(dataset)
    else: # Assume it's bubble shock
        ds = shuffle_data(get_bubbleshock(data_dir=dataset, pool_size=config.pool_size))

    if discrete:
        ds = discretize(ds)

    return make_sparse(ds, density)

def make_sparse(ds, density):
    """
    Return a new DataSet which represents a subset of the given dataset, whose size is a fraction
    density of the original dataset.
    """
    if density == None or 1.0 - density < 0.001:
        return ds

    class SparseDataSet(DataSet):
        def __init__(self):
            self.ds_ = ds
            self.stride_ = 1.0 / density

        def get_cycle(self, i):
            return self.ds_.get_cycle(i * self.stride_)

        def num_cycles(self):
            return int(ceil(float(self.ds_.num_cycles()) / self.stride_))

    return SparseDataSet()

@ds_map
def discretize(X, y):
    """
    Map the labels in a dataset to 0 or 1 based on comparison with config.decision_boundary. In
    effect, this function turns a regression problem into a binary classification problem.
    """
    if y.ndim != 1:
        raise ValueError("can only discretize 1d array (got {})".format(y.ndim))
    return X, y > config.decision_boundary

def threshold_count(ds, thresh):
    """
    Return a pair of the number of positive examples (class > threshold) and the number of negative
    examples (class <= threshold) in the given dataset.
    """

    npos = 0
    nneg = 0
    for _, y in ds.cycles():
        npos += np.sum(y > thresh)
        nneg += np.sum(y <= thresh)
    return npos, nneg

learning_data_cache = {}
data_readers = {}
def get_reader(data_dir):
    """
    Get a DataReader object which extracts features and labels from the dataset stored in the given
    directory. Uses a global cache to avoid creating redundant readers.
    """

    global data_readers

    if data_dir in data_readers:
        reader = data_readers[data_dir]
    else:
        num_partitions = get_num_partitions(data_dir)
        reader = FeatureDataReader(data_dir)
        root_info("Creating reader for data directory: {}", data_dir)
        data_readers[data_dir] = reader

    return reader

def output_feature_importance(result, data_dir):
    """
    Print feature importance for a random forest model.
    """

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

def get_num_partitions(data_dir):
    """
    Looks at directory structure and returns the number of partition index files.
    """
    files = glob.glob("%s/indexes/indexes_p*_r000.txt" % data_dir)
    return len(files)


#===============================================================================
# Private stuff
#===============================================================================

def get_learning_data_with_index_for_cycle_range(data_dir, cycles, run):
    """
    Returns a pair (index,data) where:
     - index is a list of N (cycle,zone_id) pairs
     - data is a 2d numpy array of (N instances x F features)

    cycles - list of cycles to include
    """

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

def decay(decay_function, step, num_steps):
    """
    Take last_weight and decay according to some decay function.
    """
    if decay_function=='linear':
        return 1.0-step*(1.0/num_steps)

def get_failures(data_dir):
    """
    return [failures, failed_cycles] where:

    failures is a list of (part, run, cycle, zone) tuples
    failued_cycles is a list containing only the cycles
    """

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

def get_learning_data(data_dir, start_cycle, end_cycle, sample_freq, decay_window, num_failures=-1):
    """
    Create a 2d numpy array of (instance x features)

    Good zones come from run 0. To specify a different run, use get_learning_data_for_run.
    """
    (index,data) = get_learning_data_for_run(data_dir, start_cycle, end_cycle, sample_freq, decay_window, 0, num_failures)
    return data

def get_learning_data_for_run(data_dir, start_cycle, end_cycle, sample_freq, decay_window, run_for_good_zones, num_failures=-1):
    """
    Returns a pair (index,data) where:
     - index is a list of N (cycle,zone_id) pairs
     - data is a 2d numpy array of (N instances x F features)

    Fetch data from cycle 0 and then every 'sample_freq' cycles until end
    of simulation.

    Also fetch failed zone data for all applicable cycles.

    sample_freq - frequency for sampling cycles (0 mean don't include any "good" examples)
    """

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

def add_features(index):
    """
    Input:
    - index is a list of N (cycle,zone_id) pairs

    Output:
    - N x F numpy array where:
      - F is the number of new features added.
      - Array element (n,f) is the value of feature f for the nth (cycle, zone_id) pair in index.
    """

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
