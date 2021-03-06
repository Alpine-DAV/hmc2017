from mpi4py import MPI
import numpy as np
import sys

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
end_cycle = -1    # stop sampling "good" zones at this cycle (-1 means train all cycles from run 0 - decay_window)
sample_freq = 1000 # how frequently to sample "good" zones
decay_window = 100 # how many cycles back does decay function go for failed (zone,cycle)
load_learning_data = False  # if true, load pre-created learning data
                            # other wise, load raw simulation data and
                            # calculate learning data on-the-fly

#===============================================================================
# RANDOM FOREST CONFIGURATION
#===============================================================================
rand_seed = 0
NumTrees = 50
parallelism = 1 # note: -1 = number of cores on the system
pool_size = 1000
min_samples_split = 1000

#===============================================================================
# TESTING
#===============================================================================
kfold = 10
decision_boundary = 1e-3
threshold = 0 # for sorting samples into positive/negative
np.random.seed(rand_seed)

TOTAL_CYCLES = 40182
