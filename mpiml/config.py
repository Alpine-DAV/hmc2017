from mpi4py import MPI

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
# ML MODEL NAMES
#===============================================================================
NAIVE_BAYES         = 'nb'
NAIVE_BAYES_MPI     = 'nbp'
RANDOM_FOREST       = 'rf'
RANDOM_FOREST_MPI   = 'rfp'
VALID_MODELS = [NAIVE_BAYES, NAIVE_BAYES_MPI, RANDOM_FOREST, RANDOM_FOREST_MPI]

#===============================================================================
# RANDOM FOREST CONFIGURATION
#===============================================================================
rand_seed = 0
NumTrees = 10
parallelism = -1 # note: -1 = number of cores on the system

#===============================================================================
# TESTING
#===============================================================================
kfold = 10
decision_boundary = 4e-6
