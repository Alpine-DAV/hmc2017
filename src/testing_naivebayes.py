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
from sklearn.naive_bayes import GaussianNB
import sys
import time


#===============================================================================
# GLOBAL CONSTANTS
#===============================================================================

TEST_ON_FAIL_ONLY = 1
TEST_ON_FAIL_PLUS_CYCLE_ZERO = 2
TEST_ON_FAIL_PLUS_CYCLE_MAX = 3
TEST_ON_TRAIN_SPEC = 4

enable_load_pickled_model = False
enable_save_pickled_model = False
enable_print_predictions = False
start_cycle = 0   # start sampling "good" zones at this cycle
# end_cycle = 0    # only sample "good" from cycle 0
end_cycle = -1    # stop sampling "good" zones at this cycle (-1 means train all cycles from run 0 - decay_window)
sample_freq = 1000 # how frequently to sample "good" zones
# decay_window = 1000 # how many cycles back does decay function go for failed (zone,cycle)
decay_window = 100 # how many cycles back does decay function go for failed (zone,cycle)
load_learning_data = False  # if true, load pre-created learning data
                            # other wise, load raw simulation data and
                            # calculate learning data on-the-fly
parallelism = -1 # note: -1 = number of cores on the system

SCALING_FACTOR = 1000



# A generator yielding a tuple of (training features, training labels, test features, test labels)
# for each run in a k-fold cross validation experiment. By default, k=10.
def get_k_fold_data(X, y, k=10):
    n_samples = X.shape[0]
    n_test = n_samples // k
    for i in range(k):
        train_X = np.concatenate((X[:n_test*i], X[n_test*(i+1):]))
        test_X = X[n_test*i:n_test*(i+1)]
        train_y = np.concatenate((y[:n_test*i], y[n_test*(i+1):]))
        test_y = y[n_test*i:n_test*(i+1)]
        yield (train_X, test_X, train_y, test_y)

def train_and_test_k_fold(X, y, k):
    time_train = 0
    time_test = 0

    n_samples = X.shape[0]
    n_test = n_samples // k
    # Scale the continuous class values into integers from 0 to 100
    # y = np.array([int(x*10) for x in y])
    new_y = np.zeros(len(y))
    for element in range(len(y)):
        if y[element] != 0:
            new_y[element] = 1
    y = new_y

    fp = fn = 0 # false positives and false negatives
    for i in range(k):

        train_X = np.concatenate((X[:n_test*i], X[n_test*(i+1):]))
        test_X = X[n_test*i:n_test*(i+1)]
        train_y = np.concatenate((y[:n_test*i], y[n_test*(i+1):]))
        test_y = y[n_test*i:n_test*(i+1)]

        start = time.time()
        naive_bayes = GaussianNB()
        naive_bayes.fit(train_X, train_y)
        end = time.time()
        # print "TIME train:", end-start
        time_train += end-start
        
        # Testing
        start = time.time()

        # Check results on cv set
        cv_predict = naive_bayes.predict(test_X)
        decision_boundary = 4e-6
        RMSE = np.sqrt( sum(pow(test_y - cv_predict, 2)) / test_y.size )

        # calculate false positives and false negatives
        for i in range(len(test_y)):
            if test_y[i] == 0 and cv_predict[i] > decision_boundary:
                fp += 1
            elif test_y[i] > 0 and cv_predict[i] <= decision_boundary:
                fn += 1

        end = time.time()
        # print "TIME test:", end-start
        time_test += end-start

    # print "PERFORMANCE\t%d\t%d\t%d\t%d\t%d" % (RMSE, fp, fn, len(y), round(end-start))
    print "TIME train:", time_train
    print "TIME test:",  time_test
    return fp, fn

#
# Train a single model on one train_data_path
#
def train(train_data):

    train_X = train_data[:,0:-1]
    train_Y = np.ravel(train_data[:,[-1]])
    train_Y = np.array([int(x*100) for x in train_Y])

    start = time.time()
    naive_bayes = GaussianNB()
    naive_bayes.fit(train_X, train_Y)
    end = time.time()
    print "TIME train:", end-start

    return naive_bayes

def test(test, naive_bayes):
    start = time.time()
    test_X = test[:,0:-1]
    test_Y = np.ravel(test[:,[-1]])
    cv_predict = naive_bayes.predict(test_X)
    decision_boundary = 4e-6
    RMSE = np.sqrt( sum(pow(test_Y - cv_predict, 2)) / test_Y.size )
    
    # calculate false positives and false negatives
    fp = fn = 0
    for i in range(len(test_Y)):
        if test_Y[i] == 0 and cv_predict[i] > decision_boundary:
            fp += 1
        elif test_Y[i] > 0 and cv_predict[i] <= decision_boundary:
            fn += 1

    end = time.time()
    print "TIME test:", end-start

    if enable_print_predictions:
        for i in range(len(test_Y)):
            print test_Y[i], cv_predict[i]

    print "PERFORMANCE (fp, fn) \t%d\t%d" % (fp, fn)

#
# Train a single model on all train_data_paths, evaluate separately on each of test_data_paths.
#
def train_many_test_many(train, test, test_data_spec):

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
    print "PERFORMANCE\ttest_data_spec\ttest_run\tpiston_param\tdensity_param\trmse\tfp\tfn\tnum_instances\truntime_secs"

    print "############ Starting train_many_test_many"

    # Train Naive Bayes
    train_X = train[:,0:-1]
    train_y = np.ravel(train[:,[-1]])
    # Scale the continuous class values into integers from 0 to 100
    train_y = [int(x*100) for x in train_y]

    start = time.time()
    naive_bayes = GaussianNB()
    naive_bayes.fit(train_X, train_y)
    end = time.time()
    print "TIME train: ", end-start

    # Testing
    start = time.time()

    piston_param = 0
    density_param = 0
    try:
        # (index,test) = get_learning_data_for_run(test_path, test_start_cycle, test_end_cycle, test_sample_freq, decay_window, test_run)
        # print "test data: ", test.shape
        # Check results on cv set
        test_X = test[:,0:-1]
        test_y = np.ravel(test[:,[-1]])
        cv_predict = naive_bayes.predict(test_X)
        #decision_boundary = min(cv_predict)
        decision_boundary = 4e-6
        RMSE = np.sqrt( sum(pow(test_y - cv_predict, 2)) / test_y.size )

        fp = fn = 0
        for i in range(len(test_y)):
            if test_y[i] == 0 and cv_predict[i] > decision_boundary:
                fp += 1
            elif test_y[i] > 0 and cv_predict[i] <= decision_boundary:
                fn += 1

        end = time.time()

        # if enable_print_predictions:
        #     for i in range(len(test_Y)):
        #         print test_Y[i], cv_predict[i]

        # if "piston" in test_path:
        #     piston_offset = test_path.find("piston") + len("piston")
        #     piston_param = int(test_path[piston_offset:piston_offset+3])
        #     density_offset = test_path.find("density") + len("density")
        #     density_param = float(test_path[density_offset:density_offset+4])
        print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, RMSE, fp, fn, len(test_y), round(end-start))
        sys.stdout.flush()
    except:
        end = time.time()
        print "failed"
        print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, 0, 0, 0, 0, round(end-start))

