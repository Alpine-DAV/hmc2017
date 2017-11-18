#!usr/bin/env python

import cPickle
import glob
import numpy as np
from numpy import isinf, mean, std
import os
import random
import scipy
from scipy.spatial.distance import euclidean
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
import sys
import time
from DataReader.FeatureDataReader import FeatureDataReader
from utils import get_k_fold_data
import config

#
# Train a single model on all train_data_paths, evaluate separately on each of test_data_paths.
#
def train_and_test_k_fold(X, y, k):

    time_train = 0
    time_test = 0
    fp = fn = 0

    n_samples = X.shape[0]
    n_test = n_samples // k

    for i in range(k):

        train_X = np.concatenate((X[:n_test*i], X[n_test*(i+1):]))
        test_X = X[n_test*i:n_test*(i+1)]
        train_y = np.concatenate((y[:n_test*i], y[n_test*(i+1):]))
        test_y = y[n_test*i:n_test*(i+1)]

        start = time.time()
        rand_forest = RandomForestRegressor(n_estimators=config.NumTrees, n_jobs=config.parallelism, random_state=config.rand_seed)
        rand_forest.fit(train_X, train_y)
        end = time.time()
        time_train += end-start

        start = time.time()

        # Check results on cv set
        cv_predict = rand_forest.predict(test_X)
        decision_boundary = 4e-6
        RMSE = np.sqrt( sum(pow(test_y - cv_predict, 2)) / test_y.size )
       
        # calculate false positives and false negatives
        for i in range(len(test_y)):
            if test_y[i] == 0 and cv_predict[i] > decision_boundary:
                fp += 1
            elif test_y[i] > 0 and cv_predict[i] <= decision_boundary:
                fn += 1
        end = time.time()

        time_test += end-start

        if config.enable_print_predictions:
            for i in range(len(test_y)):
                print test_y[i], cv_predict[i]

    print "TIME train:", time_train
    print "TIME test:",  time_test
    print "PERFORMANCE (fp, fn) \t%d\t%d" % (fp, fn)
    # print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, RMSE, fp, fn, len(test_y), round(end-start))
    sys.stdout.flush()
    return rand_forest

#
# Train a single model on all train_data_paths, evaluate separately on each of test_data_paths.
#
def train_and_test_k_fold_serial_merge(X, y, k):

    time_train = 0
    time_test = 0
    fp = fn = 0

    n_samples = X.shape[0]
    n_test = n_samples // k

    for i in range(k):

        train_X = np.concatenate((X[:n_test*i], X[n_test*(i+1):]))
        test_X = X[n_test*i:n_test*(i+1)]
        train_y = np.concatenate((y[:n_test*i], y[n_test*(i+1):]))
        test_y = y[n_test*i:n_test*(i+1)]

        start = time.time()

        # pretend we're running in parallel and merge
        rand_forest = RandomForestRegressor(n_estimators=config.NumTrees, n_jobs=config.parallelism, random_state=config.rand_seed)

        commsize = 8
        trees = []
        for k in range(commsize):
            samps_per_task = train_X.shape[0] // commsize

            min_bound = samps_per_task*k
            if k == commsize - 1:
                max_bound = train_X.shape[0]
            else:
                max_bound = min_bound + samps_per_task

            interval = (train_X[min_bound:max_bound], train_y[min_bound:max_bound])
            rf = RandomForestRegressor(n_estimators=config.NumTrees, n_jobs=config.parallelism, random_state=config.rand_seed)
            rf.fit(train_X[min_bound:max_bound], train_y[min_bound:max_bound])
            rand_forest.estimators_ += rf.estimators_

        end = time.time()
        time_train += end-start

        start = time.time()

        # Check results on cv set
        cv_predict = rand_forest.predict(test_X)
        decision_boundary = 4e-6
        RMSE = np.sqrt( sum(pow(test_y - cv_predict, 2)) / test_y.size )
       
        # calculate false positives and false negatives
        for i in range(len(test_y)):
            if test_y[i] == 0 and cv_predict[i] > decision_boundary:
                fp += 1
            elif test_y[i] > 0 and cv_predict[i] <= decision_boundary:
                fn += 1
        end = time.time()

        time_test += end-start

        if enable_print_predictions:
            for i in range(len(test_y)):
                print test_y[i], cv_predict[i]

    print "TIME train:", time_train
    print "TIME test:",  time_test
    print "PERFORMANCE (fp, fn) \t%d\t%d" % (fp, fn)
    # print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, RMSE, fp, fn, len(test_y), round(end-start))
    sys.stdout.flush()
    return rand_forest
    
#
# Train a single model on all train_data_paths, evaluate separately on each of test_data_paths.
#
# def train_many_test_many(train, test, test_data_spec):

#     test_run = 0

#     if test_data_spec == TEST_ON_FAIL_ONLY:
#         test_start_cycle = 0; test_end_cycle = 0; test_sample_freq = 0 # don't use any "good" examples
#     elif test_data_spec == TEST_ON_FAIL_PLUS_CYCLE_ZERO:
#         test_start_cycle = 0; test_end_cycle = 1; test_sample_freq = 1000 # use only "good" example from first cycle
#     elif test_data_spec == TEST_ON_FAIL_PLUS_CYCLE_MAX:
#         test_start_cycle = 9999999; test_end_cycle = 9999999; test_sample_freq = 1000 # use only "good" example from "max" cycle
#     elif test_data_spec == TEST_ON_TRAIN_SPEC:
#         test_start_cycle = start_cycle; test_end_cycle = end_cycle; test_sample_freq = sample_freq
#     else:
#         sys.err.write("Invalid test_data_spec '%d'. Must be one of: TEST_ON_FAIL_ONLY, TEST_ON_FAIL_PLUS_CYCLE_ZERO, TEST_ON_FAIL_PLUS_CYCLE_MAX\n" % test_data_spec)

#     # print output headers
#     print "PERFORMANCE\ttest_data_spec\ttest_run\tpiston_param\tdensity_param\trmse\tfp\tfn\tnum_instances\truntime_secs"

#     # load pickled model
#     if enable_load_pickled_model:
#         print 'Using pre-trained model from: randomforest.pkl. This may take a couple of minutes...'
#         with open('randomforest.pkl','rb') as f:
#             rand_forest = cPickle.load(f)
#     else:
#         # train = None
#         # start = time.time()
#         # train_list = []
#         # for train_data_path in train_data_paths:
#         #     print train_data_path
#         #     train_next = get_learning_data(train_data_path, start_cycle, end_cycle, sample_freq, decay_window)
#         #     train_list.append(train_next)
#         # train = np.concatenate(train_list, axis=0)
#         # print "training data: " , train.shape

#         # end = time.time()
#         # print "TIME load training data: ", end-start

#         # Train the random forest
#         train_X = train[:,0:-1]
#         train_y = np.ravel(train[:,[-1]])
#         start = time.time()
#         rand_forest = RandomForestRegressor(n_estimators=config.NumTrees, n_jobs=config.parallelism, random_state=config.rand_seed)
#         rand_forest.fit(train_X, train_y)
#         end = time.time()
#         print "TIME train: ", end-start


#         # output feature importance
#         if enable_feature_importance:
#             output_feature_importance(rand_forest, "bubbleShock")
#             # output_feature_importance(rand_forest, train_data_paths[0])

#         # pickle model for future use
#         if enable_save_pickled_model:
#             print "Writing random forest model to file: randomforest.pkl. This may take a couple of minutes..."
#             with open('randomforest.pkl','wb') as f:
#                 cPickle.dump(rand_forest,f)
#             print "Wrote random forest model to file: randomforest.pkl"

#         # for test_path in test_data_paths:

#         start = time.time()

#         piston_param = 0
#         density_param = 0
#         try:
#             # (index,test) = get_learning_data_for_run(test_path, test_start_cycle, test_end_cycle, test_sample_freq, decay_window, test_run)
#             # print "test data: ", test.shape
#             # Check results on cv set
#             test_X = test[:,0:-1]
#             test_y = np.ravel(test[:,[-1]])
#             cv_predict = rand_forest.predict(test_X)
#             #decision_boundary = min(cv_predict)
#             decision_boundary = 4e-6
#             RMSE = np.sqrt( sum(pow(test_y - cv_predict, 2)) / test_y.size )
#             #err = sum(cv_predict - test_Y) / test_Y.size 
#             #pos_indices = [i for i, x in enumerate(test_Y) if x > 0]
#             #neg_indices = [i for i, x in enumerate(test_Y) if x == 0]
#             #err_on_pos = sum(np.array([cv_predict[i] for i in pos_indices]) - np.array([test_Y[i] for i in pos_indices])) / len(pos_indices)
#             #err_on_neg = sum(np.array([cv_predict[i] for i in neg_indices]) - np.array([test_Y[i] for i in neg_indices])) / len(neg_indices)

#             # calculate false positives and false negatives
#             fp = fn = 0
#             for i in range(len(test_y)):
#                 if test_y[i] == 0 and cv_predict[i] > decision_boundary:
#                     fp += 1
#                 elif test_y[i] > 0 and cv_predict[i] <= decision_boundary:
#                     fn += 1
#             end = time.time()

#             if enable_print_predictions:
#                 for i in range(len(test_y)):
#                     print test_y[i], cv_predict[i]
            
#             # if "piston" in test_path:
#             #     piston_offset = test_path.find("piston") + len("piston")
#             #     piston_param = int(test_path[piston_offset:piston_offset+3])
#             #     density_offset = test_path.find("density") + len("density")
#             #     density_param = float(test_path[density_offset:density_offset+4])

#             # print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, RMSE, fp, fn, len(test_y), round(end-start))
#             # sys.stdout.flush()
#             return fp, fn
#         except:
#             print "failed"
#             end = time.time()
#             # print "PERFORMANCE\t%d\t%d\t%d\t%.2f\t%.15f\t%d\t%d\t%d\t%d" % (test_data_spec, test_run, piston_param, density_param, 0, 0, 0, 0, round(end-start))
