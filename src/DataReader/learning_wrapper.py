#!usr/bin/env python
import testing_naivebayes as bayes
import testing_randomforest as rand_forest
import numpy as np
import time

def wrapper(ML_type, k, data_path):
    """ input: type of machine learning, type of test, amount to test, training path, test path
        output: trains ML_type on training data and tests it on testing data
    """
    train, test = data_split(data_path,k,bayes.TEST_ON_TRAIN_SPEC)
    if ML_type == "naive bayes":
        print "############ Training using Naive Bayes ############"
        bayes.train_many_test_many(train, test, bayes.TEST_ON_TRAIN_SPEC)
        print 
        print
    elif ML_type == "random forest":
        print "############ Training using Random Forest ############"
        rand_forest.train_many_test_many(train, test, bayes.TEST_ON_TRAIN_SPEC)
        print 
        print
    else:
        raise Exception('Machine learning algorithm not recognized')

def data_split(data_path, k, test_data_spec):
    train = None
    start = time.time()
    train_list = []

    for train_data_path in data_path:
        print train_data_path
        train_next = bayes.get_learning_data(train_data_path, bayes.start_cycle, bayes.end_cycle, bayes.sample_freq, bayes.decay_window)
        train_list.append(train_next)
    all_data = np.concatenate(train_list, axis=0)
    print "training data: " , all_data.shape

    end = time.time()
    print "TIME load training data: ", end-start
    # Train the random forest
    k_percent = int(len(all_data)*k)
    train = all_data[:k_percent,:]
    test = all_data[k_percent:,:]
    return train, test


if __name__ == '__main__':
    wrapper('naive bayes', 0.9, ['bubbleShock'])
    wrapper('random forest', 0.9, ['bubbleShock'])
