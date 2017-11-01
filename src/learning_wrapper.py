#!usr/bin/env python
import testing_naivebayes as bayes
import testing_randomforest as rand_forest
import numpy as np
import time

def wrapper(ML_type, k, data_path):
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
        description='Train and test a naive Bayes classifier using the sklearn iris dataset')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--online', action='store_true', help='train in online mode')
    parser.add_argument('-k', type=int, help='number for k-fold')
    parser.add_argument('-nb', action='store_true', help='train using naive bayes')
    parser.add_argument('-rf', action='store_true', help='train using random forest')
    parser.add_argument('-nbp', action='store_true', help='train using parallel naive bayes')
    args = parser.parse_args()

    verbose    = args.verbose
    use_online = args.online
    model_nb   = args.nb
    model_rf   = args.rf
    model_nbp  = args.nbp

    if args.k:
        k = int(args.k)
    else:
        k = 10

    bubbleShock = get_bubbleshock()
    X = bubbleShock[:,0:-1]
    y = np.ravel(bubbleShock[:,[-1]])


    discretized_y = np.zeros(len(y))
    for element in range(len(y)):
        if y[element] != 0:
            discretized_y[element] = 1

    # if model_nb:
    #     wrapper('naive bayes', k, ['bubbleShock'], verbose=verbose)
    # if model_rf:
    #     wrapper('random forest', k, ['bubbleShock'], verbose=verbose)
    # if model_nbp:
    use_mpi = 'MPICH_INTERFACE_HOSTNAME' in os.environ
    use_online = False

    if use_mpi and comm.rank == 0:
        print('will train using MPI')
    if use_online and comm.rank == 0:
        print('will train in online mode')
    
    y = discretized_y

    fp, fn = nbmpi.train_and_test_k_fold(X, y, k=k, verbose=False, use_online=use_online, use_mpi=use_mpi)

    if comm.rank == 0:
        print "PERFORMANCE\t%d\t%d" % (fp, fn)
