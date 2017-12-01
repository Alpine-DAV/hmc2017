import mpiml.utils as utils
import numpy as np
from sklearn.utils.testing import *

def test_accuracy():
    actual    = np.array([1, 2, 3, 4])
    predicted = np.array([1, 3, 2, 4])
    assert_almost_equal(0.5, utils.accuracy(actual, predicted), 2)

def test_num_errors_zero_nonzero():
    actual      = np.array([0, 0, 0.1, 0.1, 0.1])
    predicted   = np.array([0, 0.1, 0, 0.1, 0])
    fp, fn = utils.num_errors(actual, predicted)
    assert_equal((1, 2), (fp, fn))

def test_num_errors_half_threshhold():
    actual      = np.array([0.4, 0.4, 0.6, 0.6, 0.6])
    predicted   = np.array([0.4, 0.6, 0.4, 0.6, 0.4])
    fp, fn = utils.num_errors(actual, predicted, threshold=0.5)
    assert_equal((1, 2), (fp, fn))

def test_num_classes_zero_nonzero():
    labels = np.array([0, 1e-8, 0, 0.1])
    npos, nneg = utils.num_classes(labels)
    assert_equal((1, 3), (npos, nneg))

def test_num_classes_half_threshold():
    labels = np.array([0.4, 0.5, 0.5, 0.6])
    npos, nneg = utils.num_classes(labels, threshold=0.5)
    assert_equal((1, 3), (npos, nneg))

def test_get_kfold_data():
    X = np.array(range(10))
    y = np.array(range(9, -1, -1))
    for i, (train_X, test_X, train_y, test_y) in enumerate(utils.get_k_fold_data(X, y, k=10)):
        assert_array_equal(test_y, [9-i])
        assert_array_equal(test_X, [i])
        assert_array_equal(train_y, [cls for cls in range(9, -1, -1) if cls != 9-i])
        assert_array_equal(train_X, [ft for ft in range(10) if ft != i])

def test_get_kfold_data_doesnt_divide():
    X = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    for i, (train_X, test_X, train_y, test_y) in enumerate(utils.get_k_fold_data(X, y, k=2)):
        if i == 0:
            assert_array_equal(train_X, [3])
            assert_array_equal(test_X, [1, 2])
            assert_array_equal(train_y, [6])
            assert_array_equal(test_y, [4, 5])
        elif i == 1:
            assert_array_equal(train_X, [1, 2])
            assert_array_equal(test_X, [3])
            assert_array_equal(train_y, [4, 5])
            assert_array_equal(test_y, [6])
        else:
            fail("Expected two splits")

class MockComm:
    def __init__(self, size):
        self.size = size
        self.rank = 0

def test_get_mpi_task_data():
    X = np.array(range(10))
    y = np.array(range(9, -1, -1))
    comm = MockComm(10)
    for i in range(10):
        comm.rank = i
        task_X, task_y = utils.get_mpi_task_data(X, y, comm=comm)
        assert_array_equal(task_X, [i])
        assert_array_equal(task_y, [9-i])

def test_get_mpi_task_data_doesnt_divide():
    X = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    comm = MockComm(2)

    comm.rank = 0
    task_X, task_y = utils.get_mpi_task_data(X, y, comm=comm)
    assert_array_equal(task_X, [1, 2])
    assert_array_equal(task_y, [4, 5])

    comm.rank = 1
    task_X, task_y = utils.get_mpi_task_data(X, y, comm=comm)
    assert_array_equal(task_X, [3])
    assert_array_equal(task_y, [6])
