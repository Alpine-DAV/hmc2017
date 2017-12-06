import mpiml.datasets as datasets
import numpy as np
from sklearn.utils.testing import *

def assert_array_not_equal(x, y):
    assert_raises(AssertionError, assert_array_equal, x, y)

def test_shuffle_data():
    '''
    Should permute order but keep same elements
    '''
    X = range(10)
    y = range(9, -1, -1)
    datasets.shuffle_data(X, y)

    assert_array_equal(sorted(X), range(10))
    assert_array_not_equal(X, range(10))
    assert_array_equal(sorted(y), range(10))
    assert_array_not_equal(y, range(9, -1, -1))

    for ft, cls in zip(X, y):
        assert_equal(cls, 9-ft)

def test_discretize():
    v = np.array([0, 0.1, 0.9, 1])
    assert_array_equal([0, 1, 1, 1], datasets.discretize(v))
