import argparse
import numpy as np
from operator import attrgetter
import skgarden.mondrian.ensemble as skg
import sklearn.base as sk_base
import sklearn.ensemble as sk

import sys

from utils import *
from config import comm
import config

__all__ = ["RandomForestRegressor"
          ,"MondrianForestRegressor"
          ]

rand_seed = 0
NumTrees = 10
parallelism = 1 # note: -1 = number of cores on the system

# Return the number of estimators that the calling task should train if the superforest should
# contain the given number
def _n_estimators_for_forest_size(forest_size):
    if running_in_mpi():
        if forest_size < comm.size:
            raise ValueError(
                'must train at least 1 tree per task ({} < {})'.format(forest_size, comm.size))

        partition = map(int, np.linspace(0, forest_size, comm.size + 1))
        return partition[comm.rank + 1] - partition[comm.rank]
    else:
        return forest_size

def _gather_estimators(estimators, root=0):
    all_estimators = comm.gather(estimators, root=root)
    if comm.rank == root:
        return [tree for trees in all_estimators for tree in trees]

class SuperForestMixin:

    def n_estimators(self, forest_size):
        return _n_estimators_for_forest_size(forest_size)

    def reduce(self, forest_size, root):
        self.estimators_ = _gather_estimators(self.estimators_)
        return self

# TODO Get oob score for individual trees
# class SubForestMixin:

#     def n_estimators(self, forest_size):
#         return forest_size

#     def reduce(self, forest_size, root):
#         root_info(self.oob_score_)
#         sorted_estimators = sorted(self.estimators_, key=attrgetter('oob_score_'))
#         self.estimators_ = _gather_estimators(
#             sorted_estimators[:_n_estimators_for_forest_size(forest_size)])
#         return self

class RandomForestBase(sk.RandomForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 warm_start=False
                 ):
        super(RandomForestBase, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
        )

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        root_info('attempting online training with unsupported model type')
        sys.exit(1)

class MondrianForestBase(skg.MondrianForestRegressor, SuperForestMixin):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0):
        super(MondrianForestBase, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        super(MondrianForestBase, self).partial_fit(X, y)

# Create a forest regressor class combining a base forest class with mixin providing merging
# behavior
def _forest_regressor(base, merging_mixin):

    class ForestRegressor(base, merging_mixin):

        def __init__(self, forest_size=config.NumTrees, *args, **kwargs):
            super(ForestRegressor, self).__init__(
                *args, n_estimators=self.n_estimators(forest_size), **kwargs)
            self.forest_size_ = forest_size

        def reduce(self, root=0):
            return super(ForestRegressor, self).reduce(self.forest_size_, root)

    ForestRegressor.__name__ = base.__name__ + '_' + merging_mixin.__name__
    return ForestRegressor

RandomForestRegressor = _forest_regressor(RandomForestBase, SuperForestMixin)
MondrianForestRegressor = _forest_regressor(MondrianForestBase, SuperForestMixin)
