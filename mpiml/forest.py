import argparse
import numpy as np
import skgarden.mondrian.ensemble as skg
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
parallelism = -1 # note: -1 = number of cores on the system

def _reduce_forest(clf, root=0):
    all_estimators = comm.gather(clf.estimators_, root=root)
    if comm.rank == root:
        super_forest = []
        for forest in all_estimators:
            super_forest.extend(forest)
        clf.estimators_ = super_forest
    return clf

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

class RandomForestRegressor(sk.RandomForestRegressor):
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
                 oob_score=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 warm_start=False
                 ):
        super(RandomForestRegressor, self).__init__(
            n_estimators=_n_estimators_for_forest_size(n_estimators),
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
        )

        debug('will train {} estimators', self.n_estimators)

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

    def partial_fit(self, X, y, classes=None):
        root_info('attempting online training with unsupported model type')
        sys.exit(1)

class MondrianForestRegressor(skg.MondrianForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0):
        super(MondrianForestRegressor, self).__init__(
            n_estimators=_n_estimators_for_forest_size(n_estimators),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

        debug('will train {} estimators', self.n_estimators)

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

    def partial_fit(self, X, y, classes=None):
        super(MondrianForestRegressor, self).partial_fit(X, y)
