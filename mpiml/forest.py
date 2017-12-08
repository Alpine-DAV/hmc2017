#! /usr/bin/env python

import argparse
import skgarden.mondrian.ensemble as skg
import sklearn.ensemble as sk
from datasets import prepare_dataset

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
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressor, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
        )

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

config.register_model('rf', RandomForestRegressor, forest=True)

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
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

config.register_model('mf', MondrianForestRegressor, forest=True)
