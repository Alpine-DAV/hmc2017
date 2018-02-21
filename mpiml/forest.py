import argparse
import numpy as np
from operator import attrgetter
import skgarden.mondrian.ensemble as skg
import sklearn.base as sk_base
import sklearn.ensemble as sk
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import r2_score
from sklearn.tree._tree import DTYPE, DOUBLE
import numpy as np

import sys

from utils import *
from config import comm
import config
import csv

__all__ = ["RandomForestRegressor"
          ,"MondrianForestRegressor"
          ]

def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices

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


class SubForestMixin:

    def n_estimators(self, forest_size):
        return forest_size

    def reduce(self, forest_size, root):
        # root_debug(self.oob_score_)
        # sorted_estimators = sorted(self.estimators_, key=attrgetter('oob_score_'))

        # Get best X% of estimators by oob score
        # self.estimators_ = _gather_estimators(
        #     sorted_estimators[:_n_estimators_for_forest_size(forest_size)/4])

        ## Get last X% of estimators by oob score
        # self.estimators_ = _gather_estimators(
        #     sorted_estimators[(len(sorted_estimators)-_n_estimators_for_forest_size(forest_size)/4):])
        
        ## Get random X% of estimators
        # self.estimators_ = _gather_estimators(self.estimators_[:_n_estimators_for_forest_size(forest_size)])
        # self.estimators_ = _gather_estimators(self.estimators_[-1:])
        return self

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
                 bootstrap=False,
                 oob_score=False,
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
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
        )

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        root_info('attempting online training with unsupported model type')
        sys.exit(1)

class MondrianForestBase(skg.MondrianForestRegressor, SubForestMixin):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 oob_score=False):
        super(MondrianForestBase, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.oob_score = oob_score

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        super(MondrianForestBase, self).partial_fit(X, y)

        # X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        # random_state = check_random_state(self.random_state)

        # # Wipe out estimators if partial_fit is called after fit.
        # first_call = not hasattr(self, "first_")
        # if first_call:
        #     self.first_ = True

        # if isinstance(self, ClassifierMixin):
        #     if first_call:
        #         if classes is None:
        #             classes = LabelEncoder().fit(y).classes_

        #         self.classes_ = classes
        #         self.n_classes_ = len(self.classes_)

        # # Remap output
        # n_samples, self.n_features_ = X.shape

        # y = np.atleast_1d(y)
        # if y.ndim == 2 and y.shape[1] == 1:
        #     warn("A column-vector y was passed when a 1d array was"
        #          " expected. Please change the shape of y to "
        #          "(n_samples,), for example using ravel().",
        #          DataConversionWarning, stacklevel=2)

        # self.n_outputs_ = 1

        # # Initialize estimators at first call to partial_fit.
        # if first_call:
        #     # Check estimators
        #     self._validate_estimator()
        #     self.estimators_ = []

        #     for _ in range(self.n_estimators):
        #         tree = self._make_estimator(append=False, random_state=random_state)
        #         self.estimators_.append(tree)

        # # XXX: Switch to threading backend when GIL is released.
        # if isinstance(self, ClassifierMixin):
        #     self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
        #         delayed(_single_tree_pfit)(t, X, y, classes) for t in self.estimators_)
        # else:
        #     self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
        #         delayed(_single_tree_pfit)(t, X, y) for t in self.estimators_)

        # if self.oob_score:
        #     self._set_oob_score(X, y)

        # return self

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        # for computing variance in oob scores
        all_oob_errors = []

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

            if p_estimator.size != 0:
                oob_error = r2_score(y[unsampled_indices, :], p_estimator)
                all_oob_errors.append(oob_error) # compute variance
            
                # Set oob score of individual trees
                estimator.oob_score_ = oob_error


        ### CODE FOR OUTPUTTING OOB ERRORS TO A CSV FILE
        # variance = np.var(np.array(all_oob_errors))
        # if comm.rank == 0:
        #     root_info("{}".format(all_oob_errors))
        #     fname = "oob_error_rank0_density1.csv"
        #     with open(fname, 'ab') as f:
        #         np.savetxt(f, np.array([all_oob_errors]).T, delimiter=",")


        if (n_predictions == 0).any():
            root_info("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_


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
MondrianForestRegressor = _forest_regressor(MondrianForestBase, SubForestMixin)
