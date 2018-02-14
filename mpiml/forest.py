import argparse
import skgarden.mondrian.ensemble as skg
import skgarden.mondrian.tree as skgtree
import sklearn.ensemble as sk
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import r2_score
from sklearn.tree._tree import DTYPE, DOUBLE
import numpy as np

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

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

    def partial_fit(self, X, y, classes=None):
        root_info('attempting online training with unsupported model type')
        sys.exit(1)

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

class MondrianForestRegressor(skg.MondrianForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 oob_score=False):
        super(MondrianForestRegressor, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        self.oob_score = oob_score

    def reduce(self, root=0):
        return _reduce_forest(self, root=root)

    def partial_fit(self, X, y, classes=None):
        # super(MondrianForestRegressor, self).partial_fit(X, y)
        print("running partial_fit",comm.rank)
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        random_state = check_random_state(self.random_state)

        # Wipe out estimators if partial_fit is called after fit.
        first_call = not hasattr(self, "first_")
        if first_call:
            self.first_ = True

        if isinstance(self, ClassifierMixin):
            if first_call:
                if classes is None:
                    classes = LabelEncoder().fit(y).classes_

                self.classes_ = classes
                self.n_classes_ = len(self.classes_)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        self.n_outputs_ = 1

        # Initialize estimators at first call to partial_fit.
        if first_call:
            # Check estimators
            self._validate_estimator()
            self.estimators_ = []

            for _ in range(self.n_estimators):
                tree = self._make_estimator(append=False, random_state=random_state)
                self.estimators_.append(tree)

        # XXX: Switch to threading backend when GIL is released.
        if isinstance(self, ClassifierMixin):
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y, classes) for t in self.estimators_)
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y) for t in self.estimators_)

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    def get_oob_scores(self):
        result = ""
        for tree in self.estimators_:
            result += "{tree_}".format(tree_=tree.tree_)
            result += "\n"
        return result


    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

            oob_error = r2_score(y[unsampled_indices, :], p_estimator)
            
            # if estimator.oob_error:
            estimator.oob_error = oob_error

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


class MondrianTreeRegressor(skgtree.MondrianTreeRegressor):
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 random_state=None,
                 oob_error=True
                 ):
        super(MondrianTreeRegressor, self).__init__(
            criterion="mse",
            splitter="mondrian",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            oob_error=oob_error
        )
