import cPickle
import numpy as np
from operator import attrgetter
import skgarden.mondrian.ensemble as skg
import skgarden.mondrian.tree.tree as skt
import sklearn.base as sk_base
import sklearn.ensemble as sk
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import r2_score
from sklearn.tree._tree import DTYPE, DOUBLE
import zlib

import sys

from utils import *
from config import comm
import config
import csv

IGNORE_VAL = 999999
__all__ = ["RandomForestRegressor"
          ,"MondrianForestRegressor"
          ,"MondrianForestPickleRegressor"
          ,"SizeUpMondrianForestRegressor"
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

def _n_estimators_for_forest_size(forest_size):
    """
    Return the number of estimators that the calling task should train if the superforest should
    contain the given number
    """
    if running_in_mpi():
        if forest_size < comm.size:
            raise ValueError(
                'must train at least 1 tree per task ({} < {})'.format(forest_size, comm.size))

        partition = map(int, np.linspace(0, forest_size, comm.size + 1))
        return partition[comm.rank + 1] - partition[comm.rank]
    else:
        return forest_size

def _gather_estimators(estimators, send_to, recv_from, root=0, send_to_all=True):
    """
    Gathers the estimators of all processes either at the root process, or if send_to_all
    is set to true, gathers all of the estimators at every process.
    """
    if send_to_all:
        for proc in range(comm.size):
            n_estimators = comm.bcast(len(estimators), root=proc)
            if comm.rank == proc:
                for e in estimators:
                    send_to(IGNORE_VAL, e, send_to_all=True)
            else:
                estimators.extend([recv_from(proc) for _ in range(n_estimators)])
            root_info("finished process: {}".format(proc))
    else:
        if comm.rank == root:
            for peer in range(comm.size):
                if peer == root: continue

                n_estimators = comm.recv(source=peer)
                estimators.extend([recv_from(peer) for _ in range(n_estimators)])
        else:
            comm.send(len(estimators), root)
            for e in estimators:
                send_to(root, e)

    return estimators

class SuperForestMixin:
    
    def n_estimators(self, forest_size):
        return _n_estimators_for_forest_size(forest_size)

    def reduce(self, forest_size, root, send_to_all=False):
        self.estimators_ = _gather_estimators(
            self.estimators_, self.send_estimator, self.receive_estimator,
            root=root, send_to_all=send_to_all)
        return self

class SizeUpSuperForestMixin:

    def n_estimators(self, forest_size):
        return forest_size

    def reduce(self, forest_size, root, send_to_all=False):
        self.estimators_ = _gather_estimators(
            self.estimators_, self.send_estimator, self.receive_estimator,
            root=root, send_to_all=send_to_all)
        return self

class SubForestMixin:

    def n_estimators(self, forest_size):
        return forest_size

    # TODO: send_to_all currently has no effect on SubForestMixin
    def reduce(self, forest_size, root, send_to_all=False):
        root_debug(self.oob_score_)
        sorted_estimators = sorted(self.estimators_, key=attrgetter('oob_score_'))

        # Get best X% of estimators by oob score
        # self.estimators_ = _gather_estimators(
        #     sorted_estimators[:_n_estimators_for_forest_size(forest_size)/4])

        ## Get last X% of estimators by oob score
        # self.estimators_ = _gather_estimators(
        #     sorted_estimators[(len(sorted_estimators)-_n_estimators_for_forest_size(forest_size)/4):])

        ## Get random X% of estimators
        # self.estimators_ = _gather_estimators(self.estimators_[:_n_estimators_for_forest_size(forest_size)])
        self.estimators_ = _gather_estimators(
            self.estimators_[:forest_size], self.send_estimator, self.receive_estimator,
            root=root, send_to_all=False)
        return self

class RandomForestBase(sk.RandomForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=1000,
                 min_samples_leaf=500,
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

    def send_estimator(self, peer, est, **kwargs):
        comm.send(est, peer)

    def receive_estimator(self, peer):
        return comm.recv(source=peer)

class MondrianForestBase(skg.MondrianForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=1000,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 oob_score=False,
                 compression=0):
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
        self.compression_ = compression

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        super(MondrianForestBase, self).partial_fit(X, y)

        if self.oob_score:
            self._set_oob_score(X, y)


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
                oob_error = r2_score(y[unsampled_indices], p_estimator)
                # all_oob_errors.append(oob_error) # compute variance

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

        # UNCOMMENT LATER - need to fix for 1d array
        # for k in range(self.n_outputs_):
        #     self.oob_score_ += r2_score(y[:, k],
        #                                 predictions[:, k])

        self.oob_score_ /= self.n_outputs_

    def send_estimator(self, peer, est, send_to_all=False):
        if send_to_all:
            skt.mpi_send(comm, peer, est, compression=self.compression_, send_to_all=True)
        else:
            # HACK to not include the send_to_all parameter (even send_to_all=False) on Surface,
            # where send_to_all is not yet recognized by skgarden
            skt.mpi_send(comm, peer, est, compression=self.compression_)

    def receive_estimator(self, peer):
        return skt.mpi_recv_regressor(comm, peer, self.n_features_, self.n_outputs_)

class MondrianForestPickleBase(skg.MondrianForestRegressor):
    def __init__(self,
                 n_estimators=config.NumTrees,
                 max_depth=None,
                 min_samples_split=1000,
                 bootstrap=False,
                 n_jobs=config.parallelism,
                 random_state=config.rand_seed,
                 verbose=0,
                 oob_score=False,
                 compression=0):
        super(MondrianForestPickleBase, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

        self.compression_ = compression

        debug('will train {} estimators', self.n_estimators)

    def partial_fit(self, X, y, classes=None):
        super(MondrianForestPickleBase, self).partial_fit(X, y)

    def send_estimator(self, peer, est, **kwargs):
        pkl = cPickle.dumps(est, cPickle.HIGHEST_PROTOCOL)
        if self.compression_ > 0:
            pkl = zlib.compress(pkl, self.compression_)
        buf = np.frombuffer(pkl, dtype=np.dtype('b'))
        nbytes = buf.shape[0]

        comm.send(nbytes, peer)
        comm.Send(buf, peer)

    def receive_estimator(self, peer):
        nbytes = comm.recv(source=peer)
        buf = np.empty(nbytes, dtype=np.dtype('b'))

        comm.Recv(buf, source=peer)

        pkl = buf.tobytes()
        if self.compression_ > 0:
            pkl = zlib.decompress(pkl)

        return cPickle.loads(pkl)

# Create a forest regressor class combining a base forest class with mixin providing merging
# behavior
def _forest_regressor(base, merging_mixin):

    class ForestRegressor(base, merging_mixin):

        def __init__(self, forest_size=config.NumTrees, *args, **kwargs):
            super(ForestRegressor, self).__init__(
                *args, n_estimators=self.n_estimators(forest_size), **kwargs)
            self.forest_size_ = forest_size

        def reduce(self, root=0, send_to_all=False):
            return super(ForestRegressor, self).reduce(self.forest_size_, root, send_to_all)

    ForestRegressor.__name__ = base.__name__ + '_' + merging_mixin.__name__
    return ForestRegressor

RandomForestRegressor = _forest_regressor(RandomForestBase, SuperForestMixin)
MondrianForestRegressor = _forest_regressor(MondrianForestBase, SuperForestMixin)
MondrianForestPickleRegressor = _forest_regressor(MondrianForestPickleBase, SuperForestMixin)

# Like a Mondrian superforest, but the size of the superforest scales up with the number of tasks,
# so that each task trains a fixed number of trees
SizeUpMondrianForestRegressor = _forest_regressor(MondrianForestBase, SizeUpSuperForestMixin)
