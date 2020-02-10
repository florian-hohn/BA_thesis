import math
import numpy as np
from sklearn.utils import validation
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from skmultiflow.core.base import ClassifierMixin, BaseSKMObject

class BaseLVQ(object):
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Fit the LVQ model to the given training data and parameters using
        gradient ascent.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the class labels of all samples in X
        classes : numpy.ndarray, optional (default=None)
            Contains all possible/known class labels. Usage varies depending
            on the learning method.
        sample_weight : Not used.

        Returns
        --------
        self
        """
        if set(unique_labels(y)).issubset(set(self.classes_)) or self.initial_fit is True:
            X, y = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all classes in first call of fit/partial_fit'.format(y))

        self._optimize(X, y)
        return self

    def _optimize(self, X, y):
        nb_prototypes = self.c_w_.size

        n_data, n_dim = X.shape
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = X[i]
            c_xi = int(y[i])
            best_euclid_corr = np.inf
            best_euclid_incorr = np.inf

            # find nearest correct and nearest wrong prototype
            for j in range(prototypes.shape[0]):
                if self.c_w_[j] == c_xi:
                    eucl_dis = euclidean_distances(xi.reshape(1, xi.size),
                                                   prototypes[j]
                                                   .reshape(1, prototypes[j]
                                                   .size))
                    if eucl_dis < best_euclid_corr:
                        best_euclid_corr = eucl_dis
                        corr_index = j
                else:
                    eucl_dis = euclidean_distances(xi.reshape(1, xi.size),
                                                   prototypes[j]
                                                   .reshape(1, prototypes[j]
                                                   .size))
                    if eucl_dis < best_euclid_incorr:
                        best_euclid_incorr = eucl_dis
                        incorr_index = j

            # Update nearest wrong prototype and nearest correct prototype
            # if correct prototype isn't the nearest
            if best_euclid_incorr < best_euclid_corr:
                self._update_prototype(j=corr_index, c_xi=c_xi, xi=xi,
                                       prototypes=prototypes)
                self._update_prototype(j=incorr_index, c_xi=c_xi, xi=xi,
                                       prototypes=prototypes)

    def predict(self, X):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples)
            Returns predicted values.
        """
        return np.array([self.c_w_[np.array([self._costf(xi, p) for p in self.w_]).argmax()] for xi in X])

    def _costf(self, x, w):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
        return - d / (2 * self.sigma)

    def _p(self, j, e, prototypes, y=None):
        if y is None:
            fs = [self._costf(e, w) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i]) for i in
                  range(prototypes.shape[0]) if
                  self.c_w_[i] == y]

        fs_max = np.amax(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j]) - fs_max) / s
        return o

    @property
    def prototypes(self):
        """The prototypes"""
        return self.w_