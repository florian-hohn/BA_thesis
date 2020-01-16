import numpy
import skmultiflow.prototype


class RSLVQSgd(skmultiflow.prototype.robust_soft_learning_vector_quantization):
    """Inherits from the scikit-multiflow framework implementation 
    of the rslvq algortihm and modifies it to an implementation where the decition is done with the sgd"""

    def _optimize(self, X, y, random_state):
        """Implementation of Stochastical Gradient Descent"""
        n_data, n_dim = X.shape5 
        nb_prototypes = self.c_w_.size
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = X[i]
            c_xi = y[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j]) 
                c = 1 / self.sigma
                if self.c_w_[j] == c_xi:
                    # Attract prototype to data point
                    self.w_[j] += c * (self._p(j, xi,prototypes=self.w_, y=c_xi) -self._p(j, xi, prototypes=self.w_)) * d
                else:
                    # Move prototype away from data point
                    self.w_[j] -= c * self._p(j, xi,prototypes=self.w_) * d

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
         if prototypes is None:
            prototypes = self.w_
            if y is None:
                fs = [self._costf(e, w, **kwargs) for w in prototypes]
            else:
                fs = [self._costf(e, prototypes[i], **kwargs) for i in
                     range(prototypes.shape[0]) if
                     self.c_w_[i] == y]

            fs_max = np.max(fs)
            s = sum([np.math.exp(f - fs_max) for f in fs])
            o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
            return o

    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
        return -d / (2 * self.sigma)