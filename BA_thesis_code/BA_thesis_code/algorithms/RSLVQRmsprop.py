import numpy
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ

class RSLVQRMSprop(object):
    """description of class"""
    def _optimize(self, x, y, random_state):
        """Implementation of RMSprop"""
        n_data, n_dim = X.shape
        nb_prototypes = self.c_w_.size
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = X[i]
            c_xi = y[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])
                if self.c_w_[j] == c_xi:
                    gradient = (self._p(j, xi, prototypes=self.w_,
                    y=c_xi) - self._p(j, xi, prototypes=self.w_))* d
                else:
                    gradient = - self._p(j, xi, prototypes=self.w_)* d

                # Accumulate gradient
                self.squared_mean_gradient[j] = 0.9 * self.squared_mean_gradient[j] + 0.1 * gradient ** 2

                # Update Prototype
                self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient

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