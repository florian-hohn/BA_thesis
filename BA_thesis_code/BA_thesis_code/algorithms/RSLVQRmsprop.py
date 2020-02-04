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