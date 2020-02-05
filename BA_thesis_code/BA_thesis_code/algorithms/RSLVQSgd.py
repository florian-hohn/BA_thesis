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