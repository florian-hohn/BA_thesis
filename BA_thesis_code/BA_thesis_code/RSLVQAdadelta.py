import numpy
import skmultiflow.prototype
    
class RSLVQAdadelta(skmultiflow.prototype.robust_soft_learning_vector_quantization):
    """Inherits from the scikit-multiflow framework implementation 
    of the rslvq algortihm and modifies it to an implementation where the decition is done with the adadelta"""

    def _optimize(self, X, y, random_state):
        if(self.gradient_descent=='Adadelta'):
            """Implementation of Adadelta"""
            n_data, n_dim = X.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = X[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])

                if self.c_w_[j] == c_xi:
                    gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                    self._p(j, xi, prototypes=self.w_)) * d
                else:
                    gradient = - self._p(j, xi, prototypes=self.w_) * d

                # Accumulate gradient
                self.squared_mean_gradient[j] = self.decay_rate *self.squared_mean_gradient[j] + \
                (1 - self.decay_rate) * gradient ** 2

                # Compute update/step
                step = ((self.squared_mean_step[j] + self.epsilon) / \
                (self.squared_mean_gradient[j] + self.epsilon)) **0.5 * gradient

                # Accumulate updates
                self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                (1 - self.decay_rate) * step ** 2

                # Attract/Distract prototype to/from data point
                self.w_[j] += step

