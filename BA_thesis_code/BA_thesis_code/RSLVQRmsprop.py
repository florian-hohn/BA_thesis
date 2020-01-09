import numpy
import skmultiflow.prototype

class RSLVQRmsprop(skmultiflow.prototype.robust_soft_learning_vector_quantization):
    """description of class"""
    def _optimize(self, x, y, random_state):
        """Implementation of Adaptive RMSprop"""
        n_data, n_dim = x.shape
        nb_prototypes = self.c_w_.size
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = x[i]
            c_xi = y[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])

                if self.c_w_[j] == c_xi:
                    gradient = (self._p(j, xi, prototypes=self.w_,
                                y=c_xi) - self._p(j, xi, prototypes=self.w_))
                else:
                    gradient = self._p(j, xi, prototypes=self.w_)

                # calc adaptive decay rate
                dec_rate = np.minimum(np.absolute(self._costf(j=j,
                                                                x=gradient**2,
                                                                w=self.squared_mean_gradient[j])), 0.9)

                self.decay_rate[j] = 1.0 - dec_rate

                # Accumulate gradient
                self.squared_mean_gradient[j] = self.decay_rate[j] * self.squared_mean_gradient[j] + \
                (1 - self.decay_rate[j]) * gradient ** 2

                # Update Prototype
                if self.c_w_[j] == c_xi:
                    self.w_[j] += (self.learning_rate /
                        ((self.squared_mean_gradient[j] + \
                        self.epsilon) ** 0.5)) * gradient * d
                else:
                    self.w_[j] -= (self.learning_rate /
                        ((self.squared_mean_gradient[j] + \
                        self.epsilon) ** 0.5)) * gradient * d