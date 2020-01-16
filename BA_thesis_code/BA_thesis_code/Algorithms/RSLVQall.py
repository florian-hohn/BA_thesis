import numpy as np
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ

class RSLVQall(RSLVQ):
    """a class that contains all the different implementations of the RSLVQ algorithm"""
    def _optimize(self, X, y, random_state):
        if(self.gradient_descent=="sgd"):
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

        elif(self.gradient_descent=="adadelta"):
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

        elif(self.gradient_descent=="rmsdrop"):
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