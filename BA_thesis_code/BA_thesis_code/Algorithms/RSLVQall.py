import math
import numpy as np
from sklearn.utils import validation
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from skmultiflow.core.base import ClassifierMixin, BaseSKMObject
from .base_lvq import BaseLVQ

class RSLVQall(BaseLVQ):
    """a class that contains all the different implementations of the RSLVQ algorithm"""
    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=None, gradient_descent='sgd',
                 gamma=0.9, decay_rate = 0.9, learning_rate = 0.001,
                 beta_1 = 0.9, beta_2 = 0.999):
        self.sigma = sigma
        self.random_state = random_state
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gradient_descent = gradient_descent
        

        if sigma <= 0:
            raise ValueError('Sigma must be greater than 0')
        if prototypes_per_class <= 0:
            raise ValueError('Prototypes per class must be more than 0')
        if gamma >= 1 or gamma < 0:
            raise ValueError('Decay rate gamma has to be between 0 and less than 1')
        if decay_rate >= 1.0 or decay_rate <= 0:
            raise ValueError('Decay rate must be greater than 0 and less than 1')
        allowed_gradient_optimizers = ['sgd','adadelta','rmsprop','rmspropada']

        if gradient_descent not in allowed_gradient_optimizers:
            raise ValueError('{} is not a valid gradient optimizer, please use one of {}'.format(gradient_descent, allowed_gradient_optimizers))

        if self.gradient_descent == 'adadelta':
            self._update_prototype = self._update_prototype_adadelta
        elif self.gradient_descent == 'rmsprop':
            self._update_prototype = self._update_prototype_rmsprop
        elif self.gradient_descent == 'rmspropada':
            self._update_prototype = self._update_prototype_rmspropada
        elif self.gradient_descent == 'adam':
            self._update_prototype = self._update_prototype_adam
        else:
            self.learning_rate = 1 / sigma
            self._update_prototype = self._update_prototype_sgd

    def _update_prototype_sgd(self, j, xi, c_xi, prototypes):
        """SGD"""
        d = xi - prototypes[j]

        if self.c_w_[j] == c_xi:
            # Attract prototype to data point
            self.w_[j] += self.learning_rate * (self._p(j, xi, prototypes=self.w_, y=c_xi) -  self._p(j, xi, prototypes=self.w_)) * d
        else:
            # Distance prototype from data point
            self.w_[j] -= self.learning_rate * self._p(j, xi, prototypes=self.w_) * d

    def _update_prototype_adadelta(self, j, c_xi, xi, prototypes):
        """Adadelta"""
        d = (xi - prototypes[j])

        if self.c_w_[j] == c_xi:
            gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) - self._p(j, xi, prototypes=self.w_)) * d
        else:
            gradient = - self._p(j, xi, prototypes=self.w_) * d

        # Accumulate gradient
        self.squared_mean_gradient[j] = self.decay_rate *self.squared_mean_gradient[j] + (1 - self.decay_rate) * gradient ** 2

        # Compute update/step
        step = ((self.squared_mean_step[j] + self.epsilon) / (self.squared_mean_gradient[j] + self.epsilon)) **0.5 * gradient

        # Accumulate updates
        self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + (1 - self.decay_rate) * step ** 2

        # Attract/Distract prototype to/from data point
        self.w_[j] += step

    def _update_prototype_rmsprop(self, j, xi, c_xi, prototypes):
        """RMSprop"""
        d = xi - prototypes[j]
                
        if self.c_w_[j] == c_xi:
            gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) - self._p(j, xi, prototypes=self.w_)) * d
        else:
            gradient = - self._p(j, xi, prototypes=self.w_) * d
            
        # Accumulate gradient
        self.squared_mean_gradient[j] = 0.9 * self.squared_mean_gradient[j] + 0.1 * gradient ** 2
        
        # Update Prototype
        self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient

    def _update_prototype_rmspropada(self, j, xi, c_xi, prototypes):
        """RMSpropada"""
        d = (xi - prototypes[j])

        if self.c_w_[j] == c_xi:
            gradient = (self._p(j, xi, prototypes=self.w_,
            y=c_xi) - self._p(j, xi, prototypes=self.w_))
        else:
            gradient = self._p(j, xi, prototypes=self.w_)

        # calc adaptive decay rate
        dec_rate = np.minimum(np.absolute(self._costf(j=j, x=gradient**2, w=self.squared_mean_gradient[j])), 0.9)

        self.decay_rate[j] = 1.0 - dec_rate

        # Accumulate gradient
        self.squared_mean_gradient[j] = self.decay_rate[j] * self.squared_mean_gradient[j] + (1 - self.decay_rate[j]) * gradient ** 2

        # Update Prototype
        if self.c_w_[j] == c_xi:
            self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient * d
        else:
            self.w_[j] -= (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient * d

    def _update_prototype_adam(self, j, xi, c_xi, prototypes):
        """Adam"""
        d = (xi - prototypes[j])

        """Calculate posterior"""
        if self.c_w_[j] == c_xi:
            gradient = (self._p(j, xi, prototypes=self.w_,y=c_xi) - self._p(j, xi, prototypes=self.w_))* d
        else:
            gradient = - self._p(j, xi, prototypes=self.w_)* d

        """Compute gradients m """
        self.gradients_m[j] = self.beta_1 * self.gradients_m[j] + (1 - self.beta_1) * gradient

        """Compute squared gradients v """
        self.gradients_v_sqrt[j] = self.beta_2 * self.gradients_v_sqrt[j] + (1 - self.beta_2) * np.sqrt(gradient) 

        """Compute m correction"""
        m_corrected = self.gradients_m[j] / (1 - self.beta_1)

        """Compute v correction"""
        v_corrected = self.gradients_v_sqrt[j] / (1 - self.beta_2)

        """Update prototype"""
        self.w_[j] += - (self.learning_rate/(np.sqrt(self.v_corrected) + self.epsilon ))*self.m_corrected

    def _validate_train_parms(self, train_set, train_lab, classes=None):
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set, train_lab.ravel())

        if self.initial_fit:
            if classes:
                self.classes_ = np.asarray(classes)
                self.protos_initialized = np.zeros(self.classes_.size)
            else:
                self.classes_ = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.classes_.size)

            # Validate that labels have correct format
            for i in range(len(self.classes_)):
                if i not in self.classes_:
                    raise ValueError('Labels have to be ascending int,starting at 0, got {}'.format(self.classes_))

        nb_classes = len(self.classes_)
        nb_features = train_set.shape[1]

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            # ppc is int so we can give same number ppc to for all classes
            if self.prototypes_per_class < 0:
                raise ValueError('prototypes_per_class must be a positive int')
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        elif isinstance(self.prototypes_per_class, list):
            # its an array containing individual number of protos per class
            # - not fully supported yet
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    'values in prototypes_per_class must be positive')
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    'length of prototypes_per_class'
                    ' does not fit the number of classes'
                    'classes=%d'
                    'length=%d' % (nb_classes, nb_ppc.size))
        else:
            raise ValueError('Invalid data type for prototypes_per_class, '
                             'must be int or list of int')

        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features],
                                   dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes_)):
                actClass = self.classes_[actClassIdx]
                nb_prot = nb_ppc[actClassIdx]  # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and
                        actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == actClass, :], 0)

                    if self.prototypes_per_class == 1:
                        # If only one prototype we init it to mean
                        self.w_[pos:pos + nb_prot] = mean
                    else:
                        # else we add some random noise to distribute them
                        self.w_[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)

                    if math.isnan(self.w_[pos, 0]):
                        raise ValueError('Prototype on position {} for class\
                                         {} is NaN.'
                                         .format(pos, actClass))
                    else:
                        self.protos_initialized[actClassIdx] = 1

                    self.c_w_[pos:pos + nb_prot] = actClass
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        if self.initial_fit:
            if self.gradient_descent == 'adadelta' or 'rmsprop' or 'rmspropada':
                self.squared_mean_gradient = np.zeros_like(self.w_)
                self.squared_mean_step = np.zeros_like(self.w_)
                self.initial_fit = False
            elif self.gradient_descent == 'adam':
                self.gradients_v_sqrt = np.zeros_like(self.w_)
                self.gradients_m = np.zeros_like(self.w_)
                self.initial_fit = False
            elif self.gradient_descent == 'sgd':
                self.initial_fit = False

        return train_set, train_lab

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

        Returns
        --------
        self
        """
        if set(unique_labels(y)).issubset(set(self.classes_)) or self.initial_fit is True:
            X, y = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'
                             .format(y))

        self._optimize(X, y, random_state)
        return self

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
                    gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) - self._p(j, xi, prototypes=self.w_)) * d
                else:
                    gradient = - self._p(j, xi, prototypes=self.w_) * d

                # Accumulate gradient
                self.squared_mean_gradient[j] = self.decay_rate *self.squared_mean_gradient[j] + (1 - self.decay_rate) * gradient ** 2

                # Compute update/step
                step = ((self.squared_mean_step[j] + self.epsilon) / (self.squared_mean_gradient[j] + self.epsilon)) **0.5 * gradient

                # Accumulate updates
                self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + (1 - self.decay_rate) * step ** 2

                # Attract/Distract prototype to/from data point
                self.w_[j] += step

        elif(self.gradient_descent=="rmsprop"):
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

        elif(self.gradient_descent=="rmspropada"):
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
                    self.squared_mean_gradient[j] = self.decay_rate[j] * self.squared_mean_gradient[j] + (1 - self.decay_rate[j]) * gradient ** 2

                    # Update Prototype
                    if self.c_w_[j] == c_xi:
                        self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient * d
                    else:
                        self.w_[j] -= (self.learning_rate / ((self.squared_mean_gradient[j] + self.epsilon) ** 0.5)) * gradient * d

        elif(self.gradient_descent=="adam"):
            """Implementation of ADAM"""
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = x[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])

                    """Calculate posterior"""
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_,y=c_xi) - self._p(j, xi, prototypes=self.w_))* d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_)* d

                    """Compute gradients m """
                    self.gradients_m[j] = self.beta_1 * self.gradients_m[j] + (1 - self.beta_1) * gradient

                    """Compute squared gradients v """
                    self.gradients_v_sqrt[j] = self.beta_2 * self.gradients_v_sqrt[j] + (1 - self.beta_2) * np.sqrt(gradient) 

                    """Compute m correction"""
                    m_corrected = self.gradients_m[j] / (1 - self.beta_1)

                    """Compute v correction"""
                    v_corrected = self.gradients_v_sqrt[j] / (1 - self.beta_2)

                    """Update prototype"""
                    self.w_[j] += - (self.learning_rate/(np.sqrt(self.v_corrected) + self.epsilon ))*self.m_corrected