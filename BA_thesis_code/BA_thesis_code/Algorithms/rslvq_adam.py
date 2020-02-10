import math
import numpy as np
from sklearn.utils import validation
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from skmultiflow.core.base import ClassifierMixin, BaseSKMObject
from .base_lvq import BaseLVQ

class RSLVQAdam(BaseLVQ):
    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=None, beta_1 = 0.9, beta_2 = 0.999,
                 learning_rate = 0.001):
        self.sigma = sigma
        self.random_state = random_state
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.v_corrected  = 0
        self.m_corrected  = 0

        if sigma <= 0:
            raise ValueError('Sigma must be greater than 0')
        if prototypes_per_class <= 0:
            raise ValueError('Prototypes per class must be more than 0')

    def _update_prototype(self, j, xi, c_xi, prototypes):
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
                    raise ValueError('Labels have to be ascending int, starting at 0, got {}'
                                     .format(self.classes_))

        nb_classes = len(self.classes_)
        nb_features = train_set.shape[1]

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            # ppc is int so we can give same number ppc to for all classes
            if self.prototypes_per_class < 0:
                raise ValueError("prototypes_per_class must be a positive int")
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            # its an array containing individual number of protos per class - not fully supported yet
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))
        
        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes_)):
                actClass = self.classes_[actClassIdx]
                nb_prot = nb_ppc[actClassIdx] # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and actClass in unique_labels(train_lab)):
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
                        raise ValueError('Prototype on position {} for class {} is NaN.'
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
            # Next two lines are Init for ADAM last m + v gradients
            self.gradients_v_sqrt = np.zeros_like(self.w_)
            self.gradients_m = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab