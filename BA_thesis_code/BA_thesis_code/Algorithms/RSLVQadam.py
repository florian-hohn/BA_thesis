import numpy as np
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ

class RSLVQAdam(RSLVQ):
    """description of class"""
    def _optimize(self, X, y, random_state):
            """Implementation of Adam"""
            alpha = 0.001
            beta1 = 0.9
            beta2 = 0.999
            t = 0
            m = 0  #initalize as vector of 0, not as 0
            v = 0  #initalize as vector of 0, not as 0
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = X[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    """Calculate posterior"""
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_,y=c_xi) - self._p(j, xi, prototypes=self.w_))* d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_)* d
                    
                    """Update t by 1"""
                    t += 1

                    """Compute decaying averages of gradients"""
                    m = beta1 * m + (1-beta1)*gradient

                    """Compute decaying averages of gradients^2"""
                    v = beta2 * v + (1-beta2)*gradient^2

                    """Update m by the last gradient"""
                    m = gradient

                    """Update m by the last squared gradient"""
                    v = gradient^2

                    """Update prototype"""
                    self.w_[j] += - (self.learning_rate/(sqrt(self._v_corrected(t,beta2,v)) + self.epsilon ))*self._m_corrected(t,beta1,m)

    def _m_corrected(self,t,beta1,m):
        M_t = m / (1-beta1^t)
        return M_t

    def _v_corrected(self,t,beta2,v):
        V_t = v / (1-beta2^t)
        return V_t

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