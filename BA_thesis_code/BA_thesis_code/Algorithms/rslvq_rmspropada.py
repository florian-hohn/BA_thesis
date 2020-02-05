from .base_rslvq import BaseRSLVQ

class RSLVQRMSpropada(BaseRSLVQ):
    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=None, gradient_descent='sgd',
                 gamma=0.9):
        self.sigma = sigma
        self.random_state = random_state
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.learning_rate = 1 / sigma
        self.gamma = gamma
        self.gradient_descent = gradient_descent

        if sigma <= 0:
            raise ValueError('Sigma must be greater than 0')
        if prototypes_per_class <= 0:
            raise ValueError('Prototypes per class must be more than 0')
        if gamma >= 1 or gamma < 0:
            raise ValueError('Decay rate gamma has to be between 0 and\
                             less than 1')


