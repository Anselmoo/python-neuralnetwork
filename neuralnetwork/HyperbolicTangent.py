import numpy as np


class HyperbolicTangent:
    def getActivation(self, net):
        return np.tanh(net)

    def getDerivative(self, net):
        return 1 - (self.getActivation(net) * self.getActivation(net))
