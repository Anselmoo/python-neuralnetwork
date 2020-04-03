import numpy as np


class Sigmoid:
    def getActivation(self, net):
        return 1 / (1 + np.exp(-1 * net))

    def getDerivative(self, net):
        return self.getActivation(net) * (1 - self.getActivation(net))
