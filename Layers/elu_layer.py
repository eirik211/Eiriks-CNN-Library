import numpy as np
import numba

class ELU :

    def __init__(self):
        self.has_weights = False


    def forwardProp(self, input):
        self.input = input
        output = elu(input, 1)
        return output

    def backwardProp(self, outputDelta):
        inputDelta = np.multiply(outputDelta, d_elu(self.input, 1))
        return inputDelta

@numba.vectorize()
def elu(x, alpha):
    return x if x >= 0 else alpha*(np.exp(x) -1)

@numba.vectorize()
def d_elu(x, alpha):
	return 1 if x > 0 else alpha*np.exp(x)