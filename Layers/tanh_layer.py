import numpy as np
import numba

class Tanh:

    def __init__(self):
        self.input = None
        self.output = None

        self.has_weights = False

    def forwardProp(self, input):
        self.input = input
        self.output = tanh(input)
        return self.output

    def backwardProp(self, outputDelta):
        inputDelta = d_tanh(self.output)
        inputDelta *= outputDelta
        return inputDelta

@numba.vectorize
def d_tanh(output):
    inputDelta = 1 - (output ** 2)
    return inputDelta

@numba.vectorize
def tanh(x):
    return np.tanh(x)