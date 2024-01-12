import numpy as np
import numba

class ReLU:

    def __init__(self):
        self.input = None
        self.output = None

        self.has_weights = False
    
    def forwardProp(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backwardProp(self, outputDelta):
        inputDelta = outputDelta * np.vectorize(self.anti_relu)(self.input)
        return inputDelta

    def anti_relu(self, x):
        if x < 0:
            return 0
        else:
            return 1


class LeakyReLU:

    def __init__(self):
        self.input = None
        self.output = None

        self.has_weights = False
    
    def forwardProp(self, input):
        self.input = input
        output = leakyrelu(input, 0.01)
        return output

    def backwardProp(self, outputDelta):
        inputDelta = leakyrelu_prime(self.input, 0.01)
        inputDelta = inputDelta * outputDelta
        return inputDelta

@numba.vectorize
def leakyrelu(x, alpha):
    return max(alpha * x, x)

@numba.vectorize
def leakyrelu_prime(x, alpha):
    return 1 if x > 0 else alpha


