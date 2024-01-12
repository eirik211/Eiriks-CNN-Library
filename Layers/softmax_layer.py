import numpy as np
import numba

class Softmax:

    def __init__(self):
        self.input = None
        self.output = None

        self.has_weights = False

    def forwardProp(self, input):
        self.input = input
        # input = zero_safety(input)
        # input = input / max(input) # safe softmax für arme
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output

        # exp = np.exp(input - np.max(input))
        # print(exp / exp.sum())
        # return exp / exp.sum()

    def backwardProp(self, outputDelta): # alles hässlich
        softmax = np.reshape(self.output, (1, -1))
        grad = np.reshape(outputDelta, (1, -1))
        d_softmax = (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)
        inputDelta = grad @ d_softmax
        inputDelta = np.reshape(inputDelta, (inputDelta.shape[1], 1))
        return inputDelta

@numba.vectorize()
def zero_safety(x: float):
    SAFE = 0.0000000001
    if x == 0:
        return SAFE
    return x

