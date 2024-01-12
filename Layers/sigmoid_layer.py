import numpy as np
import numba

class Sigmoid:

    def __init__(self):
        self.input = None
        self.output = None

        self.has_weights = False

    def forwardProp(self, input):
        self.input = input
        self.output = sigmoid(input)
        return self.output

    def backwardProp(self, outputDelta):
       inputDelta = np.multiply(outputDelta, self.output * (1 - self.output))
       return inputDelta

@numba.jit(nopython=True, fastmath=True)
def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

# @numba.jit(nopython=True, fastmath=True)
# def dx_sigmoid(input=0, output=0):
#   	if input != 0:
#         s_x = sigmoid(input)
#         return s_x * (1 - s_x)
#     else:
#       	return output * (1 - output)