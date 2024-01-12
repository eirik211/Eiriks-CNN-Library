import numpy as np
from numba import cuda, float32

class Dense:

    def __init__(self, inputSize, outputSize):
        self.neurons = np.zeros((inputSize, 1))
        self.weights = np.random.uniform(-0.5, 0.5, (outputSize, inputSize))
        self.bias = np.zeros((outputSize, 1))
        self.output = np.zeros((outputSize, 1))

        self.batch_size = 0
        self.grads = []
        self.b_grads = []
        self.has_weights = True

    def forwardProp(self, input):
        self.neurons = input
        self.output = get_output(self.neurons, self.weights, self.bias)
        return self.output

    def backwardProp(self, outputDelta):
        self.grads.append(np.multiply(outputDelta, np.transpose(self.neurons)))
        self.b_grads.append(outputDelta)
        inputDelta = np.transpose(self.weights) @ outputDelta
        return inputDelta

    def adjustWeights(self, lr):
        w_adj = np.zeros(self.weights.shape)
        b_adj = np.zeros(self.bias.shape)
        for i in range(len(self.grads)):
            w_adj += self.grads[i]
            b_adj += self.b_grads[i]
        self.weights += -lr * (w_adj / len(self.grads))
        self.bias += -lr * (b_adj / len(self.grads))
        self.grads = []
        self.b_grads = []

def get_output(input, weights, bias):
    output = bias + np.matmul(weights, input)
    return output