from statistics import correlation
from .Layers.dense_layer import Dense
from .Layers.convolutional_layer import Convolutional
from .Layers.pooling_layer import Pooling
from .Layers.dropout_layer import Dropout
from .Layers.reshape_layer import Reshape
from .Layers.sigmoid_layer import Sigmoid
from .Layers.tanh_layer import Tanh
from .Layers.relu_layer import ReLU, LeakyReLU
from .Layers.elu_layer import ELU
from .Layers.softmax_layer import Softmax
from .ErrorFunctions.errors import *
from .Regulators.bs import BSStandart

import time

class NeuralNetwork:

    def __init__(self, layers, loss):
        self.layers = layers
        self.w_layers = []
        self.loss = loss

        self.lr_regulator = None
        self.bs_regulator = None


    def forwardProp(self, input):
        output = input
        for layer in self.layers:
            output = layer.forwardProp(output)
        return output

    def backwardProp(self, output, Y):
        deltaOutput = self.loss.errorDerivative(output, Y)
        for layer in reversed(self.layers):
            deltaOutput = layer.backwardProp(deltaOutput)

    def adjustWeights(self, lr):
        for layer in self.layers:
            if layer.has_weights == True:
                layer.adjustWeights(lr)

    def train(self, xTrain, yTrain, epochs=1, lr=1, batch_size=1, shuffle=False, interimResult=False):
        
        for layer in self.layers:
            layer.batch_size = batch_size
        
        
        batches = int(np.floor(len(xTrain) / batch_size))
        corrects = 1     
        for epoch in range(epochs):
            start = 0
            end = 0
            xTrain, yTrain = shuffle_in_unison(xTrain, yTrain)
            for batch in range(1, batches - 1):
                print(f"Batch: {batch} of {batches} --> {round(corrects / ((batch-1) * batch_size + 1), 3)} --> {round(end - start, 5)}s")
                start = time.time()
                for i in range(batch_size):
                    loc = (batch-1) * batch_size + i
                    data = xTrain[loc]
                    label = yTrain[loc]

                    output = self.forwardProp(data)
                    if len(output) != 1:
                        if output.argmax() == label.argmax():
                            corrects += 1
                    else:
                        if (output > 0.5 and label == 1) or (output < 0.5 and label == 0):
                            corrects += 1
                    self.backwardProp(output, label)
                self.adjustWeights(lr)
                end = time.time()
            corrects = 0
            # lr *= 0.825

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b