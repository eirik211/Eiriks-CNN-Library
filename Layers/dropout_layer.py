import numpy as np
import numba
from random import uniform

class Dropout :

    def __init__(self, chance):
        self.chance = chance
        self.output = None
        self.grid = None
        self.has_weights = False

    def forwardProp(self, input):
        self.grid = np.zeros(input.shape)

        self.output = input * self.grid
        return self.output

    def backwardProp(self, outputDelta):
        inputDelta = self.grid * outputDelta
        return inputDelta

@numba.vectorize()
def f(x, chance):
    if uniform(0, 1) < chance:
        return 0
    else:
        return 1 / (1 - chance)
