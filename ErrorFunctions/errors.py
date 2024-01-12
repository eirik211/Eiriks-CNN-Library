from distutils.log import error
import numpy as np

class MeanSquaredError :

    def __init__(self):
        pass

    def errorFunction(self, output, Y):
        error = (output - Y) ** 2
        return error

    def errorDerivative(self, output, Y):
        error_deriv = 2 * (output - Y)
        return error_deriv

class BinaryCrossEntropy :

    def __init__(self):
        pass
    
    def errorFunction(self, output, Y):
        Y += 0.00000000001
        output += 0.00000000001
        return np.mean(-Y * np.log(output) - (1 - Y) * np.log(1 - output))

    def errorDerivative(self, output, Y):
        Y += 0.00000000001
        output += 0.00000000001
        return (((1 - Y) / (1 - output) - Y / output) / np.size(Y))


class CategoricalCrossEntropy :

    def __init__(self):
        pass

    def errorFunction(self, output, Y):
        error = -sum(Y * np.log10(output))
        return error

    def errorDerivative(self, output, Y):
        errorDeriv = -Y/(output + 10**-100)
        for val in errorDeriv:
            if np.isnan(val):
                val = 0
        return errorDeriv

