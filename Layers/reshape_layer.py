import numpy as np

class Reshape:
    """ 
    This Layer is used to reshape 3 dimensional convolution outputs 
    into dense layer compatible 2 dimensional matrices.
    """

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.has_weights = False

    def forwardProp(self, input):
        output = np.reshape(input, self.output_shape)
        return output

    def backwardProp(self, output_delta):
        input_delta = np.reshape(output_delta, self.input_shape)
        return input_delta
        