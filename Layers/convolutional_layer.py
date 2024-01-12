import numpy as np
import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from scipy import signal
from time import time

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class Convolutional():
    def __init__(self, input_depth, input_size, kernel_size, depth):
        self.input_depth = input_depth
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.depth = depth
        self.kernels = np.random.uniform(-0.5, 0.5, (depth, input_depth, kernel_size, kernel_size))
        self.bias =  np.random.uniform(-0.5, 0.5, (depth, input_size[0] - kernel_size + 1, input_size[1] - kernel_size + 1)) 
        self.input = None

        self.has_weights = True
        self.batch_size = 0
        self.grads = np.zeros(self.kernels.shape)
        self.b_grads = np.zeros(self.bias.shape)


    def forwardProp(self, input):
        self.input = input
        output = get_output(input, self.depth, self.input_size, self.kernel_size, self.input_depth, self.kernels, self.bias)
        # output = np.zeros((self.depth, self.input_size[0] - self.kernel_size + 1,
        #                    self.input_size[0] - self.kernel_size + 1))
        # for i in range(self.depth):
        #     for j in range(self.input_depth):
        #         f_kernel = np.rot90(np.rot90(self.kernels[i, j]))
        #         output[i] += signal.fftconvolve(input[j], f_kernel, mode="valid")
        return output


    def backwardProp(self, output_delta):
        kernels_gradient, input_delta = get_gradients(self.kernels, self.input, self.depth, self.input_depth, output_delta)

        self.grads += kernels_gradient
        self.b_grads += output_delta

        return input_delta

    def adjustWeights(self, lr):
        self.kernels += -lr * (self.grads / self.batch_size)
        self.bias += -lr * (self.b_grads / self.batch_size)

        self.grads = np.zeros(self.kernels.shape)
        self.b_grads = np.zeros(self.bias.shape)


@numba.njit(fastmath=True)
def get_gradients(kernels, input, depth, input_depth, output_delta):

    kernels_gradient = np.zeros(kernels.shape)
    input_delta = np.zeros(input.shape)

    for i in range(depth):
        for j in range(input_depth):
            kernels_gradient[i, j] = valid_correlate(input[j], output_delta[i])
            input_delta[j] += full_convolve(output_delta[i], kernels[i, j])
    return kernels_gradient, input_delta


@numba.njit(fastmath=True, nogil=True)
def get_output(input, depth, input_size, kernel_size, input_depth, kernels, bias):
    out = np.zeros((depth, input_size[0] - kernel_size + 1, input_size[0] - kernel_size + 1))
    for k in range(depth):
        for i in range(input_depth):
            out[k] += valid_correlate(input[i], kernels[k][i])
        out[k] += bias[k]
    return out

@numba.njit(fastmath=True, nogil=True)
def apply_filter(mat, filter, point):
    assert len(point) == 2
    point = (max(0, point[0]), max(0, point[1]))
    end_point = (min(mat.shape[0], point[0] + filter.shape[0]),
                 min(mat.shape[1], point[1] + filter.shape[1]))
    area = mat[point[0]:end_point[0], point[1]:end_point[1]]
    if filter.shape != area.shape:
        s_filter = filter[0:area.shape[0], 0:area.shape[1]]
    else:
        s_filter = filter
    result = 0.0
    for x in range(area.shape[0]):
        for y in range(area.shape[1]):
            result += area[x, y] * s_filter[x, y]
    return result


@numba.njit('float64[:,:](float64[:,:], float64[:,:])', nogil=True)
def valid_correlate(mat, filter):
    f_mat = np.zeros((mat.shape[0] - filter.shape[0] + 1, mat.shape[1] - filter.shape[0] + 1))

    for x in range(f_mat.shape[0]):
        for y in range(f_mat.shape[1]):
            f_mat[x, y] = apply_filter(mat, filter, (x, y))
    return f_mat

@numba.njit('float64[:,:](float64[:,:], float64[:,:])', nogil=True)
def full_convolve(mat, filter):
    f_mat = np.zeros((mat.shape[0] + filter.shape[0] - 1, mat.shape[1] + filter.shape[1] - 1))

    for _ in range(2):
        filter = np.rot90(filter)
    for x in range(f_mat.shape[0]):
        for y in range(f_mat.shape[1]):
            f_mat[x, y] = apply_filter(mat, filter, (x - filter.shape[0] + 1, y - filter.shape[1] + 1))
    return f_mat
