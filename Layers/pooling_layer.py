import numpy as np
import numba

class Pooling:

    def __init__(self, size):
        self.size = size
        self.input = None

        self.has_weights = False

    def forwardProp(self, input):
        self.input = input
        output = []

        for i in range(input.shape[0]):
            output.append(pool(input[i], self.size))
        output = np.asarray(output)
        return output

    def backwardProp(self, output_delta):
        input_delta = anti_pool(output_delta, self.input.shape, self.size, self.input)
        return input_delta


def anti_pool(output_delta, input_shape, size, input):

    input_delta = np.zeros(input_shape)

    for l in range(input_delta.shape[0]):
        for x in range(output_delta.shape[1]):
            for y in range(output_delta.shape[2]):
                area_start = (x * size, y * size)
                area_end = (min((x + 1) * size, input_delta.shape[1]), 
                            min((y + 1) * size, input_delta.shape[2]))
                area = (input[l, area_start[0]:area_end[0], area_start[1]:area_end[1]])
                highest_pos = np.unravel_index(area.argmax(), area.shape)
                highest_pos = [x * size + highest_pos[0],
                               y * size + highest_pos[1]]
                input_delta[l, highest_pos[0], highest_pos[1]] = output_delta[l, x, y]
    return input_delta

        
@numba.njit("float64[:,:](float64[:,:], int32)")
def pool(mat, size):
    def pool_at_position(mat, pos):
        end_pos = (min(mat.shape[0], pos[0] + size),
                   min(mat.shape[1], pos[1] + size))
        area = mat[pos[0]:end_pos[0], pos[1]:end_pos[1]]
        result = np.max(area)
        return result

    output_size = (int(np.ceil(mat.shape[0] / size)), int(np.ceil(mat.shape[1] / size)))
    output = np.zeros(output_size)
    for x in range(output_size[0]):
        for y in range(output_size[1]):
            output[x, y] = pool_at_position(mat, (x * size, y * size))
    return output


