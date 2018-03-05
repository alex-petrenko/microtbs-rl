"""
Tensorflow helpers.

"""


import math

import tensorflow as tf

from microtbs_rl.utils.common_utils import *

logger = logging.getLogger(os.path.basename(__file__))


def dense(x, layer_size, regularizer, activation=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        layer_size,
        activation_fn=activation,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
    )


def conv(x, num_filters, kernel_size, stride=1, regularizer=None, scope=None):
    return tf.contrib.layers.conv2d(
        x,
        num_filters,
        kernel_size,
        stride=stride,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
        scope=scope,
    )


def avg_pool(inputs, kernel, stride, padding='SAME', scope=None):
    """
    Adds a Avg Pooling layer.
    It is assumed by the wrapper that the pooling is only done per image and not in depth or batch.
    """
    with tf.name_scope(scope, 'AvgPool', [inputs]):
        return tf.nn.avg_pool(inputs, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding)


def count_total_parameters():
    """
    Returns total number of trainable parameters in the current tf graph.
    https://stackoverflow.com/a/38161314/1645784
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def put_kernels_on_grid(kernel, pad=1):
    """
    Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    Courtesy of: https://gist.github.com/kukuruza/03731dc494603ceab0c5

    :param kernel: tensor of shape [Y, X, NumChannels, NumKernels]
    :param pad: number of black pixels around each filter (between them)
    :return: Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    logger.warning('Who would enter a prime number of filters?')
                return i, int(n / i)

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    logger.info('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x
