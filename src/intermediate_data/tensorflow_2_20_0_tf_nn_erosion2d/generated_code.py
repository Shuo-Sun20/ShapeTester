import tensorflow as tf
import numpy as np

def call_func(inputs, strides, padding, data_format='NHWC', dilations=None, name=None):
    value = inputs[0]
    filters = inputs[1]
    output = tf.nn.erosion2d(
        value=value,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )
    return output

batch = 2
in_height = 5
in_width = 5
depth = 3
filters_height = 3
filters_width = 3

value = tf.constant(np.random.randn(batch, in_height, in_width, depth).astype(np.float32))
filters = tf.constant(np.random.randn(filters_height, filters_width, depth).astype(np.float32))
inputs = [value, filters]
strides = [1, 1, 1, 1]
padding = 'SAME'
dilations = [1, 1, 1, 1]

example_output = call_func(inputs, strides, padding, dilations=dilations)