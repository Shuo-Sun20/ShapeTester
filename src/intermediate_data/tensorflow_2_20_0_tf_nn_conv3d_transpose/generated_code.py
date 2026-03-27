import tensorflow as tf
import numpy as np

def call_func(inputs, strides, padding, data_format='NDHWC', dilations=1, name=None):
    input_tensor, filters_tensor, output_shape_tensor = inputs
    return tf.nn.conv3d_transpose(
        input=input_tensor,
        filters=filters_tensor,
        output_shape=output_shape_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

batch_size = 2
in_depth, in_height, in_width = 5, 5, 5
in_channels = 3
out_channels = 4
kernel_depth, kernel_height, kernel_width = 3, 3, 3

input_tensor = tf.constant(np.random.randn(batch_size, in_depth, in_height, in_width, in_channels), dtype=tf.float32)
filters_tensor = tf.constant(np.random.randn(kernel_depth, kernel_height, kernel_width, out_channels, in_channels), dtype=tf.float32)
output_shape_tensor = tf.constant([batch_size, in_depth*2, in_height*2, in_width*2, out_channels], dtype=tf.int32)

inputs = [input_tensor, filters_tensor, output_shape_tensor]
strides = [1, 2, 2, 2, 1]
padding = 'SAME'
data_format = 'NDHWC'

example_output = call_func(inputs, strides, padding, data_format)