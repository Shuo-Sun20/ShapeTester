import tensorflow as tf
import numpy as np

def call_func(inputs, strides, padding, data_format=None, dilations=None, name=None):
    input_tensor, filter_tensor = inputs[0], inputs[1]
    return tf.nn.depthwise_conv2d(
        input=input_tensor,
        filter=filter_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

batch_size = 2
height = 4
width = 4
in_channels = 3
channel_multiplier = 1

input_tensor = tf.convert_to_tensor(
    np.random.randn(batch_size, height, width, in_channels).astype(np.float32)
)
filter_tensor = tf.convert_to_tensor(
    np.random.randn(2, 2, in_channels, channel_multiplier).astype(np.float32)
)

example_output = call_func(
    inputs=[input_tensor, filter_tensor],
    strides=[1, 1, 1, 1],
    padding='VALID'
)