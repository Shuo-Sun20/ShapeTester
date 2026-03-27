import tensorflow as tf
import numpy as np

def call_func(inputs, strides, padding, data_format='NDHWC', dilations=None, name=None):
    if dilations is None:
        dilations = [1, 1, 1, 1, 1]
    input_tensor = inputs[0]
    filters_tensor = inputs[1]
    return tf.nn.conv3d(
        input=input_tensor,
        filters=filters_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

np.random.seed(42)
input_data = np.random.randn(2, 10, 10, 10, 4).astype(np.float32)
filter_data = np.random.randn(3, 3, 3, 4, 6).astype(np.float32)
inputs_tensors = [tf.constant(input_data), tf.constant(filter_data)]

example_output = call_func(
    inputs=inputs_tensors,
    strides=[1, 2, 2, 2, 1],
    padding='SAME',
    data_format='NDHWC',
    dilations=[1, 1, 1, 1, 1],
    name='example_conv3d'
)