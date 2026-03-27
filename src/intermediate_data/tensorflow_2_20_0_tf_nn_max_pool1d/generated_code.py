import tensorflow as tf
import numpy as np

def call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.max_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

example_input = tf.constant(np.random.randn(2, 10, 3).astype(np.float32))
example_output = call_func(inputs=example_input, ksize=2, strides=2, padding="VALID")