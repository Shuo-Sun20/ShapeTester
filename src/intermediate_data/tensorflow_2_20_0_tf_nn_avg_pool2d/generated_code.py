import tensorflow as tf
import numpy as np

def call_func(inputs, ksize, strides, padding, data_format='NHWC', name=None):
    return tf.nn.avg_pool2d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

np.random.seed(42)
example_input = tf.constant(np.random.randn(2, 8, 8, 3).astype(np.float32))
example_output = call_func(inputs=example_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')