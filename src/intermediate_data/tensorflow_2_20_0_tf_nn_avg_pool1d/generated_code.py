import tensorflow as tf
import numpy as np

def call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.avg_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

input_tensor = tf.convert_to_tensor(np.random.randn(4, 10, 3).astype(np.float32))
example_output = call_func(inputs=input_tensor, ksize=2, strides=2, padding='VALID')