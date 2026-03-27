import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    x = inputs[0]
    output = tf.linalg.trace(x, name=name)
    return output

example_input = tf.constant(np.random.randn(3, 4, 4), dtype=tf.float32)
example_output = call_func(inputs=[example_input])