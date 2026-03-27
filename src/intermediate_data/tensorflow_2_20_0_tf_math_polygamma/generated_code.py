import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    a, x = inputs[0], inputs[1]
    return tf.math.polygamma(a, x, name=name)

np.random.seed(42)
example_a = tf.constant(np.random.randn(3, 2).astype(np.float32))
example_x = tf.constant(np.random.rand(3, 2).astype(np.float32))
example_output = call_func([example_a, example_x])