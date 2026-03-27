import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    a, b = inputs
    return tf.linalg.cross(a, b, name=name)

# Create random tensors with shape (5, 3) to test innermost dimension of 3
np.random.seed(42)
a_np = np.random.randn(5, 3).astype(np.float32)
b_np = np.random.randn(5, 3).astype(np.float32)
a = tf.constant(a_np)
b = tf.constant(b_np)
example_output = call_func([a, b])