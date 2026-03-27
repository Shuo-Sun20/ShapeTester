import tensorflow as tf
import numpy as np

def call_func(inputs, full_matrices=False, name=None):
    q, r = tf.linalg.qr(input=inputs, full_matrices=full_matrices, name=name)
    return [q, r]

input_tensor = tf.constant(np.random.randn(3, 4, 2).astype(np.float32))
example_output = call_func(input_tensor, full_matrices=True)