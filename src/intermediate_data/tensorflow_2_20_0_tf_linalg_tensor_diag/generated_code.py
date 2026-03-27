import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    diagonal = inputs
    return tf.linalg.tensor_diag(diagonal=diagonal, name=name)

np.random.seed(42)
example_input = tf.constant(np.random.randn(4), dtype=tf.float32)
example_output = call_func(example_input)