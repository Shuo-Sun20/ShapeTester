import tensorflow as tf

def call_func(inputs, lower=True, adjoint=False, name=None):
    bands, rhs = inputs
    return tf.linalg.banded_triangular_solve(bands, rhs, lower, adjoint, name)

example_bands = tf.constant([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
example_rhs = tf.constant([[1.0], [1.0], [1.0]])
example_output = call_func([example_bands, example_rhs])