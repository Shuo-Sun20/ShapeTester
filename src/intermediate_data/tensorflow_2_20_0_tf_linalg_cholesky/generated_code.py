import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    # Since tf.linalg.cholesky is a function, directly call it with the provided parameters.
    return tf.linalg.cholesky(input=inputs, name=name)

# Construct a valid input: symmetric positive-definite matrix
M = 5
batch_shape = (3, 4)  # Example batch shape
# Generate random matrix A of shape [..., M, M]
A = tf.random.normal(shape=batch_shape + (M, M))
# Make it symmetric: A @ A^T
A_sym = tf.matmul(A, A, transpose_b=True)
# Ensure positive-definite by adding a multiple of identity
identity = tf.eye(M, batch_shape=batch_shape)
A_positive_definite = A_sym + 0.1 * identity

example_output = call_func(inputs=A_positive_definite)