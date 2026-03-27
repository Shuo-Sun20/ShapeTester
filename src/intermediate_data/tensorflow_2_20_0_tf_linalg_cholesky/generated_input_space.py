import tensorflow as tf
import numpy as np
from dataclasses import dataclass

def call_func(inputs, name=None):
    return tf.linalg.cholesky(input=inputs, name=name)

# Construct a valid input: symmetric positive-definite matrix
M = 5
batch_shape = (3, 4)  # Example batch shape
A = tf.random.normal(shape=batch_shape + (M, M))
A_sym = tf.matmul(A, A, transpose_b=True)
identity = tf.eye(M, batch_shape=batch_shape)
A_positive_definite = A_sym + 0.1 * identity

valid_test_case = {
    'inputs': A_positive_definite,
    'name': None
}

@dataclass
class InputSpace:
    # There are no parameters (other than 'inputs') that affect the output shape.
    pass

# Example of instantiating InputSpace
var = InputSpace()