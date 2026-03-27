import tensorflow as tf
import numpy as np

def call_func(inputs, adjoint=False, name=None):
    return tf.linalg.inv(input=inputs, adjoint=adjoint, name=name)

# Generate a random 3x3 invertible matrix
np.random.seed(42)
random_matrix = np.random.randn(3, 3).astype(np.float32)
# Ensure it's invertible by making it diagonally dominant
random_matrix = random_matrix + np.eye(3) * 3

example_output = call_func(inputs=random_matrix)