import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.linalg.adjoint(matrix=inputs, name=name)

# Generate a random complex tensor matching the example shape
random_real = np.random.randn(2, 3).astype(np.float32)
random_imag = np.random.randn(2, 3).astype(np.float32)
random_complex_tensor = tf.constant(random_real + 1j * random_imag)

example_output = call_func(inputs=random_complex_tensor)