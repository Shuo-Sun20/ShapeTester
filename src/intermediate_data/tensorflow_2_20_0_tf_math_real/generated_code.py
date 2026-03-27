import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.real(inputs, name=name)

# Generate random complex tensor for testing
np.random.seed(42)
real_part = np.random.randn(3, 4).astype(np.float32)
imag_part = np.random.randn(3, 4).astype(np.float32)
complex_tensor = tf.constant(real_part + 1j * imag_part)

# Call the function with generated input
example_output = call_func(complex_tensor)