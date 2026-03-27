import tensorflow as tf
import numpy as np

def call_func(inputs, name=None, conjugate=False):
    a = inputs[0]  # Unpack the single input tensor from the list
    return tf.linalg.matrix_transpose(a, name=name, conjugate=conjugate)

# Generate a random tensor with rank >= 2 (e.g., shape [2, 3, 4])
random_tensor = tf.constant(np.random.randn(2, 3, 4).astype(np.float32))
# Call the function with the input tensor wrapped in a list
example_output = call_func([random_tensor])