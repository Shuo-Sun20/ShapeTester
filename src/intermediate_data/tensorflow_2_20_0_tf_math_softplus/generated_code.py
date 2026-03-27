import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.softplus(features=inputs, name=name)

# Generate a random tensor
random_tensor = tf.constant(np.random.randn(3, 4).astype(np.float32))

# Call the function with the random tensor
example_output = call_func(random_tensor)