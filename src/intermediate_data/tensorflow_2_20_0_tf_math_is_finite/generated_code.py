import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.is_finite(x=inputs, name=name)

# Generate a random tensor with some non-finite values
arr = np.random.randn(10).astype(np.float32)
arr[0] = np.inf
arr[1] = np.nan
arr[2] = -np.inf
x = tf.constant(arr)

example_output = call_func(x)