import tensorflow as tf
import numpy as np

def call_func(inputs, decreasing=True, axis=-1):
    output, segments = tf.nn.isotonic_regression(inputs=inputs, decreasing=decreasing, axis=axis)
    return output

# Generate random input tensor
tf.random.set_seed(42)
random_tensor = tf.random.normal(shape=(3, 4), dtype=tf.float32)
example_output = call_func(inputs=random_tensor, axis=1)