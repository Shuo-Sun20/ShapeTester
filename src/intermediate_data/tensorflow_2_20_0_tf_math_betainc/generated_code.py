import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    a, b, x = inputs
    return tf.math.betainc(a, b, x, name=name)

# Generate valid random tensors
np.random.seed(42)
tensor_a = tf.constant(np.random.uniform(0.1, 5.0, (3, 3)), dtype=tf.float32)
tensor_b = tf.constant(np.random.uniform(0.1, 5.0, (3, 3)), dtype=tf.float32)
tensor_x = tf.constant(np.random.uniform(0.0, 1.0, (3, 3)), dtype=tf.float32)

example_output = call_func([tensor_a, tensor_b, tensor_x])