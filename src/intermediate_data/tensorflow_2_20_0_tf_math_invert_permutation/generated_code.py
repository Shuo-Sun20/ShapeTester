import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.invert_permutation(x=inputs, name=name)

np.random.seed(42)
permutation = np.random.permutation(8)
input_tensor = tf.constant(permutation, dtype=tf.int32)
example_output = call_func(inputs=input_tensor)