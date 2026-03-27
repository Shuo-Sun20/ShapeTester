import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    x = inputs[0]
    q = inputs[1]
    return tf.math.zeta(x, q, name)

# Generate random tensors for x and q
np.random.seed(42)
x_tensor = tf.constant(np.random.rand(3, 3).astype(np.float32))
q_tensor = tf.constant(np.random.rand(3, 3).astype(np.float32))

example_output = call_func([x_tensor, q_tensor])