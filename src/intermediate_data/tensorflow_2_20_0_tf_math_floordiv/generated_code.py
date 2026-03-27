import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    x, y = inputs[0], inputs[1]
    return tf.math.floordiv(x, y, name)

x_tensor = tf.constant(np.random.randn(3, 4).astype(np.float32))
y_tensor = tf.constant(np.random.randn(3, 4).astype(np.float32))
example_output = call_func([x_tensor, y_tensor])