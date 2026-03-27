import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    x = inputs[0]
    y = inputs[1]
    return tf.math.xlog1py(x, y, name)

x = tf.constant(np.random.randn(2, 3), dtype=tf.float32)
y = tf.constant(np.random.randn(2, 3), dtype=tf.float32)
example_output = call_func([x, y])