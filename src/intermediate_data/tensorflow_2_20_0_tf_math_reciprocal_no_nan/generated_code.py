import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    x = inputs[0]
    return tf.math.reciprocal_no_nan(x, name=name)

np.random.seed(42)
random_tensor = tf.constant(np.random.randn(3, 4).astype(np.float32))
random_tensor = tf.where(tf.random.uniform(random_tensor.shape) < 0.2, 0.0, random_tensor)
example_output = call_func([random_tensor])