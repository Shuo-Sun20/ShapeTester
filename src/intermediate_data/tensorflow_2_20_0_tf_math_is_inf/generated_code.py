import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.is_inf(x=inputs, name=name)

example_output = call_func(tf.constant([1.0, np.inf, -np.inf, 5.0, np.nan]))