import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.is_nan(x=inputs, name=name)

random_tensor = tf.constant([1.0, np.nan, 3.14, float('nan'), float('inf')], dtype=tf.float32)
example_output = call_func(inputs=random_tensor)