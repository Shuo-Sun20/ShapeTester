import tensorflow as tf

def call_func(inputs, name=None):
    x = inputs[0] if isinstance(inputs, list) else inputs
    return tf.math.conj(x=x, name=name)

import tensorflow as tf
import numpy as np

complex_tensor = tf.constant(np.random.randn(3, 2) + 1j * np.random.randn(3, 2), dtype=tf.complex64)
example_output = call_func(inputs=complex_tensor)