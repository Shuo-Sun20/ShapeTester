import tensorflow as tf
import numpy as np

def call_func(inputs, k=0, padding_value=0, align="RIGHT_LEFT", name=None):
    return tf.linalg.diag_part(input=inputs, k=k, padding_value=padding_value, align=align, name=name)

example_input = tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32)
example_output = call_func(inputs=example_input, k=1)