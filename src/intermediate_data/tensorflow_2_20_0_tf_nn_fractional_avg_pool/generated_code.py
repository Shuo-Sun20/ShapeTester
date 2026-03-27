import tensorflow as tf
import numpy as np

def call_func(inputs, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None):
    return tf.nn.fractional_avg_pool(value=inputs, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random, overlapping=overlapping, seed=seed, name=name)[0]

example_input = tf.constant(np.random.randn(4, 10, 10, 3).astype(np.float32))
example_output = call_func(inputs=example_input, pooling_ratio=[1.0, 1.44, 1.73, 1.0])