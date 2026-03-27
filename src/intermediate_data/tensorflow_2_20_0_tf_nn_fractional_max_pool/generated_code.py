import tensorflow as tf
import numpy as np

def call_func(inputs, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None):
    return tf.nn.fractional_max_pool(
        value=inputs,
        pooling_ratio=pooling_ratio,
        pseudo_random=pseudo_random,
        overlapping=overlapping,
        seed=seed,
        name=name
    )[0]

batch_size = 2
height = 10
width = 10
channels = 3
pooling_ratio = [1.0, 1.44, 1.73, 1.0]

input_tensor = tf.constant(np.random.randn(batch_size, height, width, channels).astype(np.float32))
example_output = call_func(input_tensor, pooling_ratio)