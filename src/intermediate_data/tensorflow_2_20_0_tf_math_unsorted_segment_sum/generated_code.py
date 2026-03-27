import tensorflow as tf
import numpy as np

def call_func(inputs, segment_ids, num_segments, name=None):
    data = inputs[0]
    return tf.math.unsorted_segment_sum(data, segment_ids, num_segments, name)

data = tf.constant(np.random.randn(3, 4).astype(np.float32))
segment_ids = tf.constant([0, 1, 0], dtype=tf.int32)
num_segments = tf.constant(2, dtype=tf.int32)
example_output = call_func([data], segment_ids, num_segments)