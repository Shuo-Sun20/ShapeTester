import tensorflow as tf
import numpy as np

def call_func(inputs, segment_ids, num_segments, name=None):
    data = inputs[0] if isinstance(inputs, list) else inputs
    return tf.math.unsorted_segment_mean(data, segment_ids, num_segments, name)

# Generate valid random input data
data_tensor = tf.constant(np.random.randn(6, 3, 4).astype(np.float32))
segment_ids_tensor = tf.constant([0, 1, 0, 2, 1, 2], dtype=tf.int32)
num_segments_val = 4

# Call the function
example_output = call_func([data_tensor], segment_ids_tensor, num_segments_val)