import tensorflow as tf

def call_func(inputs, num_segments, name=None):
    data, segment_ids = inputs
    return tf.math.unsorted_segment_max(data, segment_ids, num_segments, name=name)

data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
segment_ids = tf.constant([0, 1, 0])
num_segments = 2
example_output = call_func([data, segment_ids], num_segments)