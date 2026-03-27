import tensorflow as tf

def call_func(inputs, segment_ids, num_segments, name=None):
    return tf.math.unsorted_segment_prod(data=inputs, segment_ids=segment_ids, num_segments=num_segments, name=name)

data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.int32)
segment_ids = tf.constant([0, 1, 0], dtype=tf.int32)
num_segments = tf.constant(2, dtype=tf.int32)
example_output = call_func(inputs=data, segment_ids=segment_ids, num_segments=num_segments)