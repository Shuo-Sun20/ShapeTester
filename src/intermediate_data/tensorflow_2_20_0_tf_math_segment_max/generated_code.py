import tensorflow as tf

def call_func(inputs, name=None):
    data, segment_ids = inputs
    return tf.math.segment_max(data, segment_ids, name)

data = tf.constant([[3, 7, 2], [9, 1, 5], [4, 8, 6]], dtype=tf.float32)
segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
example_output = call_func([data, segment_ids])