import tensorflow as tf

def call_func(inputs, segment_ids, name=None):
    return tf.math.segment_sum(data=inputs, segment_ids=segment_ids, name=name)

data = tf.random.normal(shape=[7, 4], dtype=tf.float32)
segment_ids = tf.constant([0, 0, 1, 1, 2, 2, 2], dtype=tf.int32)
example_output = call_func(inputs=data, segment_ids=segment_ids)