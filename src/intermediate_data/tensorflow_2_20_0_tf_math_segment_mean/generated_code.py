import tensorflow as tf

def call_func(inputs, segment_ids, name=None):
    return tf.math.segment_mean(data=inputs, segment_ids=segment_ids, name=name)

# Construct a valid input
tf.random.set_seed(42)
data = tf.random.uniform(shape=(6, 3), minval=0, maxval=10, dtype=tf.float32)
segment_ids = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)
example_output = call_func(inputs=data, segment_ids=segment_ids)