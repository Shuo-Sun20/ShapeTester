import tensorflow as tf

def call_func(inputs, num_segments, name=None):
    data, segment_ids = inputs[0], inputs[1]
    return tf.math.unsorted_segment_min(data, segment_ids, num_segments, name)

# Generate random input data
tf.random.set_seed(42)
data = tf.random.uniform(shape=(5, 4), minval=0, maxval=10, dtype=tf.int32)
segment_ids = tf.constant([0, 1, 0, 2, 1], dtype=tf.int32)
num_segments = 3

# Call the function
example_output = call_func([data, segment_ids], num_segments)