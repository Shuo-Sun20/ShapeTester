import tensorflow as tf

def call_func(inputs, num_lower, num_upper, name=None):
    return tf.linalg.band_part(inputs, num_lower, num_upper, name=name)

example_output = call_func(
    tf.random.normal(shape=(4, 4)), 
    tf.constant(1, dtype=tf.int64), 
    tf.constant(-1, dtype=tf.int64)
)