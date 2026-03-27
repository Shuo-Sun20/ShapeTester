import tensorflow as tf

def call_func(inputs, dtype=tf.float32, name=None):
    dim, num_results, skip = inputs
    return tf.math.sobol_sample(dim, num_results, skip, dtype, name)

dim = tf.constant(3, dtype=tf.int32)
num_results = tf.constant(10, dtype=tf.int32)
skip = tf.constant(0, dtype=tf.int32)
example_output = call_func([dim, num_results, skip])