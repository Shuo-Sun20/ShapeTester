import tensorflow as tf

def call_func(inputs, name=None):
    a, x = inputs
    return tf.math.igamma(a, x, name)

a = tf.random.uniform((3, 4), minval=0.1, maxval=2.0, dtype=tf.float32)
x = tf.random.uniform((3, 4), minval=0.0, maxval=5.0, dtype=tf.float32)
example_output = call_func([a, x])