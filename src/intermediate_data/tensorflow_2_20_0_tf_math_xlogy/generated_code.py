import tensorflow as tf

def call_func(inputs, name=None):
    x, y = inputs
    return tf.math.xlogy(x, y, name=name)

x = tf.random.uniform(shape=(3, 3), dtype=tf.float32)
y = tf.random.uniform(shape=(3, 3), minval=0.1, dtype=tf.float32)
example_output = call_func([x, y])