import tensorflow as tf

def call_func(inputs, name=None):
    x, y = inputs
    return tf.math.xdivy(x, y, name=name)

x = tf.constant([1.0, 0.0, 2.0])
y = tf.constant([2.0, 1.0, 0.0])
example_output = call_func([x, y])